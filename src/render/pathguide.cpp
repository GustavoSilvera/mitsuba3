#include <mitsuba/core/warp.h>
#include <mitsuba/render/pathguide.h>
#include <stack> // std::stack

NAMESPACE_BEGIN(mitsuba)

#define check(x)                                                               \
    if (!(x))                                                                  \
        throw std::runtime_error("Assertion failed on line " +                 \
                                 std::to_string(__LINE__));

MI_VARIANT
size_t PathGuide<Float, Spectrum>::Angles2Quadrant(const Point2f &pos) {
    // takes the 2D location input and returns the corresponding quadrant
    auto cpos = dr::clamp(pos, 0.f, 1.f); // within bounds

    if (dr::any_or<true>(cpos.x() < 0.5f && cpos.y() < 0.5f)) // top left
        return 0;                                             // (quadrant 0)
    else if (dr::any_or<true>(cpos.y() < 0.5f)) // must be top right
        return 1;                               // (quadrant 1)
    else if (dr::any_or<true>(cpos.x() < 0.5f)) // must be bottom left
        return 2;                               // (quadrant 2)
    return 3;                                   // (quadrant 3)
}

MI_VARIANT
typename PathGuide<Float, Spectrum>::Point2f
PathGuide<Float, Spectrum>::NormalizeForQuad(const Point2f &pos,
                                             const size_t quad) {
    Point2f ret = dr::clamp(pos, 0.f, 1.f); // within bounds
    check(quad <= 3);
    if (quad == 0) // top left (quadrant 0)
    {
        // do nothing! (already within [0,0.5] for both x and y)
    } else if (quad == 1)            // top right (quadrant 1)
        ret -= Point2f{ 0.5f, 0.f }; // map (x) [0.5, 1] -> [0, 0.5]
    else if (quad == 2)              // bottom left (quadrant 2)
        ret -= Point2f{ 0.f, 0.5f }; // map (y) [0.5, 1] -> [0, 0.5]
    else
        ret -= Point2f{ 0.5f, 0.5f }; // map (x & y) [0.5, 1] -> [0, 0.5]
    // ret should be within [0, 0.5]
    check(dr::any_or<true>(ret.x() >= -dr::Epsilon<Float> &&
                           ret.x() <= 0.5f + dr::Epsilon<Float> &&
                           ret.y() >= -dr::Epsilon<Float> &&
                           ret.y() <= 0.5f + dr::Epsilon<Float>));
    return 2.f * ret; // map [0, 0.5] -> [0, 1]
}

//-------------------DTreeWrapper-------------------//

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::add_sample(const Vector3f &dir,
                                                          const Float lum,
                                                          const Float weight) {
    auto &tree  = current;
    Point2f pos = warp::uniform_sphere_to_square(dir);
    tree.weight += weight;
    tree.sum += lum;

    if (tree.nodes.size() == 0)
        tree.nodes.resize(1); // ensure always have a root node!
    check(tree.nodes.size() >= 1);

    // update internal nodes
    auto *node = &(tree.nodes[0]); // root
    while (true) {
        check(node != nullptr);
        const size_t quad_idx = Angles2Quadrant(pos);
        pos                   = NormalizeForQuad(pos, quad_idx);
        node->data[quad_idx] += lum;
        if (node->bIsLeaf(quad_idx))
            break;
        size_t child_idx = node->children[quad_idx];
        node             = &(tree.nodes[child_idx]);
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::reset(const size_t max_depth,
                                                     const Float rho) {
    // clear and re-initialize the nodes
    current.nodes.clear();
    current.nodes.resize(1);
    current.max_depth = 0;
    current.weight    = 0;
    current.sum       = 0.f;
    struct StackItem {
        size_t node_idx;
        size_t other_idx;
        DTreeWrapper::DirTree *tree;
        size_t depth;
    };

    std::stack<StackItem> stack;
    stack.push({ 0, 0, &prev, 1 });

    const Float prev_sum = Float(prev.sum);
    while (!stack.empty()) {
        const StackItem s = stack.top();
        stack.pop();

        current.max_depth = std::max(current.max_depth, s.depth);
        check(s.tree != nullptr);
        check(s.other_idx < s.tree->nodes.size());
        const auto &other_node = s.tree->nodes[s.other_idx];
        for (size_t quad = 0; quad < other_node.data.size(); quad++) {
            const Float quad_sum = Float(other_node.data[quad]);
            if (s.depth < max_depth &&
                dr::any_or<true>(quad_sum > prev_sum * rho)) {
                // add child and check if parent
                const size_t child_idx = current.nodes.size();
                current.nodes.emplace_back(); // create the child!
                auto &new_node = current.nodes.back();

                if (!other_node.bIsLeaf(quad)) {
                    // must be because other node comes from last tree
                    check(s.tree == &prev);
                    stack.push({ child_idx, other_node.children[quad], s.tree,
                                 s.depth + 1 });
                } else {
                    // is a leaf, from c tree or the previous
                    stack.push({ child_idx, child_idx, &current, s.depth + 1 });
                }

                // ensure this child has a parent!
                current.nodes[s.node_idx].children[quad] = child_idx;

                // distribute evenly over 4 quads
                new_node.data_fill(quad_sum / 4.f);

                if (current.nodes.size() >
                    std::numeric_limits<uint32_t>::max()) {
                    Log(Error, "DTreeReset hit max children count!");
                    stack = std::stack<StackItem>();
                    break;
                }
            }
        }
    }

    // now set all the new energy to 0
    for (auto &node : current.nodes) {
        node.data_fill(0.f);
    }
}

MI_VARIANT void PathGuide<Float, Spectrum>::DTreeWrapper::build() {
    // must always have a root node!
    check(current.nodes.size() > 0);
    check(prev.nodes.size() > 0);
    // keep track of this tree as the last iteration's
    prev = current; // copy assignment, current is deepcopied to prev
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::DTreeWrapper::sample_pdf(
    const Vector3f &dir) const {
    const auto &tree = prev;

    // pdf starts out as 1/4pi (uniform across sphere)
    Float pdf = warp::square_to_uniform_sphere_pdf(dir);
    if (tree.nodes.size() == 0 ||
        dr::any_or<true>(Float(tree.weight) == 0.f || Float(tree.sum) == 0.f))
        return pdf;

    // begin recursing into nodes
    Point2f pos      = warp::uniform_sphere_to_square(dir);
    const auto *node = &(tree.nodes[0]); // start at root
    while (true) {
        check(node != nullptr);

        const size_t quad_idx = Angles2Quadrant(pos);
        pos                   = NormalizeForQuad(pos, quad_idx);

        const Float quad_samples = Float(node->data[quad_idx]);
        if (dr::any_or<true>(quad_samples <= 0.f))
            return 0.f; // invalid pdf

        // distribute this mean evenly for all "quads"
        // equivlaent to scaling by 4x since each quadrant has area 1/4
        pdf *= (node->data.size() * quad_samples) / node->sum();

        if (node->bIsLeaf(quad_idx))
            break;

        // traverse down the tree (until leaf is found)
        size_t child_idx = node->children[quad_idx];
        node             = &(tree.nodes[child_idx]);
    }
    return pdf;
}

MI_VARIANT
bool PathGuide<Float, Spectrum>::DTreeWrapper::DirTree::DirNode::sample(
    size_t &quadrant, Sampler<Float, Spectrum> *sampler) const {
    const Float top_left  = Float(data[0]); // atomic load
    const Float top_right = Float(data[1]); // atomic load
    const Float bot_left  = Float(data[2]); // atomic load
    const Float bot_right = Float(data[3]); // atomic load
    const Float total     = top_left + top_right + bot_left + bot_right;

    // just use unit random
    if (dr::any_or<true>(total == 0.f))
        return false;

    // NOTE: quadrants are indexed like this
    // ---------
    // | 0 | 1 |
    // ---------
    // | 2 | 3 |
    // ---------

    // sample the loaded die that is the weighted quadrants according to samples
    // can probably do something smarter, see
    // https://www.keithschwarz.com/darts-dice-coins/

    const Float sample = sampler->next_1d();
    if (dr::any_or<true>(sample < top_left / total)) // dice rolls top left
    {
        quadrant = 0;
    } else if (dr::any_or<true>(sample < (top_left + top_right) /
                                             total)) // dice rolls top right
    {
        quadrant = 1;
    } else if (dr::any_or<true>((sample < (top_left + top_right + bot_left) /
                                              total))) // dice rolls bottom left
    {
        quadrant = 2;
    } else // dice rolls bottom right
    {
        check(dr::any_or<true>(sample <= 1.f + dr::Epsilon<Float>));
        quadrant = 3;
    }
    check(quadrant <= 3); // 0, 1, 2, or 3
    return true;
}

MI_VARIANT void PathGuide<Float, Spectrum>::DTreeWrapper::free_memory() {
    // free both of the quad trees
    current.free();
    prev.free();
}

MI_VARIANT
typename PathGuide<Float, Spectrum>::Vector3f
PathGuide<Float, Spectrum>::DTreeWrapper::sample_dir(
    Sampler<Float, Spectrum> *sampler) const {
    const Point2f unit_random = sampler->next_2d();
    const auto &tree          = prev;

    // early out to indicate that this tree is invalid
    if (tree.nodes.size() == 0 ||
        dr::any_or<true>(Float(tree.weight) == 0 || Float(tree.sum) == 0.f))
        return warp::square_to_uniform_sphere(unit_random);

    // recurse into the tree
    Point2f pos{ 0.f, 0.f }; // center of cartesian plane (no leaning)
    float scale = 1.0f;      // halved on each (non-leaf) iteration

    size_t which_quadrant = 0;
    const auto *node      = &(tree.nodes[0]); // start at root
    while (true) {
        check(node != nullptr);

        if (!node->sample(which_quadrant, sampler)) // invalid!
            return warp::square_to_uniform_sphere(unit_random);
        check(which_quadrant <= 3);

        // use a "quadrant origin" to push sample in corresponding quadrant
        const Point2f quadrant_origin{
            0.5f * (which_quadrant % 2 == 1), // right side of y=0
            0.5f * (which_quadrant >= 2),     // underneath x=0
        };

        if (node->bIsLeaf(which_quadrant)) // hit a leaf
        {
            // add the initial random sample to this quadrant
            pos += scale * (quadrant_origin + 0.5f * unit_random);
            break;
        } else {
            // continue burrowing into this quadrant
            pos += scale * quadrant_origin;
            scale /= 2.f;
        }

        size_t child_idx = node->children[which_quadrant];

        // iterate down the tree
        node = &(tree.nodes[child_idx]);
    }

    return warp::square_to_uniform_sphere(pos);
}

//-------------------SpatialTree-------------------//
MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::begin_next_tree_iteration() {
    for (auto &node : nodes)
        if (node.bIsLeaf())
            node.dTree.build();
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::reset_leaves(size_t max_depth,
                                                           Float rho) {
    for (auto &node : nodes)
        if (node.bIsLeaf())
            node.dTree.reset(max_depth, rho);
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::refine(
    const Float sample_threshold) {
    // traverse dTree via DFS and refine (subdivide) those leaves that qualify
    std::stack<size_t> stack;
    stack.push(0); // root node index
    while (!stack.empty()) {
        size_t idx = stack.top();
        // we use the raw indices of the nodes (in the vector) rather than
        // storing a single pointer since these elements might get reallocated
        // as the vector resizes!
        stack.pop();

        // currently hit a leaf node, might want to subdivide it (refine)
        if (node(idx).bIsLeaf() &&
            dr::any_or<true>(node(idx).dTree.get_weight() > sample_threshold)) {
            // splits the parent in 2, potentially creating more children
            subdivide(idx); // not thread safe!
        }

        if (!node(idx).bIsLeaf()) // check *again* (subdivision would create
                                  // children)
        {
            // begin iterating through children
            for (const auto idx : node(idx).children)
                stack.push(idx);
        }
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::subdivide(const size_t idx) {
    // split the parent node in 2 to refine samples
    check(node(idx).bIsLeaf()); // has no children
    // using node(idx) rather than taking a pointer to nodes[idx] because
    // nodes will resize (thus potentially reallocate) which might invalidate
    // any pointers or references!
    const Float weight        = node(idx).dTree.get_weight();
    const size_t num_children = node(idx).children.size();
    nodes.resize(nodes.size() + num_children); // prepare for new children
    for (size_t i = 0; i < num_children; i++) {
        const size_t child_idx = nodes.size() - 2 + i;
        node(idx).children[i]  = child_idx; // assign the child to the parent
        SNode &child           = nodes[child_idx];
        child.dTree            = node(idx).dTree; // copy this node's dirtree
        child.dTree.set_weight(weight / 2.f);     // approx half the samples
        // "iterate through axes on every pass" (0 for x, 1 for y, 2 for z)
        child.xyz_axis = (node(idx).xyz_axis + 1) % 3;
    }
    node(idx).dTree.free_memory(); // reset this dTree to save memory
    check(!node(idx).bIsLeaf());   // definitely has children now
}

MI_VARIANT
const typename PathGuide<Float, Spectrum>::DTreeWrapper &
PathGuide<Float, Spectrum>::SpatialTree::get_direction_tree(
    const Point3f &pos, Vector3f *size) const {
    // find the leaf node that contains this position

    // use a position normalized [0 -> 1]^3 within this dTree's bbox
    Vector3f x = (pos - bounds.min) / bounds.extents();

    check(nodes.size() > 0); // need at least a root node!

    const float split = 0.5f; // decision boundary between left and right child
    size_t idx        = 0;    // start at root node, descent down tree
    while (!node(idx).bIsLeaf()) {
        const auto ax = node(idx).xyz_axis;
        check(ax <= 2); // x, y, z

        size_t child_idx = 0;                // assume going to child 0
        if (dr::any_or<true>(x[ax] > split)) // actually going to child 1
        {
            child_idx = 1;
            x[ax] -= split; // (0.5,1) -> (0,0.5)
        }
        x[ax] /= split; // re-normalize (0,0.5) -> (0,1)
        if (size != nullptr)
            (*size)[ax] /= 2.f;
        idx = node(idx).children[child_idx]; // go to next child
    }
    return node(idx).dTree;
}

//-------------------PathGuide-------------------//

MI_VARIANT void
PathGuide<Float, Spectrum>::initialize(const ScalarBoundingBox3f &bbox) {
    spatial_tree.bounds = bbox;
    refine(spatial_tree_thresh); // initial refinement/reset
}

MI_VARIANT
void PathGuide<Float, Spectrum>::refine(const Float thresh) {
    spatial_tree.refine(thresh);
    spatial_tree.reset_leaves(max_DTree_depth, rho);
}

MI_VARIANT void PathGuide<Float, Spectrum>::refine() {
    spatial_tree.begin_next_tree_iteration(); // keep track of last trees
    refinement_iter++;
    // next iter should have sqrt(2^n) times the threshold
    refine(dr::pow(2.f, (refinement_iter + 2) * 0.5f) * spatial_tree_thresh);
}

MI_VARIANT
void PathGuide<Float, Spectrum>::add_radiance(
    const Point3f &pos, const Vector3f &dir, const Float luminance,
    Sampler<Float, Spectrum> *sampler) {
    if (dr::any_or<true>(!dr::isfinite(luminance) || luminance < 0.f))
        return;
    Point3f newPos        = pos;
    const Float weight    = 0.25f;
    const Float pDoJitter = 0.5f; // probability of jittering the sample:
    Vector3f neigh_size(1, 1, 1);
    DTreeWrapper &exact_dir_tree = // without jitter
        spatial_tree.get_direction_tree(pos, &neigh_size);
    if (sampler != nullptr &&
        dr::any_or<true>(sampler->next_1d() > pDoJitter)) {
        { // perform stochastic filtering on spatial tree
            // jitter within bounding box of leaf node containing pos
            Vector3f offset = neigh_size;
            offset.x() *= sampler->next_1d() - 0.5f;
            offset.y() *= sampler->next_1d() - 0.5f;
            offset.z() *= sampler->next_1d() - 0.5f;
            newPos += offset;
            newPos = dr::minimum(newPos, spatial_tree.bounds.max);
            newPos = dr::maximum(newPos, spatial_tree.bounds.min);
        }
        // traverse again down the spatial tree, but include jitter
        DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(newPos);
        dir_tree.add_sample(dir, luminance, weight);
    } else {
        exact_dir_tree.add_sample(dir, luminance, weight);
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::add_throughput(const Point3f &pos,
                                                const Vector3f &dir,
                                                const Spectrum &result,
                                                const Spectrum &throughput,
                                                const Float woPdf) {
    /// NOTES:
    // *result* stores the sum of all radiance up to this point.
    // This includes a progressively accumulated *throughput* for
    // the path from the point to the sensor as well as summing all
    // the direct-connections from next-event-estimation
    // ---
    /// conceptually, if we think of NEE as having created V paths
    /// from each bounce
    // (light source -> eye) then *result* stores the sum of these V
    // paths' radiance while *throughput* only stores the immediate
    // radiance along the path to here
    Spectrum path_radiance = result; // how much radiance is flowing
                                     // through the path ending here
    if (thru_vars.size() > 0) {
        auto &[o, d, _, result_prev, T, woPdf] = thru_vars.back();
        // delta between result computes the lighting for this path
        path_radiance = result - result_prev;
    }
    thru_vars.emplace_back(pos, dir, path_radiance, result, throughput, woPdf);
}

MI_VARIANT
void PathGuide<Float, Spectrum>::calc_radiance_from_thru(
    Sampler<Float, Spectrum> *sampler) {
    /// NOTE:
    // at each bounce we track how much radiance we have seen so far,
    // and at the end we have the total radiance (including NEE) from
    // end to eye so we can subtract what we've seen. This will give us
    // the sum of the remaining NEE paths until the end (from the
    // beginning) but we want the incident radiance starting from this
    // bounce, so we then divide by the current throughput seen so far
    // to cancel out those terms.

    auto lum = [](const Spectrum &spec) {
        if constexpr (is_rgb_v<Spectrum>) {
            return luminance(spec);
        } else if constexpr (is_monochromatic_v<Spectrum>) {
            return spec[0];
        } else {
            return dr::mean(spec);
        }
    };

    bool final_found        = false;
    Spectrum final_radiance = 0.f;
    for (auto rev = thru_vars.rbegin(); rev != thru_vars.rend(); rev++) {
        // add indirect lighting, o/w pathguide strongly prefers direct
        const auto &[o, d, path_radiance, _, thru, woPdf] = (*rev);

        if (!final_found && dr::any_or<true>(lum(path_radiance) > 0.f)) {
            // once the latest path-radiance is computed (last non-zero
            // path-radiance) use this path radiance for the indirect
            // lighting of all previous bounces
            final_radiance = path_radiance;
            final_found    = true;
            continue; // don't record this bounce (direct illumination)
        }
        const Spectrum radiance = (final_radiance / thru) / woPdf;
        this->add_radiance(o, d, lum(radiance), sampler);
    }
    thru_vars.clear();
}

MI_VARIANT
std::pair<typename PathGuide<Float, Spectrum>::Vector3f, Float>
PathGuide<Float, Spectrum>::sample(const Vector3f &pos,
                                   Sampler<Float, Spectrum> *sampler) const {
    // logarithmic complexity to traverse both trees (spatial & directional)
    const DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    const Vector3f wo            = dir_tree.sample_dir(sampler);
    const Float pdf              = dir_tree.sample_pdf(wo);
    return { wo, pdf };
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::sample_pdf(const Point3f &pos,
                                             const Vector3f &dir) const {
    const DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    return dir_tree.sample_pdf(dir);
}

// use this to declare the class with template instantiation while in .cpp
// https://stackoverflow.com/questions/1639797/template-issue-causes-linker-error-c
MI_INSTANTIATE_CLASS(PathGuide)

NAMESPACE_END(mitsuba)
