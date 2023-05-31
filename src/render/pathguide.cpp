#include <limits> // std::numeric_limits
#include <mitsuba/core/warp.h>
#include <mitsuba/render/pathguide.h>
#include <stack> // std::stack

NAMESPACE_BEGIN(mitsuba)

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
    Assert(quad <= 3);
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
    Assert(dr::any_or<true>(ret.x() >= 0.0f - dr::Epsilon<Float> &&
                            ret.x() <= 0.5f + dr::Epsilon<Float> &&
                            ret.y() >= 0.0f - dr::Epsilon<Float> &&
                            ret.y() <= 0.5f + dr::Epsilon<Float>));
    return 2.f * ret; // map [0, 0.5] -> [0, 1]
}

//-------------------DTreeWrapper-------------------//

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::add_sample(const Vector3f &dir,
                                                          const Float lum,
                                                          const Float weight) {
    auto &tree  = current; // only adding samples to the current (building) tree
    Point2f pos = warp::uniform_sphere_to_square(dir);
    tree.weight += weight;
    tree.sum += lum;

    // should always have a root node!
    Assert(tree.nodes.size() >= 1);

    // update internal nodes
    auto *node = &(tree.nodes[0]); // root
    while (true) {
        Assert(node != nullptr);
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
        Assert(s.tree != nullptr);
        Assert(s.other_idx < s.tree->nodes.size());
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
                    Assert(s.tree == &prev);
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
    Assert(current.nodes.size() > 0);
    Assert(prev.nodes.size() > 0);
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
        Assert(node != nullptr);

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
    const Float top_L = Float(data[0]); // atomic load quadrants
    const Float top_R = Float(data[1]); // atomic load quadrants
    const Float bot_L = Float(data[2]); // atomic load quadrants
    const Float bot_R = Float(data[3]); // atomic load quadrants
    const Float total = top_L + top_R + bot_L + bot_R;

    // just use unit random, no data to sample from yet!
    if (dr::any_or<true>(total == 0.f))
        return false; // fail to sample

    // NOTE: quadrants are indexed like this
    // ---------
    // | 0 | 1 |
    // ---------
    // | 2 | 3 |
    // ---------

    // sample the loaded die that is the weighted quadrants according to a
    // discrete distribution from the data. Can probably also investigate
    // https://www.keithschwarz.com/darts-dice-coins/

    // roll a dice from 0 to total and see where it lands in relation to the
    // boundaries set by the data
    const Float sample = sampler->next_1d() * total;
    if (dr::any_or<true>(sample < top_L)) {
        // dice rolls top left
        quadrant = 0;
    } else if (dr::any_or<true>(sample < top_L + top_R)) {
        // dice rolls top right
        quadrant = 1;
    } else if (dr::any_or<true>((sample < top_L + top_R + bot_L))) {
        // dice rolls bottom left
        quadrant = 2;
    } else {
        // dice rolls bottom right
        quadrant = 3;
    }
    Assert(quadrant <= 3); // 0, 1, 2, or 3
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
        Assert(node != nullptr);

        if (!node->sample(which_quadrant, sampler)) // invalid!
            return warp::square_to_uniform_sphere(unit_random);
        Assert(which_quadrant <= 3);

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
void PathGuide<Float, Spectrum>::SpatialTree::refine(const Float threshold) {
    // traverse dTree via DFS and refine (subdivide) those leaves that surpass
    // the weight threshold. Note this method is NOT thread-safe since it may
    // reallocate the entire nodes vector (adding children)
    std::stack<size_t> stack;
    stack.push(0); // root node index
    while (!stack.empty()) {
        size_t idx = stack.top();
        // we use the raw indices of the nodes (in the vector) rather than
        // storing a single pointer since these elements might get reallocated
        // as the vector resizes!
        stack.pop();

        // currently hit a leaf node, might want to subdivide it (refine)
        if (nodes[idx].bIsLeaf() &&
            dr::any_or<true>(nodes[idx].dTree.get_weight() > threshold)) {
            // splits the parent in 2, potentially creating more children
            subdivide(idx); // not thread safe!
        }

        // recursive through children if needed
        if (!nodes[idx].bIsLeaf()) // check *again* for new children
        {
            for (const auto idx : nodes[idx].children)
                stack.push(idx); // add children to the stack
        }
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::subdivide(const size_t idx) {
    // split the parent node in 2 to refine samples
    Assert(nodes[idx].bIsLeaf()); // has no children
    // using nodes[idx] rather than taking a pointer to nodes[idx] because
    // nodes will resize (thus potentially reallocate) which might invalidate
    // any pointers or references!
    const Float weight        = nodes[idx].dTree.get_weight();
    const size_t num_children = nodes[idx].children.size();
    nodes.resize(nodes.size() + num_children); // prepare for new children
    for (size_t i = 0; i < num_children; i++) {
        const size_t child_idx = nodes.size() - 2 + i;
        nodes[idx].children[i] = child_idx; // assign the child to the parent
        SNode &child           = nodes[child_idx];
        child.dTree            = nodes[idx].dTree; // copy this node's dirtree
        child.dTree.set_weight(weight / 2.f);      // approx half the samples
        // "iterate through axes on every pass" (0 for x, 1 for y, 2 for z)
        child.xyz_axis = (nodes[idx].xyz_axis + 1) % 3;
    }
    nodes[idx].dTree.free_memory(); // reset this dTree to save memory
    Assert(!nodes[idx].bIsLeaf());  // definitely has children now
}

MI_VARIANT
const typename PathGuide<Float, Spectrum>::DTreeWrapper &
PathGuide<Float, Spectrum>::SpatialTree::get_direction_tree(
    const Point3f &pos, Vector3f *size) const {
    // find the leaf node that contains this position

    // use a position normalized [0 -> 1]^3 within this dTree's bbox
    Vector3f x = (pos - bounds.min) / bounds.extents();

    Assert(nodes.size() > 0); // need at least a root node!

    const float split = 0.5f; // decision boundary between left and right child
    size_t idx        = 0;    // start at root node, descent down tree
    while (!nodes[idx].bIsLeaf()) {
        const auto ax = nodes[idx].xyz_axis;
        Assert(ax <= 2); // x, y, z

        size_t child_idx = 0;                // assume going to child 0
        if (dr::any_or<true>(x[ax] > split)) // actually going to child 1
        {
            child_idx = 1;
            x[ax] -= split; // (0.5,1) -> (0,0.5)
        }
        x[ax] /= split; // re-normalize (0,0.5) -> (0,1)
        if (size != nullptr)
            (*size)[ax] /= 2.f;
        idx = nodes[idx].children[child_idx]; // go to next child
    }
    return nodes[idx].dTree;
}

//-------------------PathGuide-------------------//

MI_VARIANT void
PathGuide<Float, Spectrum>::initialize(const uint32_t scene_spp,
                                       const ScalarBoundingBox3f &bbox) {
    // calculate the number of refinement operations to perform (each one with
    // 2x spp of before) to approximately match the training threshold
    total_train_spp = static_cast<size_t>(scene_spp * training_budget);
    // number of iterations ("render passes") where spp is doubled for training
    num_training_refinements = dr::log2i(total_train_spp);
    // any overflow from the desired training budget that will be included in
    // the final training pass (see get_pass_spp(size_t))
    spp_overflow = total_train_spp - dr::pow(2, num_training_refinements);
    if (spp_overflow > 0) {
        // total_train_spp cant perfectly fit in the geometric series (not a
        // power of 2), so we have two options: either tack on the overflow to
        // the end (final pass) or run an extra pass with only the overflow. A
        // fine heuristic for this is whether or not the overflow is larger than
        // the spp in the (geometric) final pass. As this should be enough to
        // constitute its own pass. We want to avoid rendering a new pass with a
        // very small spp at the end if possible (which harms the learned
        // distribution approximation)
        const size_t final_pass_spp = dr::pow(2, num_training_refinements - 1);
        if (spp_overflow < final_pass_spp) { // if the overflow is large enough
            // append any overflow to the final pass
            spp_overflow += final_pass_spp;
        } else {
            // include another pass for the overflow
            num_training_refinements++;
        }
    }

    if (num_training_refinements == 0) {
        Log(Warn,
            "Calculated maximum number of refinements is 0. Training budget "
            "is too low, this will result in an ineffective path guider "
            "with potentially higher variance than BSDF sampling.");
    }
    spatial_tree.bounds = bbox;
    refine(spatial_tree_thresh);
}

MI_VARIANT uint32_t
PathGuide<Float, Spectrum>::get_pass_spp(const uint32_t pass_idx) const {
    // geometrically increasing => double spp on each iteration
    uint32_t spp = dr::pow(2, pass_idx);
    // if we need to include some overflow in the final pass
    if (spp_overflow && pass_idx == num_training_refinements - 1) {
        return spp_overflow;
    }
    if (pass_idx >= num_training_refinements) {
        Log(Warn,
            "Path guider has reached past the expected number of samples for "
            "training (%d). Returning 0 spp for this pass (%d)",
            num_training_refinements, pass_idx);
        return 0; // should be done training!
    }
    return spp;
}

MI_VARIANT
void PathGuide<Float, Spectrum>::refine(const Float thresh) {
    spatial_tree.refine(thresh);
    spatial_tree.reset_leaves(max_DTree_depth, rho);
}

MI_VARIANT void PathGuide<Float, Spectrum>::perform_refinement() {
    // performs one refinement iteration. This method should be called at the
    // end of each training pass since it is not thread safe
    spatial_tree.begin_next_tree_iteration(); // keep track of last trees
    refinement_iter++;                        // not atomic!
    // "The decision whether to split is driven only by the number of path
    // vertices that were recorded in the volume of the node in the previous
    // iteration... Specifically, we split a node if there have been at least c
    // * sqrt(2^k) path vertices, where 2^k is proportional to the amount of
    // traced paths in the k-th iteration and c is derived from the resolution
    // of the quadtrees"
    // Therefore each subsequent pass needs to contain roughly c * sqrt(2^k)
    // path vertices where c = spatial_tree_thresh, and k = refinement_iter
    refine(dr::sqrt(dr::pow(2.f, refinement_iter)) * spatial_tree_thresh);
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
        // calculate radiance from this bounce to the light source
        const Spectrum radiance   = final_radiance / thru;
        const Spectrum irradiance = radiance / woPdf;
        this->add_radiance(o, d, lum(irradiance), sampler);
    }
    thru_vars.clear();
    update_progress();
}

MI_VARIANT
void PathGuide<Float, Spectrum>::update_progress() {
    if (progress == nullptr)
        return;
    const size_t spp_done = (++atomic_spp_count); // atomic incr and fetch
    // should have 100 updates (update the progress bar every 1%)
    // (these are static since the values can be cached)
    const static size_t total_spp   = total_train_spp * screensize;
    const static size_t update_iter = total_spp / 100;
    // update a maximum of 100 times
    if (update_iter == 0 || spp_done % update_iter == 0) {
        // update the progress meter (0.f to 1.f)
        progress->update(static_cast<float>(spp_done) / total_spp);
    }
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

MI_IMPLEMENT_CLASS_VARIANT(PathGuide, Object, "pathguide")
MI_INSTANTIATE_CLASS(PathGuide)

NAMESPACE_END(mitsuba)
