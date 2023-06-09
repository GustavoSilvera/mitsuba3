#include <limits> // std::numeric_limits
#include <mitsuba/core/warp.h>
#include <mitsuba/render/pathguide.h>
#include <stack> // std::stack

NAMESPACE_BEGIN(mitsuba)

//-------------------DTreeWrapper-------------------//

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::add_sample(const Vector3f &dir,
                                                          const Float lum,
                                                          const Float weight) {
    auto &tree = current; // only adding samples to the current (building) tree
    tree.weight += weight;
    if (dr::all(lum == 0.f))
        return; // no need to add 0 luminance to all the tree nodes
    tree.sum += lum;

    // should always have a root node!
    Assert(tree.nodes.size() >= 1);

    // update internal nodes
    Point2f pos = warp::uniform_sphere_to_square(dir);
    tree.add_lum_helper(0, pos, lum); // start at root and recurse
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::DTreeWrapper::sample_pdf(
    const Vector3f &dir) const {
    const auto &tree = prev;

    // pdf starts out as 1/4pi (uniform across sphere)
    Float pdf = warp::square_to_uniform_sphere_pdf(dir);
    if (tree.nodes.size() == 0 ||
        dr::all(Float(tree.weight) == 0.f || Float(tree.sum) == 0.f))
        return pdf;

    // begin recursing into nodes
    Point2f pos = warp::uniform_sphere_to_square(dir);
    pdf *= tree.get_pdf_helper(0, pos);
    return pdf;
}

MI_VARIANT
uint8_t PathGuide<Float, Spectrum>::DTreeWrapper::get_child_idx(Point2f &p) {
    /**
     * \brief Return the corresponding quad index for p in and re-normalize it
     *
     * Takes a 2D point that lies within a quad (separated between {0, 0.5, 1}
     * for x and y) and returns the quad. Also renormalizes the Point2f and
     * to fit within (0, 1)^2 so it can be used again further down the
     * (direction) tree. The quad index (between 0, 1, 2, 3) is used for
     * indexing into the children array for the nodes of the direction tree.
     *
     * Quadrants are indexed like this:
     *  0.......1
     *  ---------  0
     *  | 0 | 1 |  .
     *  ---------  .
     *  | 2 | 3 |  .
     *  ---------  1
     */
    uint8_t quad = 0;
    // Point2f has 2 dimensions (by design)
    for (uint8_t i = 0; i < 2; i++) {
        if (dr::all(p[i] < 0.5f)) {
            p[i] *= 2.f; // (0, 0.5) -> (0, 1)
            // quad |= 0 << i; // no-op
        } else {
            p[i] = (p[i] - 0.5f) * 2.f; // (0.5, 1) -> (0, 1)
            quad |= 1 << i;
        }
    }
    return quad;
}

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::reset(const uint32_t max_depth,
                                                     const Float rho) {
    // clear and re-initialize the nodes
    current.nodes.clear();
    current.nodes.resize(1); // ensure a root node is present
    current.weight = 0;
    current.sum    = 0.f;
    struct StackItem {
        uint32_t node_idx;                 // index of node in current tree
        uint32_t source_idx;               // index of node in source tree
        DTreeWrapper::DirectionTree *tree; // source tree
        uint32_t depth = 0;                // recursion depth
    };

    std::stack<StackItem> stack;
    stack.push({ 0, 0, &prev, 1 });

    const Float prev_sum = Float(prev.sum);
    while (!stack.empty()) {
        const StackItem s = stack.top();
        stack.pop();

        Assert(s.tree != nullptr);
        Assert(s.source_idx < s.tree->nodes.size());
        // always index into the the nodes array rather than taking a
        // pointer/ref because it may get reallocated when creating new children
        auto source_node = [s]() { return s.tree->nodes[s.source_idx]; };
        for (uint32_t quad = 0; quad < 4; quad++) {
            const Float quad_sum = Float(source_node().data[quad]);
            if (s.depth < max_depth && dr::all(quad_sum > prev_sum * rho)) {
                // prepare for adding a new child
                const uint32_t child = current.nodes.size();
                // push the child onto the stack if the quadrant is not a leaf
                // of the source node. The source node cannot be a leaf iff it
                // comes from the previous tree.

                if (!source_node().bIsLeaf(quad)) {
                    Assert(s.tree == &prev);
                    uint32_t other = source_node().children[quad];
                    stack.push({ child, other, s.tree, s.depth + 1 });
                } else {
                    stack.push({ child, child, &current, s.depth + 1 });
                }

                // ensure the child has a parent assigned to it
                current.nodes[s.node_idx].children[quad] = current.nodes.size();
                // create the child (possibly reallocating the array)
                current.nodes.emplace_back();
                // distribute the parent's sum evenly over the child's 4 quads
                for (auto &quad_data : current.nodes[child].data)
                    quad_data = quad_sum / 4.f;

                if (current.nodes.size() > std::numeric_limits<int>::max()) {
                    Log(Error, "DTreeWrapper::reset hit max children count!");
                    stack = std::stack<StackItem>();
                    break;
                }
            }
        }
    }

    // now set all the new energy to 0
    for (auto &node : current.nodes) {
        for (auto &quad_data : node.data)
            quad_data = 0.f;
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
typename PathGuide<Float, Spectrum>::Vector3f
PathGuide<Float, Spectrum>::DTreeWrapper::sample_dir(Point2f &sample) const {
    const auto &tree = prev;

    // early out to indicate that this tree is invalid
    if (tree.nodes.size() == 0 ||
        dr::all(Float(tree.weight) == 0 || Float(tree.sum) == 0.f))
        return warp::square_to_uniform_sphere(sample);

    // recurse into the tree
    Point2f pos{ 0.f, 0.f }; // center of cartesian plane (no leaning)
    float scale = 1.0f;      // halved on each (non-leaf) iteration

    uint32_t which_quadrant = 0;
    uint32_t index          = 0;
    do {
        const auto &node = tree.nodes[index];

        if (!node.sample(which_quadrant, sample.x()))
            return warp::square_to_uniform_sphere(sample); // invalid!
        Assert(which_quadrant <= 3);

        // use a "quadrant origin" to push sample in corresponding quadrant
        const Point2f quadrant_origin{
            0.5f * (which_quadrant % 2 == 1), // right side of y=0
            0.5f * (which_quadrant >= 2),     // underneath x=0
        };

        if (node.bIsLeaf(which_quadrant)) // hit a leaf
        {
            // add the initial random sample to this quadrant
            pos += scale * (quadrant_origin + 0.5f * sample);
        } else {
            // continue burrowing into this quadrant
            pos += scale * quadrant_origin;
            scale /= 2.f;
        }

        // iterate down the tree
        index = node.children[which_quadrant];
    } while (index != 0); // index = 0 => leaf node (or root)

    return warp::square_to_uniform_sphere(pos);
}

//---------------------DirectionTree---------------------//

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::DirectionTree::add_lum_helper(
    const uint32_t node_idx, Point2f &pos, const Float lum) {
    auto &node = nodes[node_idx];

    uint8_t quad_idx = get_child_idx(pos);

    // add luminance to every node down this path
    node.data[quad_idx] += lum; // see QuantizedAtomicFloatAccumulator

    // continue traversing down the tree until leaf node
    if (!node.bIsLeaf(quad_idx))
        add_lum_helper(node.children[quad_idx], pos, lum);
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::DTreeWrapper::DirectionTree::get_pdf_helper(
    const uint32_t node_idx, Point2f &pos) const {
    const auto &node = nodes[node_idx];

    uint8_t quad_idx = get_child_idx(pos);

    const Float quad_i = Float(node.data[quad_idx]);
    if (dr::all(quad_i <= 0.f))
        return 0.f; // invalid pdf

    // compute the total sum of the 4 quads
    Float total = 0.f;
    for (const auto &q : node.data)
        total += Float(q);

    // distribute evenly for Angular domain split into 4 quadrants
    // (unit square) each with 1/4 unit area.
    Float p = ((4.f * quad_i) / total); // contribution of this node

    if (node.bIsLeaf(quad_idx))
        return p;

    // recurse down the tree
    return p * get_pdf_helper(node.children[quad_idx], pos);
}

//---------------------DTreeNode---------------------//

MI_VARIANT
bool PathGuide<Float, Spectrum>::DTreeWrapper::DirectionTree::DTreeNode::sample(
    uint32_t &quad, Float &r1) const {
    // r1 should be a random sample within (0, 1) that will get re-normalized
    // during this sampling process (to avoid drawing new samples)
    const Float top_L = Float(data[0]);
    const Float top_R = Float(data[1]);
    const Float bot_L = Float(data[2]);
    const Float bot_R = Float(data[3]);
    const Float total = top_L + top_R + bot_L + bot_R;

    // just use unit random, no data to sample from yet!
    if (dr::all(total == 0.f))
        return false; // fail to sample

    // sample the loaded die that is the weighted quadrants according to a
    // discrete distribution from the data. Can probably also investigate
    // https://www.keithschwarz.com/darts-dice-coins/

    // roll a dice from 0 to total and see where it lands in relation to the
    // boundaries set by the data
    r1 = dr::clamp(r1, dr::Epsilon<Float>, dr::OneMinusEpsilon<Float>);
    const Float sample = r1 * total;
    if (dr::all(sample < top_L)) {
        quad = 0; // dice rolls top left
        r1   = sample / top_L;
    } else if (dr::all(sample < top_L + top_R)) {
        quad = 1; // dice rolls top right
        r1   = (sample - top_L) / top_R;
    } else if (dr::all((sample < top_L + top_R + bot_L))) {
        quad = 2; // dice rolls bottom left
        r1   = (sample - (top_L + top_R)) / bot_L;
    } else {
        quad = 3; // dice rolls bottom right
        r1   = (sample - (top_L + top_R + bot_L)) / bot_R;
    }
    r1 = dr::clamp(r1, dr::Epsilon<Float>, dr::OneMinusEpsilon<Float>);
    Assert(quad <= 3); // 0, 1, 2, or 3
    return true;
}

//-------------------SpatialTree-------------------//

MI_VARIANT
PathGuide<Float, Spectrum>::SpatialTree::SpatialTree() {
    struct STreeNode root;
    root.dTree = std::make_unique<DTreeWrapper>();
    nodes.push_back(std::move(root));
    /// TODO: find a better initial allocation amnt based on the scene
    /// complexity or perhaps based on the number of children that are created
    /// on each refinement pass? Doesn't seem to affect performance much.
    nodes.reserve(200);
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::prepare_for_refinement() {
    /// TODO: parallelize?
    for (auto &node : nodes) {
        // valid dTree (unique) pointers should only occur on leaf nodes
        if (node.bIsLeaf()) {
            Assert(node.dTree);
            node.dTree->build(); // build the previous tree from the current one
        } else {
            Assert(!node.dTree);
        }
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::reset_leaves(uint32_t max_depth,
                                                           Float rho) {
    /// TODO: parallelize?
    Assert(nodes.size() % 2 == 1); // should have odd #nodes (binary tree)
    for (auto &node : nodes) {
        // valid dTree (unique) pointers should only occur on leaf nodes
        if (node.bIsLeaf()) {
            Assert(node.dTree);
            node.dTree->reset(max_depth, rho);
        } else {
            Assert(!node.dTree);
        }
    }
}

MI_VARIANT
void PathGuide<Float, Spectrum>::SpatialTree::refine(const Float threshold) {
    // traverse dTree via DFS and refine (subdivide) those leaves that surpass
    // the weight threshold. Note this method is NOT thread-safe since it may
    // reallocate the entire nodes vector (adding children)
    std::stack<uint32_t> stack;
    stack.push(0); // root node index
    while (!stack.empty()) {
        uint32_t idx = stack.top();
        // we use the raw indices of the nodes (in the vector) rather than
        // storing a single pointer since these elements might get reallocated
        // as the vector resizes!
        stack.pop();

        // currently hit a leaf node, might want to subdivide it (refine)
        if (nodes[idx].bIsLeaf() &&
            dr::all(nodes[idx].dTree->get_weight() > threshold)) {
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
void PathGuide<Float, Spectrum>::SpatialTree::subdivide(const uint32_t idx) {
    auto parent = [this, idx]() -> STreeNode & { return nodes[idx]; };
    // using this lambda for getting the "parent" by indexing directly into
    // nodes[idx] rather than taking a pointer to nodes[idx] because the nodes
    // array will resize (and potentially reallocate) which might invalidate
    // any pointers or references!
    Assert(parent().bIsLeaf()); // has no children
    Assert(parent().dTree);     // should have a valid leaf node
    DTreeWrapper *parent_tree   = parent().dTree.release();
    const Float weight          = parent_tree->get_weight();
    const uint32_t num_children = 2;
    Assert(parent().children.size() == num_children); // binary tree
    // resize (potentially reallocate) to make room for new children
    nodes.resize(nodes.size() + num_children); // prepare for new children
    for (uint32_t i = 0; i < num_children; i++) {
        const uint32_t child_idx = nodes.size() - num_children + i;
        parent().children[i]     = child_idx; // assign child
        // child copies the parent's dTree with half the weight and next axis
        struct STreeNode &child = nodes[child_idx];
        child.dTree             = std::make_unique<DTreeWrapper>(*parent_tree);
        child.dTree->set_weight(weight / 2.f); // approx half the samples
        // "iterate through axes on every pass" (0 for x, 1 for y, 2 for z)
        child.xyz_axis = (parent().xyz_axis + 1) % 3;
    }
    delete parent_tree;          // save memory, only allocate on leaf nodes
    Assert(!parent().dTree);     // should be invalid bc not a leaf
    Assert(!parent().bIsLeaf()); // definitely has children now
}

MI_VARIANT
const typename PathGuide<Float, Spectrum>::DTreeWrapper &
PathGuide<Float, Spectrum>::SpatialTree::get_leaf(const Point3f &pos,
                                                  Vector3f *size) const {

    // use a position normalized [0 -> 1]^3 within this dTree's bbox
    Vector3f x = (pos - m_bounds.min) / m_bounds.extents();

    Assert(nodes.size() > 0); // need at least a root node!

    uint32_t idx = 0; // start at root node, descent down tree
    while (!nodes[idx].bIsLeaf()) {
        const uint8_t ax = nodes[idx].xyz_axis;
        Assert(ax <= 2); // x, y, z

        uint32_t child_idx = 0;    // assume going to child 0
        if (dr::all(x[ax] > 0.5f)) // actually going to child 1
        {
            child_idx = 1;
            x[ax] -= 0.5f; // 0.5,1) -> (0,0.5)
        }
        x[ax] /= 0.5f; // re-normalize (0,0.5) -> (0,1)
        if (size != nullptr)
            (*size)[ax] /= 2.f;
        idx = nodes[idx].children[child_idx]; // go to next child
    }
    Assert(nodes[idx].dTree); // should be valid
    const DTreeWrapper &dTree = (*nodes[idx].dTree.get());
    return dTree;
}

//-------------------PathGuide-------------------//

MI_VARIANT
PathGuide<Float, Spectrum>::PathGuide(const float training_budget,
                                      const float p_jitter)
    : m_training_budget(training_budget), m_jitter_prob(p_jitter) {
    if (m_training_budget < 0.f)
        Log(Warn, "Path guide cannot train for negative samples. Disabling "
                  "path guider.");
    if (m_training_budget >= 1.f) {
        Log(Warn, "Using entirety of sampling budget for training the path "
                  "guider. This means none of the samples will be used for "
                  "(inference) rendering the final image!");
    }
}

MI_VARIANT void
PathGuide<Float, Spectrum>::initialize(const uint32_t scene_spp,
                                       const ScalarBoundingBox3f &bbox) {
    // calculate the number of refinement operations to perform (each one with
    // 2x spp of before) to approximately match the training threshold
    total_train_spp = static_cast<uint32_t>(scene_spp * m_training_budget);
    // number of iterations ("render passes") where spp is doubled for training
    num_training_refinements = dr::log2i(total_train_spp);
    // any overflow from the desired training budget that will be included in
    // the final training pass (see get_pass_spp(uint32_t))
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
        const uint32_t final_spp = dr::pow(2, num_training_refinements - 1);
        if (spp_overflow < final_spp) { // if the overflow is large enough
            // append any overflow to the final pass
            spp_overflow += final_spp;
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
    spatial_tree.m_bounds = bbox;
    refine(STree_thresh);
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
    spatial_tree.reset_leaves(DTree_maxdepth, rho);
}

MI_VARIANT void PathGuide<Float, Spectrum>::perform_refinement() {
    // performs one refinement iteration. This method should be called at the
    // end of each training pass since it is not thread safe
    spatial_tree.prepare_for_refinement();
    refinement_iter++;

    /**
     * From [1]: "The decision whether to split is driven only by the number of
     * path vertices that were recorded in the volume of the node in the
     * previous iteration... Specifically, we split a node if there have been at
     * least c * sqrt(2^k) path vertices, where 2^k is proportional to the
     * amount of traced paths in the k-th iteration and c is derived from the
     * resolution of the quadtrees" Therefore each subsequent pass needs to
     * contain roughly c * sqrt(2^k) path vertices where c = STree_thresh,
     * and k = refinement_iter
     */
    refine(dr::sqrt(dr::pow(2.f, refinement_iter)) * STree_thresh);
}

MI_VARIANT
void PathGuide<Float, Spectrum>::add_radiance(
    const Point3f &pos, const Vector3f &dir, const Float luminance,
    Sampler<Float, Spectrum> *sampler) {
    if (dr::all(!dr::isfinite(luminance) || luminance < 0.f))
        return;
    Point3f newPos     = pos;
    const Float weight = 0.25f;
    Vector3f neigh_size(1, 1, 1);
    // get exact leaf (direction tree)
    DTreeWrapper &exact_dTree = spatial_tree.get_leaf(pos, &neigh_size);
    if (sampler != nullptr && dr::all(sampler->next_1d() < m_jitter_prob)) {
        // perform stochastic filtering on spatial tree by jittering within
        // bounding box of leaf node containing pos
        Vector3f offset = neigh_size;
        offset.x() *= sampler->next_1d() - 0.5f;
        offset.y() *= sampler->next_1d() - 0.5f;
        offset.z() *= sampler->next_1d() - 0.5f;
        newPos += offset;
        // ensure the new position still lies within the tree bounds
        newPos = dr::minimum(newPos, spatial_tree.m_bounds.max);
        newPos = dr::maximum(newPos, spatial_tree.m_bounds.min);
        // traverse again down the spatial tree, but include jitter
        DTreeWrapper &jittered_dTree = spatial_tree.get_leaf(newPos);
        jittered_dTree.add_sample(dir, luminance, weight);
    } else {
        exact_dTree.add_sample(dir, luminance, weight);
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

        if (!final_found && dr::all(lum(path_radiance) > 0.f)) {
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
    const uint32_t spp_done = (++atomic_spp_count); // atomic incr and fetch
    // should have 100 updates (update the progress bar every 1%)
    // (these are static since the values can be cached)
    const static uint32_t total_spp   = total_train_spp * screensize;
    const static uint32_t update_iter = total_spp / 100;
    // update a maximum of 100 times
    if (update_iter == 0 || spp_done % update_iter == 0) {
        // update the progress meter (0.f to 1.f)
        progress->update(static_cast<float>(spp_done) / total_spp);
    }
}

MI_VARIANT
std::pair<typename PathGuide<Float, Spectrum>::Vector3f, Float>
PathGuide<Float, Spectrum>::sample(const Vector3f &pos, Point2f sample) const {
    // logarithmic complexity to traverse both trees (spatial & directional)
    const DTreeWrapper &dTree = spatial_tree.get_leaf(pos);
    const Vector3f wo         = dTree.sample_dir(sample);
    const Float pdf           = dTree.sample_pdf(wo);
    return { wo, pdf };
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::sample_pdf(const Point3f &pos,
                                             const Vector3f &dir) const {
    const DTreeWrapper &dTree = spatial_tree.get_leaf(pos);
    return dTree.sample_pdf(dir);
}

MI_IMPLEMENT_CLASS_VARIANT(PathGuide, Object, "pathguide")
MI_INSTANTIATE_CLASS(PathGuide)

NAMESPACE_END(mitsuba)
