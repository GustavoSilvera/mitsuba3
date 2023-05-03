#include <mitsuba/core/warp.h>
#include <mitsuba/render/pathguide.h>
#include <stack> // std::stack

NAMESPACE_BEGIN(mitsuba)

#define check(x)                                                               \
    if (!(x))                                                                  \
        throw std::runtime_error("Assertion failed on line " +                 \
                                 std::to_string(__LINE__));

// utility method to go from Vector3f (roll, pitch, yaw) -> Vector2f (theta,
// phi)
MI_VARIANT
typename PathGuide<Float, Spectrum>::Vector2f
PathGuide<Float, Spectrum>::Euler2Angles(const Vector3f &dir) {
    // Point3f p(dir.x(), dir.y(), dir.z());
    auto p2 = warp::uniform_sphere_to_square(dir);
    return Vector2f(p2.x(), p2.y());
}

// utility method to go from Vector2f (theta, phi) -> Vector3f (roll, pitch,
// yaw)
MI_VARIANT
typename PathGuide<Float, Spectrum>::Vector3f
PathGuide<Float, Spectrum>::Angles2Euler(const Vector2f &pos) {
    Point2f p(pos.x(), pos.y());
    return warp::square_to_uniform_sphere(p);
}

MI_VARIANT
size_t PathGuide<Float, Spectrum>::Angles2Quadrant(const Vector2f &pos) {
    // takes the 2D location input and returns the corresponding quadrant
    check(dr::any_or<true>(pos.x() >= -dr::Epsilon<Float> &&
                           pos.x() <= 1.0f + dr::Epsilon<Float> &&
                           pos.y() >= -dr::Epsilon<Float> &&
                           pos.y() <= 1.0f + dr::Epsilon<Float>));

    if (dr::any_or<true>(pos.x() < 0.5f && pos.y() < 0.5f)) // top left
                                                            // (quadrant 0)
        return 0;
    else if (dr::any_or<true>(pos.y() < 0.5f)) // must be top right (quadrant 1)
        return 1;
    else if (dr::any_or<true>(pos.x() < 0.5f)) // must be bottom left (quadrant
                                               // 2)
        return 2;
    return 3;
}

MI_VARIANT
typename PathGuide<Float, Spectrum>::Vector2f
PathGuide<Float, Spectrum>::NormalizeForQuad(const Vector2f &pos,
                                             const size_t quad) {
    check(dr::any_or<true>(pos.x() >= -dr::Epsilon<Float> &&
                           pos.x() <= 1.0f + dr::Epsilon<Float> &&
                           pos.y() >= -dr::Epsilon<Float> &&
                           pos.y() <= 1.0f + dr::Epsilon<Float>));
    check(quad <= 3);
    Vector2f ret = pos;
    if (quad == 0) // top left (quadrant 0)
    {              // do nothing! (already within [0,0.5] for both x and y)
        check(dr::any_or<true>(ret.x() >= -dr::Epsilon<Float> &&
                               ret.x() <= 0.5f + dr::Epsilon<Float>));
        check(dr::any_or<true>(ret.y() >= -dr::Epsilon<Float> &&
                               ret.y() <= 0.5f + dr::Epsilon<Float>));
    } else if (quad == 1)             // top right (quadrant 1)
        ret -= Vector2f{ 0.5f, 0.f }; // map (x) [0.5,
                                      // 1] -> [0, 0.5]
    else if (quad == 2)               // bottom left (quadrant 2)
        ret -= Vector2f{ 0.f, 0.5f }; // map (y) [0.5,
                                      // 1] -> [0, 0.5]
    else
        ret -= Vector2f{ 0.5f, 0.5f }; // map (x & y) [0.5,
                                       // 1] -> [0, 0.5]
    check(dr::any_or<true>(ret.x() >= -dr::Epsilon<Float> &&
                           ret.x() <= 0.5f + dr::Epsilon<Float>));
    check(dr::any_or<true>(ret.y() >= -dr::Epsilon<Float> &&
                           ret.y() <= 0.5f + dr::Epsilon<Float>));
    return 2.f * ret; // map [0, 0.5] -> [0, 1]
}

//-------------------DTreeWrapper-------------------//

MI_VARIANT
void PathGuide<Float, Spectrum>::DTreeWrapper::add_sample(const Vector3f &dir,
                                                          const Float lum) {
    auto &tree   = current;
    Vector2f pos = Euler2Angles(dir);
    tree.num_samples++;
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
    current.max_depth   = 0;
    current.num_samples = 0;
    current.sum         = 0.f;
    struct StackItem {
        size_t node_idx;
        size_t other_idx;
        DTreeWrapper::DirTree *tree;
        size_t depth;
    };

    std::stack<StackItem> stack;
    stack.push({ 0, 0, &prev, 1 });

    const size_t max_children = 100000;

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
                const size_t child_idx =
                    current.nodes.size(); // new child's index!
                // create the child!
                current.nodes.emplace_back();
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

                if (current.nodes.size() > max_children) {
                    std::ostringstream oss;
                    oss << "PathGuide::DTreeWrapper reset hit max "
                           "children count!"
                        << std::endl;
                    Log(Error, "%s", oss.str());
                    stack = std::stack<StackItem>(); // reset, break from loop
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
    prev = current; // copy assignment works as intended, current is deepcopied
                    // to prev
}

MI_VARIANT
Float PathGuide<Float, Spectrum>::DTreeWrapper::sample_pdf(
    const Vector3f &dir) const {
    const auto &tree = prev;

    Float pdf = 1.f / (4.f * dr::Pi<Float>); // default naive pdf (unit sphere)
    if (tree.nodes.size() == 0 || tree.num_samples.load() == 0 ||
        dr::any_or<true>(Float(tree.sum) == 0.f))
        return pdf;

    // begin recursing into nodes

    Vector2f pos     = Euler2Angles(dir);
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
    const Vector2f unit_random = sampler->next_2d();
    const auto &tree           = prev;

    // early out to indicate that this tree is invalid
    if (tree.nodes.size() == 0 || tree.num_samples.load() == 0 ||
        dr::any_or<true>(Float(tree.sum) == 0.f))
        return Angles2Euler(unit_random);

    // recurse into the tree
    Vector2f pos{ 0.f, 0.f }; // center of cartesian
                              // plane (no leaning)
    float scale = 1.0f;       // halved on each (non-leaf) iteration

    size_t which_quadrant = 0;
    const auto *node      = &(tree.nodes[0]); // start at root
    while (true) {
        check(node != nullptr);

        if (!node->sample(which_quadrant, sampler)) // invalid!
            return Angles2Euler(unit_random);
        check(which_quadrant <= 3);

        // use a "quadrant origin" to add sample{x,y} to the corresponding
        // quadrant
        const Vector2f quadrant_origin{
            0.5f * (which_quadrant % 2 == 1), // right side of y=0
            0.5f * (which_quadrant >= 2),     // underneath x=0
        };

        if (node->bIsLeaf(which_quadrant)) // hit a leaf
        {
            // add the initial random sample to this quadrant
            pos += scale * (quadrant_origin + 0.5f * unit_random);
            break;
        } else {
            // continue burrowing straight into this quadrant
            pos += scale * quadrant_origin;
            scale /= 2.f;
        }

        size_t child_idx = node->children[which_quadrant];

        // iterate down the tree
        node = &(tree.nodes[child_idx]);
    }

    return Angles2Euler(pos);
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
    const size_t sample_threshold) {
    // traverse dTree via DFS and refine (subdivide) those leaves that qualify
    std::stack<size_t> stack;
    stack.push(0); // root node index
    while (!stack.empty()) {
        size_t idx = stack.top();
        /// NOTE: we use the raw indices of the nodes (in the vector) rather
        /// than
        // storing a single pointer since these elements might get reallocated
        // as the vector resizes!
        stack.pop();

        // currently hit a leaf node, might want to subdivide it (refine)
        if (node(idx).bIsLeaf() &&
            node(idx).dTree.get_num_samples() > sample_threshold) {
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
    /// NOTE: using node(idx) rather than taking a pointer to nodes[idx] because
    // nodes will resize (thus potentially reallocate) which might invalidate
    // any pointers or references!
    const size_t num_samples = node(idx).dTree.get_num_samples();
    nodes.resize(nodes.size() + node(idx).children.size()); // prepare for new
                                                            // children
    for (size_t i = 0; i < node(idx).children.size(); i++) {
        const size_t child_idx = nodes.size() - 2 + i;
        node(idx).children[i]  = child_idx; // assign the child to the parent
        SNode &child           = nodes[child_idx];
        child.dTree            = node(idx).dTree; // copy this node's dirtree
        child.dTree.set_num_samples(num_samples / 2); // approx half the samples
        // "iterate through axes on every pass"
        child.xyz_axis =
            (node(idx).xyz_axis + 1) % 3; // 0 for x, 1 for y, 2 for z
    }
    node(idx).dTree.free_memory(); // reset this dTree to save memory
    check(!node(idx).bIsLeaf());   // definitely has children now
}

MI_VARIANT
const typename PathGuide<Float, Spectrum>::DTreeWrapper &
PathGuide<Float, Spectrum>::SpatialTree::get_direction_tree(
    const Point3f &pos) const {
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
        // go to next child
        idx = node(idx).children[child_idx];
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
void PathGuide<Float, Spectrum>::refine(const size_t thresh) {
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
void PathGuide<Float, Spectrum>::add_radiance(const Point3f &pos,
                                              const Vector3f &dir,
                                              const Color3f &radiance) const {
    Float rad = luminance(radiance);
    if (!dr::any_or<true>(dr::isfinite(rad)))
        return;
    this->add_radiance(pos, dir, rad); // convert to luminance
}

MI_VARIANT
void PathGuide<Float, Spectrum>::add_radiance(const Point3f &pos,
                                              const Vector3f &dir,
                                              const Float luminance) const {
    // if (dr::any_or<true>(luminance <= 0.f)) // 0 luminance won't affect samples
    //     return;
    const DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    const_cast<DTreeWrapper &>(dir_tree).add_sample(dir, luminance);
}

MI_VARIANT
typename PathGuide<Float, Spectrum>::Vector3f
PathGuide<Float, Spectrum>::sample(const Vector3f &pos, Float &pdf,
                                   Sampler<Float, Spectrum> *sampler) const {
    // O(log(n)) search through cartesian space to get the direction dTree at
    // pos
    const DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    // O(log(n)) search through directional coordinates to sample direction and
    // pdf
    const Vector3f ret = dir_tree.sample_dir(sampler);
    pdf                = dir_tree.sample_pdf(ret);
    return ret;
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
