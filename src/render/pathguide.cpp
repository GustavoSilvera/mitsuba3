#include <dirt/pathguide.h>
#include <stack> // std::stack

#define check(x)                                                               \
    if (!(x))                                                                  \
        throw std::runtime_error("Assertion failed on line " +                 \
                                 std::to_string(__LINE__));

// utility method to go from Vec3f (roll, pitch, yaw) -> Vec2f (theta, phi)
static Vec2f Euler2Angles(const Vec3f &dir) {
    const float cosTheta = std::min(std::max(dir.z, -1.0f), 1.0f);
    float phi            = std::atan2(dir.y, dir.x);
    while (phi < 0.f) {
        phi += 2.f * M_PI;
    }

    return Vec2f((cosTheta + 1.f) * 0.5f, phi * 1.f / (2.f * M_PI));
}

// utility method to go from Vec2f (theta, phi) -> Vec3f (roll, pitch, yaw)
static Vec3f Angles2Euler(const Vec2f &pos) {
    const float cosTheta = 2.f * pos.x - 1.f;
    const float phi      = 2.f * M_PI * pos.y;
    const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);
    return Vec3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

static size_t Angles2Quadrant(const Vec2f &pos) {
    // takes the 2D location input and returns the corresponding quadrant
    check(pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f);

    if (pos.x < 0.5f && pos.y < 0.5f) // top left (quadrant 0)
        return 0;
    else if (pos.y < 0.5f) // must be top right (quadrant 1)
        return 1;
    else if (pos.x < 0.5f) // must be bottom left (quadrant 2)
        return 2;
    return 3;
}

static Vec2f NormalizeForQuad(const Vec2f &pos, const size_t quad) {
    check(pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f);
    check(quad <= 3);
    Vec2f ret = pos;
    if (quad == 0) // top left (quadrant 0)
    {              // do nothing! (already within [0,0.5] for both x and y)
        check(ret.x >= 0.f && ret.x <= 0.5f);
        check(ret.y >= 0.f && ret.y <= 0.5f);
    } else if (quad == 1)          // top right (quadrant 1)
        ret -= Vec2f{ 0.5f, 0.f }; // map (x) [0.5, 1] -> [0, 0.5]
    else if (quad == 2)            // bottom left (quadrant 2)
        ret -= Vec2f{ 0.f, 0.5f }; // map (y) [0.5, 1] -> [0, 0.5]
    else
        ret -= Vec2f{ 0.5f, 0.5f }; // map (x & y) [0.5, 1] -> [0, 0.5]
    check(ret.x >= 0.f && ret.x <= 0.5f);
    check(ret.y >= 0.f && ret.y <= 0.5f);
    return 2.f * ret; // map [0, 0.5] -> [0, 1]
}

//-------------------DTreeWrapper-------------------//

void PathGuide::DTreeWrapper::add_sample(const Vec3f &dir, const float lum) {
    auto &tree = current;
    Vec2f pos  = Euler2Angles(dir);
    tree.num_samples++;
    tree.sum = tree.sum.load() + lum;

    if (tree.nodes.size() == 0)
        tree.nodes.resize(1); // ensure always have a root node!
    check(tree.nodes.size() >= 1);

    // update internal nodes
    DirTree::DirNode *node = &(tree.nodes[0]); // root
    while (true) {
        check(node != nullptr);
        const size_t quad_idx = Angles2Quadrant(pos);
        pos                   = NormalizeForQuad(pos, quad_idx);
        node->data[quad_idx]  = node->data[quad_idx].load() + lum;
        if (node->bIsLeaf(quad_idx))
            break;
        size_t child_idx = node->children[quad_idx];
        node             = &(tree.nodes[child_idx]);
    }
}

void PathGuide::DTreeWrapper::reset(const size_t max_depth, const float rho) {
    // clear and re-initialize the nodes
    current.nodes.clear();
    current.nodes.resize(1);
    current.max_depth   = 0;
    current.num_samples = 0;
    current.sum         = 0.f;
    struct StackItem {
        size_t node_idx;
        size_t other_idx;
        const PathGuide::DTreeWrapper::DirTree *tree;
        size_t depth;
    };

    std::stack<StackItem> stack;
    stack.push({ 0, 0, &prev, 1 });

    const size_t max_children = 100000;

    const float prev_sum = prev.sum.load();
    while (!stack.empty()) {
        const StackItem s = stack.top();
        stack.pop();

        current.max_depth = std::max(current.max_depth, s.depth);
        check(s.tree != nullptr);
        check(s.other_idx < s.tree->nodes.size());
        const DirTree::DirNode &other_node = s.tree->nodes[s.other_idx];
        for (size_t quad = 0; quad < other_node.data.size(); quad++) {
            const float quad_sum = other_node.data[quad].load();
            if (s.depth < max_depth && quad_sum > prev_sum * rho) {
                // add child and check if parent
                const size_t child_idx =
                    current.nodes.size(); // new child's index!
                // create the child!
                current.nodes.emplace_back();
                DirTree::DirNode &new_node = current.nodes.back();

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
                    std::cout << "PathGuide::DTreeWrapper reset hit max "
                                 "children count!"
                              << std::endl;
                    stack = std::stack<StackItem>(); // reset, break from loop
                    break;
                }
            }
        }
    }

    // now set all the new energy to 0
    for (DirTree::DirNode &node : current.nodes) {
        node.data_fill(0.f);
    }
}

void PathGuide::DTreeWrapper::build() {
    // must always have a root node!
    check(current.nodes.size() > 0);
    check(prev.nodes.size() > 0);
    // keep track of this tree as the last iteration's
    prev = current; // copy assignment works as intended, current is deepcopied
                    // to prev
}

float PathGuide::DTreeWrapper::sample_pdf(const Vec3f &dir) const {
    const auto &tree = prev;

    float pdf = 1.f / (4.f * M_PI); // default naive pdf (unit sphere)
    if (tree.nodes.size() == 0 || tree.num_samples == 0 || tree.sum == 0.f)
        return pdf;

    // begin recursing into nodes

    Vec2f pos                    = Euler2Angles(dir);
    const DirTree::DirNode *node = &(tree.nodes[0]); // start at root
    while (true) {
        check(node != nullptr);

        const size_t quad_idx = Angles2Quadrant(pos);
        pos                   = NormalizeForQuad(pos, quad_idx);

        const float quad_samples = node->data[quad_idx].load();
        if (quad_samples <= 0.f)
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

bool PathGuide::DTreeWrapper::DirTree::DirNode::sample(size_t &quadrant) const {
    const float top_left  = data[0].load();
    const float top_right = data[1].load();
    const float bot_left  = data[2].load();
    const float bot_right = data[3].load();
    const float total     = top_left + top_right + bot_left + bot_right;

    // just use unit random
    if (total == 0.f)
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

    const float sample = randf();
    if (sample < top_left / total) // dice rolls top left
    {
        quadrant = 0;
    } else if (sample < (top_left + top_right) / total) // dice rolls top right
    {
        quadrant = 1;
    } else if (sample < (top_left + top_right + bot_left) /
                            total) // dice rolls bottom left
    {
        quadrant = 2;
    } else // dice rolls bottom right
    {
        check(sample <= 1.f);
        quadrant = 3;
    }
    check(quadrant <= 3); // 0, 1, 2, or 3
    return true;
}

void PathGuide::DTreeWrapper::free_memory() {
    // free both of the quad trees
    current.free();
    prev.free();
}

Vec3f PathGuide::DTreeWrapper::sample_dir() const {
    const Vec2f unit_random{ randf(), randf() };
    const auto &tree = prev;

    // early out to indicate that this tree is invalid
    if (tree.nodes.size() == 0 || tree.num_samples == 0 || tree.sum == 0.f)
        return Angles2Euler(unit_random);

    // recurse into the tree
    Vec2f pos{ 0.f, 0.f }; // center of cartesian plane (no leaning)
    float scale = 1.0f;    // halved on each (non-leaf) iteration

    size_t which_quadrant        = 0;
    const DirTree::DirNode *node = &(tree.nodes[0]); // start at root
    while (true) {
        check(node != nullptr);

        if (!node->sample(which_quadrant)) // invalid!
            return Angles2Euler(unit_random);
        check(which_quadrant <= 3);

        // use a "quadrant origin" to add sample{x,y} to the corresponding
        // quadrant
        const Vec2f quadrant_origin{
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
PathGuide::SpatialTree::SpatialTree() {
    nodes.resize(1); // allocate root node
}

void PathGuide::SpatialTree::set_bounds(const Box3f &b) { bounds = b; }

void PathGuide::SpatialTree::begin_next_tree_iteration() {
    for (auto &node : nodes)
        if (node.bIsLeaf())
            node.dTree.build();
}

void PathGuide::SpatialTree::reset_leaves(size_t max_depth, float rho) {
    for (auto &node : nodes)
        if (node.bIsLeaf())
            node.dTree.reset(max_depth, rho);
}

void PathGuide::SpatialTree::refine(const size_t sample_threshold) {
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

void PathGuide::SpatialTree::subdivide(const size_t idx) {
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

const PathGuide::DTreeWrapper &
PathGuide::SpatialTree::get_direction_tree(const Vec3f &pos) const {
    // find the leaf node that contains this position

    // use a position normalized [0 -> 1]^3 within this dTree's bbox
    const Vec3f bbox_size = bounds.pMax - bounds.pMin;
    Vec3f x               = (pos - bounds.pMin) / bbox_size;

    check(nodes.size() > 0); // need at least a root node!

    const float split = 0.5f; // decision boundary between left and right child
    size_t idx        = 0;    // start at root node, descent down tree
    while (!node(idx).bIsLeaf()) {
        const auto ax = node(idx).xyz_axis;
        check(ax <= 2); // x, y, z

        size_t child_idx = 0; // assume going to child 0
        if (x[ax] > split)    // actually going to child 1
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

void PathGuide::initialize(const Box3f &bbox, const size_t num_iters) {
    std::cout << "initialized path guide with bounds: " << bbox.pMin << " -> "
              << bbox.pMax << std::endl;
    num_refinements_necessary = num_iters;
    spatial_tree.set_bounds(bbox);
    refine_and_reset(spatial_tree_thresh); // initial refinement/reset
}

void PathGuide::refine_and_reset(const size_t thresh) {
    spatial_tree.refine(thresh);
    spatial_tree.reset_leaves(max_DTree_depth, rho);
}

void PathGuide::refine_and_reset() {
    spatial_tree.begin_next_tree_iteration(); // keep track of last trees
    num_reset_iters++;
    // next iter should have sqrt(2^n) times the threshold
    size_t thresh =
        std::pow(2.f, (num_reset_iters + 2) * 0.5f) * spatial_tree_thresh;
    this->refine_and_reset(thresh);
    // ready for sampling if enough iterations have been met
    sample_ready = (num_reset_iters >= num_refinements_necessary);
}

void PathGuide::add_radiance(const Vec3f &pos, const Vec3f &dir,
                             const Color3f &radiance) {
    this->add_radiance(pos, dir, luminance(radiance)); // convert to luminance
}

void PathGuide::add_radiance(const Vec3f &pos, const Vec3f &dir,
                             const float luminance) {
    DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    dir_tree.add_sample(dir, luminance);
}

Vec3f PathGuide::sample_dir(const Vec3f &pos, float &pdf) const {
    // O(log(n)) search through cartesian space to get the direction dTree at
    // pos
    const DTreeWrapper &dir_tree = spatial_tree.get_direction_tree(pos);
    // O(log(n)) search through directional coordinates to sample direction and
    // pdf
    const Vec3f ret = dir_tree.sample_dir();
    pdf             = dir_tree.sample_pdf(ret);
    return ret;
}