#pragma once

#include <mitsuba/core/atomic.h>   // AtomicFloat
#include <mitsuba/core/bbox.h>     // ScalarBoundingBox3f
#include <mitsuba/core/fwd.h>      // MI_IMPORT_CORE_TYPES_PREFIX
#include <mitsuba/core/spectrum.h> // Spectrum
#include <mitsuba/core/vector.h>   // Vector

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class MI_EXPORT_LIB PathGuide {
public:
    MI_IMPORT_CORE_TYPES() // imports types such as Vector3f, Point3f, Color3f

private: // hyperparameters
    // these are the primary hyperparameters to tune the pathguiding algorithm
    const size_t spatial_tree_thresh = 12000; // spatial tree sample threshold
    // amount of energy from the previous tree to use for refiment
    const Float rho = 0.01f;
    // maximum number of children in leaf d-trees
    const size_t max_DTree_depth = 20;
    // number of refinements until can sample
    size_t num_refinements_necessary = 6;

    size_t refinement_iter = 0;     // how many refinements have happened
    bool sample_ready      = false; // whether or not we can sample
    void refine(const size_t);      // refines the SD-tree, then prepares
                                    // for next iteration

public: // public API
    PathGuide() = default;
    // begin construction of the SD-tree
    void initialize(const ScalarBoundingBox3f &bbox);

    // return whether the PathGuiding is ready for sampling or needs to be built
    bool ready() const {
        return (refinement_iter >= num_refinements_necessary);
    }

    // return how many refinements are needed to be ready for sampling
    size_t num_refinements_needed() const { return num_refinements_necessary; }

    // refine spatial tree from last buffer
    void refine();

    // to keep track of radiance in the lightfield
    void add_radiance(const Point3f &pos, const Vector3f &dir,
                      const Color3f &radiance) const;
    void add_radiance(const Point3f &pos, const Vector3f &dir,
                      const Float luminance) const;

    // to (importance) sample a direction and its corresponding pdf
    Vector3f sample(const Vector3f &pos, Float &pdf) const;

public:
    // utility methods
    static Vector2f Euler2Angles(const Vector3f &dir);
    static Vector3f Angles2Euler(const Vector2f &pos);
    static size_t Angles2Quadrant(const Vector2f &pos);
    static Vector2f NormalizeForQuad(const Vector2f &pos, const size_t quad);

private: // DirectionTree (and friends) declaration
    class DTreeWrapper {
    public:
        void reset(size_t max_depth, Float rho);
        void build();

        size_t get_num_samples() const { return current.num_samples.load(); }

        void set_num_samples(const size_t num_samples) {
            current.num_samples.store(num_samples);
        }

        Float sample_pdf(const Vector3f &dir) const;
        Vector3f sample_dir() const;

        void add_sample(const Vector3f &dir, const Float lum);

        // destroy all memory used by this class (danger!)
        // (probably only want this if you KNOW you aren't going to use this
        // tree's nodes anymore)
        void free_memory();

    private:
        struct DirTree {
            DirTree() {
                nodes.resize(1);
                num_samples = 0;
                sum         = 0;
            }

            void free() {
                std::vector<DirNode>().swap(nodes); // de-allocate all memory
            }

            struct DirNode {
                DirNode() = default;
                std::array<AtomicFloat<Float>, 4> data;
                std::array<size_t, 4> children{};
                bool sample(size_t &quadrant) const;
                bool bIsLeaf(size_t idx) const { return children[idx] == 0; }
                Float sum() const {
                    Float total = 0.f;
                    for (const auto &x : data)
                        total += Float(x); // x.load(std::memory_order_relaxed)
                    return total;
                }
                void data_fill(const Float new_data) {
                    for (auto &x : data) {
                        x = new_data; // atomic store
                    }
                }
                DirNode &operator=(const DirNode &other) {
                    children = other.children;
                    for (size_t i = 0; i < data.size(); i++) {
                        data[i] = Float(other.data[i]);
                    }
                    return *this;
                }
                DirNode(const DirNode &other) : children(other.children) {
                    for (size_t i = 0; i < data.size(); i++) {
                        data[i] = Float(other.data[i]);
                    }
                }
            };

            // assignment operator
            DirTree &operator=(const DirTree &other) {
                max_depth = other.max_depth;
                nodes     = other.nodes;
                // assign atomics here
                sum = Float(other.sum); // load(std::memory_order_relaxed)
                num_samples = other.num_samples.load();
                return *this;
            }

            // copy constructor
            DirTree(const DirTree &other)
                : max_depth(other.max_depth), nodes(other.nodes) {
                // assign atomics here
                num_samples = other.num_samples.load();
                sum = Float(other.sum); // load(std::memory_order_relaxed)
            }

            std::atomic<size_t> num_samples;
            AtomicFloat<Float> sum;
            size_t max_depth = 0;
            std::vector<DirNode> nodes;
        };

        // keep track of current and previous direction trees in the same
        // overarching "DTreeWrapper" wrapper to only need one binary search
        // through the spatial tree to reach these leaves. The underlying
        // spatial structure for both trees is the same regardless.

        /*
        Section 5.2: Memory Usage:

        "because the spatial binary tree of L^k is merely a more refined version
        of the spatial tree of L^{k−1}, it is straightforward to use the same
        spatial tree for both distributions, where each leaf contains two
        directional quadtrees; one for L^{k−1} and one for L^k
        */
        DirTree current, prev;
    };

private: // SpatialTree (whose leaves are DirectionTrees) declaration
    class SpatialTree {
    public:
        SpatialTree() { nodes.resize(1); }
        ScalarBoundingBox3f bounds;
        void begin_next_tree_iteration();
        void refine(const size_t sample_threshold);
        void reset_leaves(const size_t max_depth, const Float rho);

        DTreeWrapper &get_direction_tree(const Point3f &pos) {
            return const_cast<DTreeWrapper &>(
                const_cast<const SpatialTree *>(this)->get_direction_tree(pos));
        }
        const DTreeWrapper &get_direction_tree(const Point3f &pos) const;
        // {
        //     return const_cast<SpatialTree *>(this)->get_direction_tree(pos);
        // }

    private:
        struct SNode // spatial-tree-node
        {
            DTreeWrapper dTree;
            std::array<size_t, 2> children{};
            uint8_t xyz_axis{}; // (0:x, 1:y, 2:z) which axis to split on
                                // (cycles through children)
            inline bool bIsLeaf() const {
                return children[0] ==
                       children[1]; // equal children => no children
            }
        };

        void subdivide(const size_t parent_idx);
        const SNode &node(const size_t idx) const {
            // check(idx < nodes.size());
            return nodes[idx];
        }
        SNode &node(const size_t idx) {
            // check(idx < nodes.size());
            return nodes[idx];
        }

        std::vector<SNode> nodes;
    };

private: // class instances
    class SpatialTree spatial_tree;
};

NAMESPACE_END(mitsuba)
