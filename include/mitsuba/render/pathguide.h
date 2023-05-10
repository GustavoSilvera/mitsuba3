#pragma once

#include <mitsuba/core/atomic.h>    // AtomicFloat
#include <mitsuba/core/bbox.h>      // ScalarBoundingBox3f
#include <mitsuba/core/spectrum.h>  // Spectrum
#include <mitsuba/render/sampler.h> // Sampler

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class PathGuide {
public:
    MI_IMPORT_CORE_TYPES() // imports types such as Vector3f, Point3f, Color3f

private: // hyperparameters
    // these are the primary hyperparameters to tune the pathguiding algorithm
    const Float spatial_tree_thresh = 1000.f; // spatial tree sample threshold
    // amount of energy from the previous tree to use for refiment
    const Float rho = 0.01f;
    // maximum number of children in leaf d-trees
    const size_t max_DTree_depth = 20;
    // number of refinements until can sample
    size_t num_refinements_necessary = 10;

    const bool bIsEnabled = true;

    size_t refinement_iter = 0;     // how many refinements have happened
    bool sample_ready      = false; // whether or not we can sample
    void refine(const Float);       // refines the SD-tree, then prepares
                                    // for next iteration

    // Variable for tracking intermediate radiance for path guiding
    /// NOTE: this is thread_local so that these accumulations can occur in
    //        parallel along various threads (each thread has its own storage)
    //        and inline static comes from the requirement that the thread_local
    //        members must be static. Overall this allows for T separate storage
    //        instances of the vector to be allocated where T is #threads
    thread_local inline static std::vector<
        std::tuple<Point3f, Vector3f, Spectrum, Spectrum, Spectrum, Float>>
        thru_vars;

public: // public API
    PathGuide() = default;
    // begin construction of the SD-tree
    void initialize(const ScalarBoundingBox3f &bbox);

    bool enabled() const { return bIsEnabled; }

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
                      const Float luminance,
                      Sampler<Float, Spectrum> *sampler = nullptr);

    // keep track of throughput to calculate incident radiance at every boucne
    void add_throughput(const Point3f &pos, const Vector3f &dir,
                        const Spectrum &result, const Spectrum &throughput,
                        const Float woPdf);

    // when the radiance is not computed recursively, it is nontrivial to get
    // the incident radiance at every bounce. So this provides the means to
    // store intermediate variables and recompute these quantities for training

    void calc_radiance_from_thru(Sampler<Float, Spectrum> *sampler);

    // to (importance) sample a direction and its corresponding pdf
    std::pair<Vector3f, Float> sample(const Vector3f &pos,
                                      Sampler<Float, Spectrum> *sampler) const;
    Float sample_pdf(const Point3f &pos, const Vector3f &dir) const;

public:
    // utility methods
    static size_t Angles2Quadrant(const Point2f &pos);
    static Point2f NormalizeForQuad(const Point2f &pos, const size_t quad);

private: // DirectionTree (and friends) declaration
    class DTreeWrapper {
    public:
        void reset(size_t max_depth, Float rho);
        void build();

        Float get_weight() const {
            return Float(current.weight); // atomic load
        }

        void set_weight(const Float weight) {
            current.weight = weight; // atomic store
        }

        Float sample_pdf(const Vector3f &dir) const;
        Vector3f sample_dir(Sampler<Float, Spectrum> *sampler) const;

        void add_sample(const Vector3f &dir, const Float lum,
                        const Float weight);

        // destroy all memory used by this class (danger!)
        // (probably only want this if you KNOW you aren't going to use this
        // tree's nodes anymore)
        void free_memory();

    private:
        struct DirTree {
            DirTree() {
                nodes.resize(1);
                weight = 0.f;
                sum    = 0.f;
            }

            void free() {
                std::vector<DirNode>().swap(nodes); // de-allocate all memory
            }

            struct DirNode {
                DirNode() = default;
                std::array<AtomicFloat<Float>, 4> data;
                std::array<size_t, 4> children{};
                bool sample(size_t &quadrant,
                            Sampler<Float, Spectrum> *sampler) const;
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
                sum    = Float(other.sum); // load(std::memory_order_relaxed)
                weight = Float(other.weight);
                return *this;
            }

            // copy constructor
            DirTree(const DirTree &other)
                : max_depth(other.max_depth), nodes(other.nodes) {
                // assign atomics here
                weight = Float(other.weight);
                sum    = Float(other.sum); // load(std::memory_order_relaxed)
            }

            AtomicFloat<Float> weight;
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
        void refine(const Float sample_threshold);
        void reset_leaves(const size_t max_depth, const Float rho);

        DTreeWrapper &get_direction_tree(const Point3f &pos,
                                         Vector3f *size = nullptr) {
            return const_cast<DTreeWrapper &>(
                const_cast<const SpatialTree *>(this)->get_direction_tree(
                    pos, size));
        }
        const DTreeWrapper &get_direction_tree(const Point3f &pos,
                                               Vector3f *size = nullptr) const;

    private:
        struct SNode // spatial-tree-node
        {
            DTreeWrapper dTree;
            std::array<size_t, 2> children{};
            uint8_t xyz_axis{}; // (0:x, 1:y, 2:z) which axis to split on
                                // (cycles through children)
            inline bool bIsLeaf() const {
                // equal children => no children
                return children[0] == children[1];
            }
        };

        void subdivide(const size_t parent_idx);
        inline const SNode &node(const size_t idx) const {
            // check(idx < nodes.size());
            return nodes[idx];
        }
        inline SNode &node(const size_t idx) {
            // check(idx < nodes.size());
            return nodes[idx];
        }

        std::vector<SNode> nodes;
    };

private: // class instances
    class SpatialTree spatial_tree;
};

NAMESPACE_END(mitsuba)
