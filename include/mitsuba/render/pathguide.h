#pragma once

#include <mitsuba/core/atomic.h>    // AtomicFloat
#include <mitsuba/core/bbox.h>      // ScalarBoundingBox3f
#include <mitsuba/core/object.h>    // Object
#include <mitsuba/core/progress.h>  // Progress
#include <mitsuba/core/spectrum.h>  // Spectrum
#include <mitsuba/render/sampler.h> // Sampler

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Path Guiding
 *
 * TODO
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB PathGuide : public Object {
public:
    MI_IMPORT_CORE_TYPES() // imports types such as Vector3f, Point3f, Color3f
    PathGuide(const float training_budget_percent)
        : training_budget(training_budget_percent) {
        if (training_budget < 0.f)
            Log(Warn, "Path guide cannot train for negative samples. Disabling "
                      "path guider.");
        if (training_budget >= 1.f) {
            Log(Warn, "Using entirety of sampling budget for training the path "
                      "guider. This means none of the samples will be used for "
                      "(inference) rendering the final image!");
        }
    }

    MI_DECLARE_CLASS()
private: // hyperparameters
    // spatial tree sampling threshold, until a node qualifies for refinement
    const Float spatial_tree_thresh = 4000.f;
    // fraction of energy from the previous tree to use for refiment
    const Float rho = 0.01f;
    // maximum number of children in leaf d-trees
    const size_t max_DTree_depth = 20;
    // percentage of samples in render that are used for training
    // 0.0 => disabled path guider (no training)
    // 0.5 => 50% of the spp for the total render is dedicated for training
    // 0.9 => 90% of the spp for the total render is dedicated for training
    // >= 1.0 causes the final render to have 0 samples (probably not ideal)
    const float training_budget; // set in constructor
    // total number of refinement iterations before training is complete
    size_t num_training_refinements; // set in initialize()

private: // internal use
    // member variables used for internal representation
    size_t refinement_iter = 0; // number of refinements (training passes)
    size_t spp_overflow    = 0; // remaining spp not part of doubling

    // refines the SD-tree, then prepares for next iteration
    void refine(const Float);

    // Variable for tracking intermediate radiance for path guiding
    /// NOTE: this is thread_local so that these accumulations can occur in
    //        parallel along various threads (each thread has its own storage)
    //        and inline static comes from the requirement that the thread_local
    //        members must be static. Overall this allows for T separate storage
    //        instances of the vector to be allocated where T is #threads
    thread_local inline static std::vector<
        std::tuple<Point3f, Vector3f, Spectrum, Spectrum, Spectrum, Float>>
        thru_vars;

    // progress tracking
    ProgressReporter *progress           = nullptr; // train progress reporter
    size_t total_train_spp               = 0;       // total spp for training
    size_t screensize                    = 0;       // sensor resolution
    std::atomic<size_t> atomic_spp_count = 0;       // spp for progress tracking
    void update_progress();                         // after each (atomic) 1 spp

public: // public API
    // begin construction of the SD-tree
    void initialize(const uint32_t scene_spp, const ScalarBoundingBox3f &bbox);

    // query whether or not the path guider should be used at all
    bool enabled() const { return training_budget > 0.f; }

    // query whether or not the path guider is ready for sampling (inference)
    bool ready_for_sampling() const {
        // "We train a sequence L1, L2, ... LM where L1 is estimated with just
        // BSDF sampling and for all k > 1, Lk is esetimated by combining
        // samples of LK-1 and the BSDF via multiple importance sampling."
        return (refinement_iter >= 1);
    }

    // query whether or not the path guider is finished training
    bool done_training() const {
        return (refinement_iter >= num_training_refinements);
    }

    // get number of spp on a particular pass
    uint32_t get_pass_spp(const uint32_t pass_idx) const;

    // return percentage of samples from total scene to be used for training
    float get_training_budget() const { return training_budget; }

    // refine spatial tree from last buffer
    void perform_refinement();

    // call once to allow the path guider to track training progress
    inline void set_train_progress(ProgressReporter *p,
                                   const size_t num_pixels) {
        progress   = p;
        screensize = num_pixels;
    }

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
            Assert(idx < nodes.size());
            return nodes[idx];
        }
        inline SNode &node(const size_t idx) {
            Assert(idx < nodes.size());
            return nodes[idx];
        }

        std::vector<SNode> nodes;
    };

private: // class instances
    class SpatialTree spatial_tree;
};

MI_EXTERN_CLASS(PathGuide)
NAMESPACE_END(mitsuba)
