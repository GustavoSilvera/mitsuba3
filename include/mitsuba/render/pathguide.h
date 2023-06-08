#pragma once

#include <mitsuba/core/bbox.h>      // ScalarBoundingBox3f
#include <mitsuba/core/object.h>    // Object
#include <mitsuba/core/progress.h>  // Progress
#include <mitsuba/core/spectrum.h>  // Spectrum
#include <mitsuba/render/sampler.h> // Sampler

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Path Guiding
 *
 * A technique enabling importance-sampling of the scattering integral
 * by learning an approximation of the incident radiance and sampling
 * it accordingly. This approach is commonly combined with BSDF sampling via
 * multiple importance sampling to attain low variance [1]
 *
 * Implementation of "Practical Path Guiding" from primarily
 * [1] "Practical Path Guiding for Efficient Light-Transport Simulation"
 *       Thomas Muller, Markus Gross, Jan Novak
 *       Proceedings of EGSR 2017, vol. 36, no.4
 *     (https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf)
 *
 * with some additional improvements from the follow-up discussion in
 * [2] "Practical Path Guiding In Production" by Thomas Muller.
 *     (https://tom94.net/data/courses/vorba19guiding/vorba19guiding-chapter10.pdf)
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB PathGuide : public Object {
public:
    MI_IMPORT_CORE_TYPES() // imports types such as Vector3f, Point3f, Color3f

    PathGuide(const float training_budget, const float p_jitter);

    MI_DECLARE_CLASS()

public: /* public API */
    /**
     * \brief Provide run-time arguments for path guider initialization
     */
    void initialize(const uint32_t scene_spp, const ScalarBoundingBox3f &bbox);

    /**
     * \brief Query whether or not the path guider should be used at all
     *
     * If the training budget <= 0% then we can safely assume the user does not
     * want to use path guiding at all. So this is our disabled criteria.
     */
    bool enabled() const { return m_training_budget > 0.f; }

    /**
     * \brief Query if the path guider is ready for sampling (inference)
     *
     * The path guider should only be ready for sampling past a certain number
     * of training iterations. This could be a user-defined parameter.
     */
    bool ready_for_sampling() const {
        /* "We train a sequence L1, L2, ... LM where L1 is estimated with just
         * BSDF sampling and for all k > 1, Lk is esetimated by combining
         * samples of LK-1 and the BSDF via multiple importance sampling"[1]
         */
        return (refinement_iter >= 1);
    }

    /**
     * \brief Query if the path guider is finished training.
     *
     * The path guider "training" consists of several iterative rendering passes
     * that each refine the SD-tree used to approximate the incident radiance of
     * the scene. With more refinements the approximation gets better but the
     * spp ~doubles on each pass
     */
    bool done_training() const {
        return (refinement_iter >= num_training_refinements);
    }

    /**
     * \brief Returns the number of spp for training on a particular pass
     *
     * Typically, this follows the geometric-series 2^i as described in [1] but
     * in case the desired training budget does not fall evenly in a
     * power-of-two, there will be some overflow that can be either added to the
     * final pass or constitute its own extra final pass.
     */
    uint32_t get_pass_spp(const uint32_t pass_idx) const;

    /**
     * \brief Return percentage of samples from scene to be used for training
     *
     * A number from (0, 1) used to separate the total sampling budget into (1)
     * training the path guider and (2) querying the path guider for the render.
     */
    float get_training_budget() const { return m_training_budget; }

    /**
     * \brief Refine the SD-tree to conclude one training pass
     *
     * This method begins the next training iteration and refines the SD-tree by
     * subdividing the leaves of the spatial-tree if they are large enough
     */
    void perform_refinement();

    /**
     * \brief Call once to allow the path guider to track training progress
     *
     * Gives the path guider a progress reporter to update and the total sensor
     * size (in pixels) for normalizing the progress
     */
    inline void set_train_progress(ProgressReporter *p,
                                   const uint32_t num_pixels) {
        progress   = p;
        screensize = num_pixels;
    }

    /**
     * \brief Tracking a radiance sample in the 5D lightfield, used for training
     *
     * Adding radiance from the 5D (3D xyz + 2D dir) lightfield first traverses
     * down the spatial tree to find the leaf containing a directional tree to
     * add the sample. Optionally, the sample can be jittered to act as a
     * stochastic filter and reduce artifacts [2]
     */
    void add_radiance(const Point3f &pos, const Vector3f &dir,
                      const Float luminance,
                      Sampler<Float, Spectrum> *sampler = nullptr);

    /**
     * \brief Track variables in throughput accumulation for incident radiance
     *
     * See path.cpp::sample() to see how the add_throughput method is used
     */
    void add_throughput(const Point3f &pos, const Vector3f &dir,
                        const Spectrum &result, const Spectrum &throughput,
                        const Float woPdf);

    /**
     * \brief Calculate radiance from intermediate throughput variables
     *
     * When the radiance is not computed recursively, it is nontrivial to get
     * the incident radiance at every bounce. So we need to store some
     * intermediate variables and calculate the incident radiance from
     * throughput for each bounce along the path.
     */
    void calc_radiance_from_thru(Sampler<Float, Spectrum> *sampler);

    /**
     * \brief To (importance) sample a direction according to the learned 5D
     * lightfield.
     *
     * Returns both the sampled direction (Vec3) and the pdf of this sample
     * (Float) in a tuple.
     */
    std::pair<Vector3f, Float> sample(const Vector3f &pos,
                                      Point2f sample) const;

    /**
     * \brief Returns the pdf of sampling the direction (dir) from the position
     * (pos) with the current path guiding system
     */
    Float sample_pdf(const Point3f &pos, const Vector3f &dir) const;

private: /* Utility Methods */
    /**
     * \brief Converting a 2D direction to a quadrant (0, 1, 2, 3)
     *
     * Quadrants are indexed like this
     *  0.......1
     *  ---------  0
     *  | 0 | 1 |  .
     *  ---------  .
     *  | 2 | 3 |  .
     *  ---------  1
     */
    static uint32_t Angles2Quadrant(const Point2f &pos);

    /**
     * \brief Re-normalize a point within a quad back to (0,1)^2
     *
     * Takes a 2D point that lies within a quad (separated between {0, 0.5, 1}
     * for x and y) and renormalizes the point to be within (0, 1) for x and y
     * so this point can be used further down the tree.
     */
    static Point2f NormalizeForQuad(const Point2f &pos, const uint32_t quad);

private: /* constructor parameters */
    /**
     * \brief percentage of samples in render that are used for training
     *
     * 0.0 => disabled path guider (no training)
     * 0.5 => 50% of the spp for the total render is dedicated for training
     * 0.9 => 90% of the spp for the total render is dedicated for training
     * >= 1.0 causes the final render to have 0 samples (probably not ideal)
     */
    const float m_training_budget;

    /**
     * \brief Probability of jittering a sample within its spatial neighbourhood
     *
     * This acts as a filter over the spatial domain to reduce artifacts caused
     * by the boundary conditions of the spatial subdivisions. See "Stochastic
     * Spatial Filter" in "Practical Path Guiding” in Production [2]
     */
    const Float m_jitter_prob;

private: /* hyperparameters (from the paper recommendations)*/
    /**
     * \brief Spatial tree threshold, until a node qualifies for refinement.
     *
     * In the original (2017) paper [1] this parameter was recommended to be
     * 12000, but in the follow-up (2019) [2] the improvements with the spatial
     * and directional filtering allowed for a better learned approximation
     * while avoiding artifacts (hence a smaller threshold of 4000)
     */
    const Float spatial_tree_thresh = 4000.f;

    /**
     * \brief Maximum number of children in leaf d-trees.
     *
     * "limiting the maximum depth of the quadtree to 20, which is sufficient
     * for guiding towards extremely narrow radiance sources without precision
     * issues" [1]
     */
    const uint32_t max_DTree_depth = 20;

    /**
     * \brief Fraction of energy from the previous tree to use for refiment
     *
     * "We found rho=0.01 to work well, which in practice results in an average
     * of roughly 300 nodes per quadtree ...[and typically far below] the
     * theoretical maximum of 4 * 20 / rho = 8000"[1]
     */
    const Float rho = 0.01f;

private: /* Internal implementation */
    /** \brief member variables used for internal representation */
    uint32_t refinement_iter = 0; // number of refinements (training passes)
    uint32_t num_training_refinements; // number of refinements for training
    uint32_t spp_overflow = 0; // any remaining spp from geometric series

    /**
     * \brief One step to refine the entire SD-tree
     *
     * Traverses through the tree and subdivides the leaves that surpass the
     * weight threshold (passed in as a vairable)
     */
    void refine(const Float);

    /**
     * \brief Variable for tracking intermediate radiance for path guiding.
     *
     * In cases where radiance is computed by accumulating throughput from the
     * eye to the light source (rather than from the light source to the eye),
     * some intermediate variables need to be tracked in order to calculate the
     * radiance from any point along the path to the eventual light source.
     *
     * This is thread_local so that these accumulations can occur in parallel
     * along various threads (each thread has its own storage) and inline static
     * comes from the requirement that the thread_local members must be static.
     *
     * This could have instead been an automatic variable within each
     * integrator's sample() method, but this approach is less intrusive albeit
     * less intuitive upon first glance.
     */
    thread_local inline static std::vector<
        std::tuple<Point3f, Vector3f, Spectrum, Spectrum, Spectrum, Float>>
        thru_vars;

    /**
     * \brief Progress tracking of training process
     *
     * If we only updated the progress on every training pass, the progress
     * meter would be updated in doubling-increments since the spp doubles on
     * each pass. This is not very pleasant to look at since the first few
     * finish nearly instantly, then the progress meter is stuck at 50% for
     * ~half the time. By adding these internal variables we can track how many
     * samples have gone through the training process compared to the total
     * amount and update the progress tracker uniformally during training.
     */
    ProgressReporter *progress             = nullptr; // train progress reporter
    uint32_t total_train_spp               = 0;       // total spp for training
    uint32_t screensize                    = 0;       // sensor resolution
    std::atomic<uint32_t> atomic_spp_count = 0; // spp for progress tracking
    void update_progress();                     // after each (atomic) 1 spp

private: /* Quantized Atomic Float Accumulator */
    /**
     * \brief Quantized float data structure for atomic positive accumulation
     *
     * Like AtomicFloat (see mitsuba/core/atomic.h) but quantizing the floating
     * point value into a fixed-precision integer accumulator to preserve
     * addition associativity. Otherwise there may be indeterminism when
     * atomically adding floats just by the order in which they were
     * (atomically) added.
     *
     * Note this class is only designed for summing positive floating point
     * values such as luminance/radiance. Some assumptions are therefore used
     * for representing the quantized (positive) float and detecting overflow.
     */
    struct QuantizedAtomicFloatAccumulator {
        QuantizedAtomicFloatAccumulator() = default;
        QuantizedAtomicFloatAccumulator(
            const QuantizedAtomicFloatAccumulator &other) {
            (*this) = other;
        }
        QuantizedAtomicFloatAccumulator &
        operator=(const QuantizedAtomicFloatAccumulator &other) {
            data.store(other.data.load());
            return *this;
        }

        /**
         * \brief Convert between continuous floating point and fixed-point
         *
         * Convert a Float to a quantized uint64_t by including the last N
         * decimal places in the integral component, then truncating to a
         * uint64_t. Note this may fail with overflow if the floating point
         * number times the k_scale (decimal precision) exceeds 2^64-1
         */
        static inline uint64_t quantize(const Float num) {
            const float to_quantize = to_float(num) * k_scale;
            if (to_quantize > std::numeric_limits<uint64_t>::max()) {
                Log(Warn, "Quantizing %f results in an overflow", num);
            }
            return static_cast<uint64_t>(to_quantize);
        }

        /** \brief "de-quantize" back to Float with N decimal places */
        operator Float() const {
            return Float(data.load(std::memory_order_relaxed) / k_scale);
        }

        /** \brief helper method converting Float (template) to normal float */
        static inline float to_float(const Float in) {
            float out = 0.f;
            if constexpr (std::is_same_v<Float, float>) {
                out = in;
            } else { // assume vector of size 1
                out = static_cast<float>(in.entry(0));
            }
            return out;
        }

        /** \brief positive atomic saturating accumulation */
        void operator+=(const Float other) {
            if (dr::all(other < 0))
                Log(Warn, "Quantized atomic float accumulator is only designed "
                          "to add positive floating values.");
            if (dr::all(other == 0.f))
                return; // no-op on adding 0
            const uint64_t summand = quantize(other);
            uint64_t prev          = data.load();
            // the following do-while loop performs a saturating-atomic-add
            // where we ensure (through the compare_exchange) that the value of
            // data has not been changed by another thread when doing our
            // saturation checks.
            do {
                uint64_t new_sum = prev + summand;
                // saturating add (handles overflow)
                if (prev > new_sum) { // overflow detected
                    data.store(std::numeric_limits<uint64_t>::max());
                    Log(Warn, "Quantized atomic float hit saturation!");
                    return;
                }
                // ensures the value of data hasn't changed from prev since we
                // last loaded it, or we run through the saturation check again
            } while (!data.compare_exchange_weak(prev, prev + summand));
        }

        /** \brief Float assignment operator */
        void operator=(const Float in) { data.store(quantize(in)); }

        // increasing the number of digits (k_scale) may cause
        // addition-saturation sooner because fewer bits are reserved for the
        // integer quantity (more for the decimal).
        constexpr static float k_scale = 10000.f; // 4 decimal digits
        std::atomic<uint64_t> data     = 0;       // underlying integral atomic
    };

private: /* DirectionTree (and friends) declaration */
    class DTreeWrapper {
    public:
        void reset(uint32_t max_depth, Float rho);
        void build();

        Float get_weight() const {
            return Float(current.weight); // atomic load
        }

        void set_weight(const Float weight) {
            current.weight = weight; // atomic store
        }

        Float sample_pdf(const Vector3f &dir) const;
        Vector3f sample_dir(Point2f &sample) const;

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
                std::array<QuantizedAtomicFloatAccumulator, 4> data;
                std::array<uint32_t, 4> children{};
                bool sample(uint32_t &quadrant, Float &r1) const;
                bool bIsLeaf(uint32_t idx) const { return children[idx] == 0; }
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
                    for (uint32_t i = 0; i < data.size(); i++) {
                        data[i] = Float(other.data[i]);
                    }
                    return *this;
                }
                DirNode(const DirNode &other) : children(other.children) {
                    for (uint32_t i = 0; i < data.size(); i++) {
                        data[i] = Float(other.data[i]);
                    }
                }
            };

            // assignment operator (deepcopy)
            DirTree &operator=(const DirTree &other) {
                max_depth = other.max_depth;
                nodes     = other.nodes;
                sum       = Float(other.sum);
                weight    = Float(other.weight);
                return *this;
            }

            // copy constructor
            DirTree(const DirTree &other)
                : max_depth(other.max_depth), nodes(other.nodes) {
                weight = Float(other.weight);
                sum    = Float(other.sum);
            }

            QuantizedAtomicFloatAccumulator weight, sum;
            uint32_t max_depth = 0;
            std::vector<DirNode> nodes;
        };

        /**
         * \brief keep track of current and previous direction trees in the same
         * overarching "DTreeWrapper" wrapper to only need one binary search
         * through the spatial tree to reach these leaves. The underlying
         * spatial structure for both trees is the same regardless.
         *
         * From [1]:
         * Section 5.2: Memory Usage:
         *
         * "because the spatial binary tree of L^k is merely a more refined
         * version of the spatial tree of L^{k−1}, it is straightforward to use
         * the same spatial tree for both distributions, where each leaf
         * contains two directional quadtrees; one for L^{k−1} and one for L^k
         */
        DirTree current, prev;
    };

private: /* SpatialTree (whose leaves are DirectionTrees) declaration */
    class SpatialTree {
    public:
        SpatialTree() { nodes.resize(1); }
        ScalarBoundingBox3f bounds;
        void begin_next_tree_iteration();
        void refine(const Float sample_threshold);
        void reset_leaves(const uint32_t max_depth, const Float rho);

        DTreeWrapper &get_leaf(const Point3f &pos, Vector3f *size = nullptr) {
            return const_cast<DTreeWrapper &>(
                const_cast<const SpatialTree *>(this)->get_leaf(pos, size));
        }
        const DTreeWrapper &get_leaf(const Point3f &pos,
                                     Vector3f *size = nullptr) const;

    private:
        struct SNode // spatial-tree-node
        {
            class DTreeWrapper dTree;           // direction tree
            std::array<uint32_t, 2> children{}; // 2 children ids in binary tree
            uint8_t xyz_axis{}; // (0:x, 1:y, 2:z) which axis to split on
            // equal children => no children => is a leaf node
            inline bool bIsLeaf() const { return children[0] == children[1]; }
        };

        void subdivide(const uint32_t parent_idx);
        // representation of the binary tree through an array of indices
        std::vector<SNode> nodes;
    } spatial_tree;
};

MI_EXTERN_CLASS(PathGuide)
NAMESPACE_END(mitsuba)
