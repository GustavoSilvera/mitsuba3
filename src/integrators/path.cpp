#include <tuple>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
----------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation many randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../../resources/data/docs/images/integrator/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where a path tracer can be
expected to produce reasonable results: this is the case when the emitters are
easily "accessible" by the contents of the scene. For instance, an interior
scene that is lit by an area light will be considerably harder to render when
this area light is inside a glass enclosure (which effectively counts as an
occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. note:: This integrator does not handle participating media

.. tabs::
    .. code-tab::  xml
        :name: path-integrator

        <integrator type="path">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'path',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
class PathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathIntegrator(const Properties &props) : Base(props) { }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        UInt32 depth                  = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                = !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        // Variable for tracking intermediate radiance for path guiding
        std::vector<std::tuple<Point3f, Vector3f, Spectrum, Spectrum, Spectrum, Float>> pg_vars;
        pg_vars.reserve(m_max_depth);

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.

           The first argument identifies the loop by name, which is helpful for
           debugging. The subsequent list registers all variables that encode
           the loop state variables. This is crucial: omitting a variable may
           lead to undefined behavior. */
        dr::Loop<Bool> loop("Path Tracer", sampler, ray, throughput, result,
                            eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active, pg_vars);

        /* Inform the loop about the maximum number of loop iterations.
           This accelerates wavefront-style rendering by avoiding costly
           synchronization points that check the 'active' flag. */
        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            /* dr::Loop implicitly masks all code in the loop using the 'active'
               flag, so there is no need to pass it to every function */

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));

            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr))) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                result = spec_fma(
                    throughput,
                    ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf,
                    result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            BSDFPtr bsdf = si.bsdf(ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // ------------------------ Path Guiding ------------------------
            if (this->pg.enabled() && this->pg.ready()) {

                /// NOTES:
                // bsdf_val, bsdf_pdf is eval(si.p, wo) of emitted bounce, so
                // basically ignore them for pathguiding!

                // if we sample a delta reflection, there is 0 probability of
                // path guiding, so ignore
                if (dr::any_or<true>(!has_flag(bsdf_sample.sampled_type,
                                               BSDFFlags::Delta))) {
                    // flip a coin to enable mixture sampling between
                    // pathguiding and BSDF sampling
                    const Float alpha = 0.5f;
                    Float pg_pdf      = 0.f;
                    Float bs_pdf      = 0.f;
                    Spectrum fs       = 0.f;
                    Vector3f pg_wo;
                    if (dr::any_or<true>(sampler->next_1d() < alpha)) {
                        // update the pathguide-recommended sample values
                        std::tie(pg_wo, pg_pdf) = this->pg.sample(si.p, sampler);
                        // convert world-aligned dir to surface-aligned dir
                        pg_wo = si.to_local(pg_wo);
                        // evaluate bsdf with the pathguide recommended dir
                        std::tie(fs, bs_pdf) = bsdf->eval_pdf(bsdf_ctx, si, pg_wo);
                    } else {
                        // use the bsdf_sample & bsdf_weight as the bsdf terms
                        pg_wo  = bsdf_sample.wo;
                        bs_pdf = bsdf_sample.pdf;

                        // the bsdf_weight is the "BSDF value divided by the
                        // probability (multiplied by the cosine foreshortening
                        // factor)" so we do the opposite here (note that
                        // foreshortening cancels out)
                        fs = bsdf_weight * bsdf_sample.pdf;
                        // now with the bsdf value, we can use the pdf mixture
                        pg_pdf = this->pg.sample_pdf(si.p, si.to_world(pg_wo));
                    }

                    // mix together the bsdf probability and the pg probability
                    Float pdf_mixture = dr::lerp(bs_pdf, pg_pdf, alpha);
                    if (dr::any_or<true>(pdf_mixture > 0.f)) {
                        bsdf_weight     = (fs / pdf_mixture);
                        bsdf_sample.wo  = pg_wo;
                        bsdf_sample.pdf = pdf_mixture;
                    }
                }
            }

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                result[active_em] = spec_fma(
                    throughput, bsdf_val * em_weight * mis_em, result);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));
            if (this->pg.enabled() && this->pg.ready()) {
                #if DEBUG
                valid_ray=true;
                auto rgb2spec = [](Spectrum &s, const Color3f &c){
                    static_assert(!is_monochromatic_v<Spectrum>, "Is spectral");
                    s[0] = c.x(); s[1] = c.y(); s[2] = c.z();
                };
                rgb2spec(result, dr::clamp(ray.d, 0.f, 1.f));
                break;
                #endif
            }

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            if (this->pg.enabled() && !this->pg.ready() &&
                dr::any_or<true>(prev_bsdf_pdf > 0.f && !prev_bsdf_delta && valid_ray)) {
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
                if (pg_vars.size() > 0) {
                    auto &[o, d, _, result_prev, T, woPdf] = pg_vars.back();
                    // delta between result computes the lighting for this path
                    path_radiance = result - result_prev;
                }
                pg_vars.emplace_back(ray.o, ray.d, path_radiance, result,
                                     throughput, bsdf_sample.pdf);
            }

            active = active_next && (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f);
        }

        if (this->pg.enabled() && !this->pg.ready()) {
            /// NOTE:
            // at each bounce we track how much radiance we have seen so far,
            // and at the end we have the total radiance (including NEE) from
            // end to eye so we can subtract what we've seen. This will give us
            // the sum of the remaining NEE paths until the end (from the
            // beginning) but we want the incident radiance starting from this
            // bounce, so we then divide by the current throughput seen so far
            // to cancel out those terms.
            auto to_rgb = [&](const Spectrum &spec) {
                Color3f rgb;
                if constexpr (is_monochromatic_v<Spectrum>) {
                    rgb = spec.x();
                } else if constexpr (is_rgb_v<Spectrum>) {
                    rgb = spec;
                } else {
                    static_assert(is_spectral_v<Spectrum>);
                    auto pdf = pdf_rgb_spectrum(ray.wavelengths);
                    auto unpol_spec =
                        spec * dr::select(dr::neq(pdf, 0.f), dr::rcp(pdf), 0.f);
                    rgb = spectrum_to_srgb(unpol_spec, ray.wavelengths, active);
                }
                return rgb;
            };
            bool final_found        = false;
            Spectrum final_radiance = 0.f;
            for (auto r_it = pg_vars.rbegin(); r_it != pg_vars.rend(); r_it++) {
                // add indirect lighting, o/w pathguide strongly prefers direct
                const auto &[o, d, path_radiance, _, thru, woPdf] = (*r_it);

                if (!final_found &&
                    dr::any_or<true>(luminance(to_rgb(path_radiance)) > 0.f)) {
                    // once the latest path-radiance is computed (last non-zero
                    // path-radiance) use this path radiance for the indirect
                    // lighting of all previous bounces
                    final_radiance = path_radiance;
                    final_found    = true;
                    continue; // don't record this bounce (direct illumination)
                }
                const Spectrum radiance = (final_radiance / thru) / woPdf;
                this->pg.add_radiance(o, d, to_rgb(radiance), sampler);
            }
        }

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(PathIntegrator, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(PathIntegrator, "Path Tracer integrator");
NAMESPACE_END(mitsuba)
