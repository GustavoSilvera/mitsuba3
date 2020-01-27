import mitsuba
import pytest
import enoki as ek
from enoki.dynamic import Float32 as Float

from mitsuba.python.test import variant_scalar, variant_packet


def test01_create(variant_scalar):
    from mitsuba.render import BSDFFlags
    from mitsuba.core.xml import load_string

    b = load_string("<bsdf version='2.0.0' type='diffuse'></bsdf>")
    assert b is not None
    assert b.component_count() == 1
    assert b.flags(0) == BSDFFlags.DiffuseReflection | BSDFFlags.FrontSide
    assert b.flags() == b.flags(0)


def test02_eval_pdf(variant_scalar):
    from mitsuba.core import Frame3f
    from mitsuba.render import BSDFContext, BSDFFlags, SurfaceInteraction3f
    from mitsuba.core.xml import load_string

    bsdf = load_string("<bsdf version='2.0.0' type='diffuse'></bsdf>")

    si    = SurfaceInteraction3f()
    si.p  = [0, 0, 0]
    si.n  = [0, 0, 1]
    si.wi = [0, 0, 1]
    si.sh_frame = Frame3f(si.n)

    ctx = BSDFContext()

    for theta in ek.linspace(Float, 0, ek.pi / 2, 20):
        wo = [ek.sin(theta), 0, ek.cos(theta)]

        v_pdf  = bsdf.pdf(ctx, si, wo=wo)
        v_eval = bsdf.eval(ctx, si, wo=wo)[0]
        assert ek.allclose(v_pdf, wo[2] / ek.pi)
        assert ek.allclose(v_eval, 0.5 * wo[2] / ek.pi)


def test03_chi2(variant_packet):
    from mitsuba.python.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain
    from mitsuba.core import ScalarBoundingBox2f

    sample_func, pdf_func = BSDFAdapter("diffuse", '')

    chi2 = ChiSquareTest(
        domain = SphericalDomain(),
        sample_func = sample_func,
        pdf_func = pdf_func,
        sample_dim = 3
    )

    result = chi2.run(0.1)
    # chi2._dump_tables()
    assert result