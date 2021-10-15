import pytest
import numpy as np
from flaremodel import eDist

G_STEPS = 40

def gamma_logspace(g_min, g_max, g_steps):
    return np.logspace(np.log10(g_min), np.log10(g_max), np.int(np.log10(g_max/g_min)*g_steps))

class TestEDistsNorm:

    @pytest.mark.parametrize("thetaE", [10., 20., 30.])
    def test_thermal(self, thetaE, g_min=1., g_max=1e7, g_steps=G_STEPS):

        gamma = gamma_logspace(g_min, g_max, g_steps)

        e_dist = eDist(gamma, 1., [thetaE], "thermal")
        res = np.trapz(e_dist, gamma)
        assert np.abs(1-res) < 1e-2

    @pytest.mark.parametrize("kappa", [2.5, 3.5, 4])
    @pytest.mark.parametrize("kappa_w", [30.])
    def test_kappa(self, kappa, kappa_w, g_min=1., g_max=1e7, g_steps=G_STEPS):

        gamma = gamma_logspace(g_min, g_max, g_steps)

        e_dist = eDist(gamma, 1., [kappa, kappa_w], "kappa")
        res = np.trapz(e_dist, gamma)
        assert np.abs(1-res) < 2e-2

    @pytest.mark.parametrize("p", [1., 2., 3.])
    @pytest.mark.parametrize("g_min", [1., 1e1, 1e2])
    @pytest.mark.parametrize("g_max", [1e3, 1e4, 1e5])
    def test_powerlaw(self, p, g_min, g_max, g_steps=50):

        gamma = gamma_logspace(g_min, g_max, g_steps)

        e_dist = eDist(gamma, 1., [p, g_min, g_max], "powerlaw")
        res = np.trapz(e_dist, gamma)
        assert np.abs(1-res) < 1e-2

    @pytest.mark.parametrize("p1", [1., 2., 3.])
    @pytest.mark.parametrize("p2", [1., 2., 3.])
    @pytest.mark.parametrize("g_b", [1e3, 1e4, 1e5])
    def test_bknpowerlaw(self, p1, p2, g_b, g_min=1e1, g_max=1e6, g_steps=G_STEPS):

        gamma = gamma_logspace(g_min, g_max, g_steps)

        e_dist = eDist(gamma, 1., [p1, p2, g_b, g_min, g_max], "bknpowerlaw")
        res = np.trapz(e_dist, gamma)
        assert np.abs(1-res) < 1e-2

    @pytest.mark.parametrize("p", [1., 2., 3.])
    @pytest.mark.parametrize("g_min", [1., 1e1, 1e2])
    @pytest.mark.parametrize("g_max", [1e3, 1e4, 1e5])
    def test_powerlawexpc(self, p, g_min, g_max, g_steps=50):

        gamma = gamma_logspace(g_min, g_max*10, g_steps)

        e_dist = eDist(gamma, 1., [p, g_min, g_max], "powerlaw")
        e_dist_exp = eDist(gamma, 1., [p, g_min, g_max], "powerlawexpcutoff")

        assert np.std(e_dist*np.exp(-gamma/g_max)/e_dist_exp) < 1e-2

    @pytest.mark.parametrize("p1", [1., 2., 3.])
    @pytest.mark.parametrize("p2", [1., 2., 3.])
    @pytest.mark.parametrize("g_b", [1e3, 1e4, 1e5])
    def test_bknpowerlawexpc(self, p1, p2, g_b, g_min=1e1, g_max=1e6, g_steps=G_STEPS):

        gamma = gamma_logspace(g_min, g_max*10, g_steps)

        e_dist = eDist(gamma, 1., [p1, p2, g_b, g_min, g_max], "bknpowerlaw")
        e_dist_exp = eDist(gamma, 1., [p1, p2, g_b, g_min, g_max], "bknpowerlawexpcutoff")

        assert np.std(e_dist*np.exp(-gamma/g_max)/e_dist_exp) < 1e-2
