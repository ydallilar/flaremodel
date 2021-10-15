import pytest 
import numpy as np
import matplotlib.pyplot as plt
from flaremodel import j_nu_brute, a_nu_brute

def j_nu_JOS74(nu, ne, B, p):

    jF = {"3": 0.6981, "2" : 0.7939}
    a = (p-1)/2.
    return 2.7070e-22*(2.7992e6)**a*B**(1+a)*nu**(-a)/(4*np.pi)*jF[str(p)]*(p-1)

def a_nu_JOS74(nu, ne, B, p):

    aF = {"3": 4.1869, "2" : 2.7925}
    a = (p-1)/2.
    return 1.9782e7*(2.7992e6)**a*B**(a+1.5)*nu**(-a-2.5)*aF[str(p)]*(p-1)

class TestJnu:

    @pytest.mark.parametrize("p", [2, 3])
    def test_againstJOS74(self, p, ne=1., B=10):
        nus = np.array([1e14, 1e15, 1e16])
        j_nu_calc = j_nu_brute(nus, ne, B, [p, 1, 1e5], "powerlaw", incang=-1)
        j_nu_est = j_nu_JOS74(nus, ne, B, p)

        assert np.abs(1-np.mean(j_nu_calc/j_nu_est)) < 2e-1 

class TestAnu:

    @pytest.mark.parametrize("p", [2, 3])
    def test_againstJOS74(self, p, ne=1., B=10):
        nus = np.array([1e14, 1e15, 1e16])
        a_nu_calc = a_nu_brute(nus, ne, B, [p, 1, 1e5], "powerlaw", incang=-1)
        a_nu_est = a_nu_JOS74(nus, ne, B, p)

        plt.loglog(nus, a_nu_calc)
        plt.loglog(nus, a_nu_est)
        plt.savefig("test.png")

        assert np.abs(1-np.mean(a_nu_calc/a_nu_est)) < 2e-1

