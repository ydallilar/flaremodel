
# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np 
import matplotlib.pyplot as plt
import lmfit 
import os


from .utils import *
from .sources import *
from .plotctrl import *

__all__ = ["SEDFitLM"]

# Lmfit
class SEDFitLM(lmfit.Minimizer):
    """
    This class is a wrapper around `lmfit.Minimizer`_ class from lmfit package.
    It is designed to provide more friendly interface for SED Fitting purposes.

    Parameters
    ----------
    data : list
        Data in the form of list with nu, L_nu, L_nu_err values, each as 1-D np.ndarray
    p0 : list 
        Initial guess for fit [None]
    bounds_l : list
        List of lower limits [None]
    bounds_h : list
        List of upper limits [None]
    vary : list 
        List of bools, wheter or not to vary a given params [None]
    params: list
        `lmfit.Parameters` class which alternatively overrides; p0, bounds_l, bounds_h, vary
    """
    c_s = ['b', 'r', 'g', 'y', 'm']

    def __init__(self, data, p0=None, bounds_l=None, bounds_h=None, vary=None, params=None):
        self.data = data
        self.p0 = p0
        self.bounds_l = bounds_l
        self.bounds_h = bounds_h
        self.vary = vary
        self.set_models()
        self.organize_data()
        if params is None:
            self.make_params()
        else:
            self.params = params
        super(SEDFitLM, self).__init__(self.chi, self.params, (self.data_store["nu"], self.data_store["nuLnu"], self.data_store["err_nuLnu"],))

    def set_models(self):
        pass

    def make_params(self):
        params = lmfit.Parameters()
        for name, p0, bounds_l, bounds_h, vary in zip(self.param_names, self.p0, self.bounds_l, self.bounds_h, self.vary):
            params.add(name, p0, min=bounds_l, max=bounds_h, vary=vary)

        self.params = params

    def fit_fun(self, nu, p):
        """
        Evaluates the model function for given nu and `lmfit.Parameters`_.
        
        Parameters
        ----------
        nu : np.ndarray
            nu values at which to evalute model functions
        p : `lmfit.Parameters`_
            Parameters at which to evalute model functions
             
        Returns 
        -------
        luminosity : np.ndarray
            Evaluates the total SED for a set of defined models [erg s-1]
        """
        res = np.zeros(nu.shape[0])
        for i in range(len(self.models)):
            res += self.models[i](nu, p)*nu
        return res

    # not squared I think need to check
    def chi(self, p, nu, data, err):
        return (data-self.fit_fun(nu,p))/err

    def organize_data(self):

        self.data_store = { "nu"        :   np.concatenate([data[0] for data in self.data]),
                            "nuLnu"     :   np.concatenate([data[1] for data in self.data]),
                            "err_nuLnu" :   np.concatenate([data[2] for data in self.data])}

    def plot_data(self, ax, kwargs_for_data={}):
        """
        Plot of the data.
        
        Parameters
        ----------
        ax : `matplotlib.pyplot.axis`_
            axis object to which the data plot is added
        kwargs_for_data : dict
            kwargs for `matplotlib.pyplot.errorbar`_
        Returns
        -------
        None
        """
        if "fmt" not in kwargs_for_data.keys():
            kwargs_for_data["fmt"] = "o"
        if "color" in kwargs_for_data.keys():
            color = kwargs_for_data["color"]
            del(kwargs_for_data["color"])
            for n, data in enumerate(self.data):
                ax.errorbar(data[0], data[1], yerr=data[2], color=color[n], **kwargs_for_data)
        else:
            for n, data in enumerate(self.data):
                ax.errorbar(data[0], data[1], yerr=data[2], color="C"+str(n), **kwargs_for_data)
                
    def plot_fun(self, nu, p, ax, kwargs_for_fit={}):
        """
        Creates a plot of the fit using the best fit 
        
        Parameters
        ----------
        nu : np.ndarray
            Frequencies to evalutate SED model
        p : `lmfit.Parameters`_
            Parameters to evaluate the model.
        ax : `matplotlib.pyplot.axis`
             axis object to which the data plot is added
        kwargs_for_fit : dict
            keywords arguments passed to kwargs_for data. if keyword has "model" as a prefix matplotlib keyword is passed to the combined fit, else is passed to individal fits [{}]
        Returns
        -------
        None
        """
        kwargs_for_fit_tmp = kwargs_for_fit.copy()
        if "model_color" in kwargs_for_fit_tmp.keys():
            model_color = kwargs_for_fit_tmp["model_color"]
            del(kwargs_for_fit_tmp["model_color"])
        else:
            model_color = "k"
        if "model_ls" in kwargs_for_fit_tmp.keys():
            model_ls = kwargs_for_fit_tmp["model_ls"]
            del(kwargs_for_fit_tmp["model_ls"])
        else:
            model_ls = "-"
        if "model_lw" in kwargs_for_fit_tmp.keys():
            model_lw = kwargs_for_fit_tmp["model_lw"]
            del(kwargs_for_fit_tmp["model_lw"])
        else:
            model_lw = 1
            
        for i in range(len(self.models)):
            if "color" not in kwargs_for_fit_tmp.keys():
                ax.loglog(nu, nu * self.models[i](nu, p), color=self.c_s[i], **kwargs_for_fit_tmp)
            else:
                if len(kwargs_for_fit_tmp["color"]) < len(self.models):
                    raise ValueError("You need to provide the same amount of colors as there are models. you provided: ", len(kwargs_for_fit_tmp["color"]), " but there are ", len(self.models), " models!")
                colors = kwargs_for_fit_tmp["color"]
                del(kwargs_for_fit_tmp["color"])
                ax.loglog(nu, nu * self.models[i](nu, p), color=colors[i], **kwargs_for_fit_tmp)
        if "ls" in kwargs_for_fit_tmp.keys():
            del(kwargs_for_fit_tmp["ls"])
        if "lw" in kwargs_for_fit_tmp.keys():
            del(kwargs_for_fit_tmp["lw"])
        if "color" in kwargs_for_fit_tmp.keys():
            del(kwargs_for_fit_tmp["color"])
        if "zorder" in kwargs_for_fit_tmp.keys():
            zorder = 1 + kwargs_for_fit_tmp["zorder"]
            del(kwargs_for_fit_tmp["zorder"])
        else:
            zorder = len(self.models) + 1
        ax.loglog(nu, self.fit_fun(nu, p), ls=model_ls, lw=model_lw, color=model_color, zorder=zorder, **kwargs_for_fit_tmp)

    def fit(self, *args, **kwargs):
        """
        Interface to `lmfit.Minimizer.minimize`_. After successful execeution the result is stored in SEDFitLM.MinimizerResult 
        with the same format as `lmfit.Minimizer.MinimizerResult`_.
        
        Example:

        >>> kwargs_for_emcee = {"nsamples":400, "nwalkers":400, "nsteps":1000}
        >>> fitobject.fit(method="emcee", **kwags_for_emcee)
        
        Parameters
        ----------
        *args : 
            args for `lmfit.Minimizer.minimize`_
        **kwargs : 
            kwargs for `lmfit.Minimizer.minimize`_
        
        Returns
        ----------
        None
        """
        self.MinimizerResult = self.minimize(*args,**kwargs)

    def guess(self, **kwargs):
        """
        Estimates the fit using `lmfit.Minimizer.minimize` with "Nelder-Mead" method
        and updates the :py:attr:`self.params`.
        
        Parameters
        ----------
        **kwargs : dict
                 keyword arguements pass self.fit
        Returns 
        -------
        None
        """
        self.fit(method="Nelder", **kwargs)
        self.params = self.MinimizerResult.params
        
    def report_fit(self, **kwargs):
        """
        Wrapper around `lmfit.fit_report`_ to report fit results.
        
        Parameters
        ----------
        kwargs : dict
            kwargs for `lmfit.fit_report`_
        Returns
        -------
        None
        """
        print(lmfit.fit_report(self.MinimizerResult.params, **kwargs))
        print("\n")
        print("red. Chi^2: " + str(self.MinimizerResult.redchi))

    def plot_fit(self, ax=None, nu=np.logspace(10, 20, 50), show=True, xlim=None, ylim=None, kwargs_for_data={}, kwargs_for_fit={}):
        """
        Plots fit results
        
        Parameters
        ----------
        ax : `matplotlib.pyplot.axis`_, default=None
            if None, plot fit will create a new figure and axis object. Otherwise plot will be added to axis object
        nu : np.ndarray, default=np.logspace(10, 20, 50)
            the x values for at which the best fit solution is evaluted
        show : bool, default=True
            if False plt.show is not called.
        xlim : tuple, default=None
            if not None, sets the xlimits of the plot
        ylim : tuple, default=None 
            if not None, sets the ylimits of the plot
        kwargs_for_data : dict, default={}
            Keywords for plt.plot to plot data.
        kwargs_for_fit : dict
            Keyword arguments passed to plt.plot for SED model plot. The plot of the combined fit of all models can be controlled by adding "model" to the keyword, e.g. kwargs_for_fit={"model_lw":10, "lw":5} sets the line width of the combined model to 10, where as the individual model components have line width 5. [{}]
        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        self.plot_data(ax, kwargs_for_data=kwargs_for_data)
        self.plot_fun(nu, self.MinimizerResult.params, ax, kwargs_for_fit=kwargs_for_fit)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        if show: plt.show()
        
    def plot_initial_guess(self, ax=None, p0=None, nu=np.logspace(10, 20, 50), show=True, xlim=None, ylim=None):
        """
        Plots the luminosity of the inital guess provided to the fit object, or the the initial guess provided to the function
        
        
        Parameters
        ----------
        ax : `matplotlib.pyplot.axis`_, default=None
            if None, plot fit will create a new figure and axis object. Otherwise plot will be added to axis object
        p0 : `lmfit.Parameters`_
            List of alternative parameters, if None uses the `lmfit.Parameters`_ after :py:meth:`SEDFitLM.guess`
        nu : np.ndarray, default=np.logspace(10, 20, 50)
            Frequencies at which the best fit solution is evaluted
        show : bool, default=True
            if False plt.show is not called.
        xlim : tuple, default=None
            if not None, sets the xlimits of the plot
        ylim : tuple, default=None 
            if not None, sets the ylimits of the plot
        
        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        self.plot_data(ax)
        if p0 is None:
            p0 = self.params
        self.plot_fun(nu, p0, ax)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        if show: plt.show()

    def edist_fun_from_fit(self, gamma):
        """
        Computes the electron distribution of the best fit for a given gamma.
        
        Parameters
        ----------
        gamma : np.array
            the gamma values at which the electron distribution is evaluted
        Returns
        -------
        dn/dgamma/dV : np.array
            The values of the electron distribution of the best fit, given gamma
        """
        try:
            edist_name = self.flare.electronDistribution.name 
        except:
            raise NotImplementedError("self.flare is not implemented for %s" % self.__class__.__name__)

        try:
            fit_params = self.MinimizerResult.params.copy()
        except:
            raise RuntimeError("Fit has not been performed yet.")

        params_dict = {"thermal": ["theta_E"],
                        "kappa" : ["kappa", "kappa_width"],
                        "powerlaw" : ["p", "gamma_min", "gamma_max"],
                        "bknpowerlaw" : ["p1", "p2", "gamma_b", "gamma_min", "gamma_max"],
                        "powerlawexpcutoff" : ["p", "gamma_min", "gamma_max"],
                        "bknpowerlawexpcutoff" : ["p1", "p2", "gamma_b", "gamma_min", "gamma_max"]}
        
        if self.__class__.__name__ == "PowerLawCoolingBreakLM":
            edist_fun = lambda gamma: self.flare.electronDistribution.get_ngamma(gamma, 10**fit_params["log10_ne"], 
                            [fit_params["p"], fit_params["p"]+1, fit_params["gamma_b"], fit_params["gamma_min"], fit_params["gamma_max"]])
        else:
            edist_fun = lambda gamma: self.flare.electronDistribution.get_ngamma(gamma, 10**fit_params["log10_ne"], [fit_params[x] for x in params_dict[edist_name]])
        trunc_fun = lambda gamma: 0

        if edist_name == "powerlaw" or edist_name == "bknpowerlaw":
            fun = np.piecewise(gamma, [gamma < fit_params["gamma_min"], np.logical_and(fit_params["gamma_max"] > gamma, gamma >= fit_params["gamma_min"]), gamma >= fit_params["gamma_max"]], 
                                        [trunc_fun, edist_fun, trunc_fun])
        elif edist_name.endswith("expcutoff"):
            fun = np.piecewise(gamma, [gamma < fit_params["gamma_min"], gamma >= fit_params["gamma_min"]], 
                                        [trunc_fun, edist_fun])
        else:
            fun = edist_fun(gamma)

        return fun

    def plot_edist(self, ax=None, gamma=np.logspace(0, 8, 200), show=True, xlim=None, ylim=None):
        """
        Plots electron distribution of the best fit
        
        Parameters
        ----------
        ax : `matplotlib.pyplot.axis`_, default=None
            if None, plot fit will create a new figure and axis object. Otherwise plot will be added to axis object
        gamma : np.ndarray, default=np.logspace(0, 8, 200)
            Gamma at which the electron distribution is evaluated from best fit result. 
        show : bool, default=True
            if False plt.show is not called.
        xlim : tuple, default=None 
            if not None, sets the xlimits of the plot
        ylim : tuple, default=None 
            if not None, sets the ylimits of the plot
        
        Returns
        -------
        None
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.loglog(gamma, self.edist_fun_from_fit(gamma))
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_ylabel(r"dN/d$\gamma/dV$")
        ax.set_xlabel(r"$\gamma$")
        if show: plt.show()

