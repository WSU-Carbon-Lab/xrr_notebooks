"""
Helper functions for reflectivity fitting analysis.

This module contains utility functions for calculating fitting statistics
and analyzing reflectivity data.
"""

import numpy as np


def reduced_chi2(objective):
    """
    Calculate reduced chi-squared statistic.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate chi-squared for

    Returns
    -------
    float
        Reduced chi-squared value
    """
    ndata = len(objective.data.s.x) + len(objective.data.p.x)
    nparams = len(objective.varying_parameters())
    return objective.chisqr() / (ndata - nparams)


def aic(objective):
    """
    Calculate Akaike Information Criterion.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate AIC for

    Returns
    -------
    float
        AIC value
    """
    nparams = len(objective.varying_parameters())
    return objective.chisqr() + 2 * nparams


def bic(objective):
    """
    Calculate Bayesian Information Criterion.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate BIC for

    Returns
    -------
    float
        BIC value
    """
    ndata = len(objective.data.s.x) + len(objective.data.p.x)
    nparams = len(objective.varying_parameters())
    return objective.chisqr() + nparams * np.log(ndata)


def rxr(x, model, pol):
    """
    Calculate reflectivity for a given polarization.

    Parameters
    ----------
    x : array-like
        Q values
    model : refnx.reflect.ReflectModel
        The reflectivity model
    pol : str
        Polarization ('s' or 'p')

    Returns
    -------
    array-like
        Reflectivity values
    """
    _pol = model.pol
    model.pol = pol
    y = model(x)
    model.pol = _pol
    return y


def anisotropy(x, model):
    """
    Calculate anisotropy from model.

    Parameters
    ----------
    x : array-like
        Q values
    model : refnx.reflect.ReflectModel
        The reflectivity model

    Returns
    -------
    array-like
        Anisotropy values: (R_p - R_s) / (R_p + R_s)
    """
    r_s = rxr(x, model, "s")
    r_p = rxr(x, model, "p")
    return (r_p - r_s) / (r_p + r_s)
