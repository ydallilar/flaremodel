# Licensed under a 3-clause BSD style license - see LICENSE

import matplotlib.pyplot as plt
from cycler import cycler

# plt controls
def set_rcparams(fs=5, wdt=2.5, figsize=(16,9)):
    plt.rcParams['axes.prop_cycle'] = cycler('color', ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'])

    plt.rcParams.update({
            'figure.figsize' : figsize,
            'font.size': fs,
            'xtick.direction': 'in',
            'xtick.labelsize': fs,
            'xtick.major.size': 12,
            'xtick.major.width': wdt,
            'xtick.minor.bottom': True,
            'xtick.minor.size': 6,
            'xtick.minor.width': wdt,
            'ytick.direction': 'in',
            'ytick.labelsize': fs,
            'ytick.major.size': 12,
            'ytick.major.width': wdt,
            'ytick.minor.size': 6,
            'ytick.minor.width': wdt,
            'axes.linewidth': wdt,
            'xtick.minor.pad': 10,
            'xtick.minor.pad': 10})
