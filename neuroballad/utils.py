'''Utility Functions

1. `PSTH`: compute PSTH using spikes in a given window
2. `raster`: plot spikes of a neuron
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')

def PSTH(spikes, d_t, window, interval):
    """
    Compute the peri-stimulus time histogram.
    Arguments:
        spikes (ndarray): spike sequences.
        d_t (float): time step.
        window (float): the size of the window.
        interval (float): the time shift between two consecutive windows.
    Returns:
        rates (ndarray): the average spike rate for each windows.
        stamps (ndarray): the time stamp for each windows.

    Note:
        taken from https://github.com/chungheng/neural/blob/f952aff18369e23f96e6dc7ac46e9792cb9aa2f5/neural/utils.py#L135
    """
    if len(spikes.shape) > 1:
        axis = int(spikes.shape[0] > spikes.shape[1])
        spikes = np.mean(spikes, axis=axis)

    cum_spikes = np.cumsum(spikes)

    start = np.arange(0., d_t*len(spikes)-window, interval) // d_t
    stop = np.arange(window, d_t*len(spikes)-d_t, interval) // d_t
    start = start.astype(int, copy=False)
    stop = stop.astype(int, copy=False)

    start = start[:len(stop)]

    rates = (cum_spikes[stop] - cum_spikes[start]) / window
    stamps = np.arange(0, len(rates)*interval-d_t, interval)

    return rates, stamps


def raster(data, ax=None, color=[(0, 0, 0)], offset=0, length=None, linewidth=0.2, dt=1, names=None):
    '''Create Raster Plot

    Paramters
    ---------
    data: ndarray of binary with shape (time, neurons)
        spike state data
    '''
    fmt = lambda x, pos: '%.2f' % (float(x)*dt)
    if length is None:
        if data.ndim == 1:
            length = 0.8
        else:
            length = np.ones(data.shape[1])/2.0
    if data.ndim == 1: #if only 1 dim vector, i.e. single neuron
        if not names:
            names = ['0']
        if isinstance(names, str):
            names = [names]

        positions = np.nonzero(data)[0]
        if positions.size == 0:
            positions = np.array([0])
        if ax is None:
            ax = plt.eventplot(positions,
                               colors=color,
                               lineoffsets=offset,
                               linelengths=length,
                               linewidth=linewidth)
        else:
            ax.eventplot(positions,
                         colors=color,
                         lineoffsets=offset,
                         linelengths=length,
                         linewidth=linewidth)
    else:
        if not names:
            names = [str(n) for n in range(data.shape[0])]
        positions = [np.nonzero(data[:, i])[0] for i in range(data.shape[1])]
        for k in range(len(positions)):
            if positions[k].size == 0:
                positions[k] = np.array([0])
        if ax is None:
            ax = plt.eventplot(positions,
                               colors=color,
                               lineoffsets=np.arange(data.shape[1]) + offset,
                               linelengths=length,
                               linewidth=linewidth)
        else:
            ax.eventplot(positions,
                         colors=color,
                         lineoffsets=np.arange(data.shape[1]) + offset,
                         linelengths=length,
                         linewidth=linewidth)
    locs = ax.get_yticks()
    labels = [item.get_text() for item in ax.get_yticklabels(which='both')]
    if any([k % 1 != 0 for k in locs]):
        locs = []
        labels = []
    locs = np.append(locs, np.arange(len(names))+offset)
    labels += names
    ax.set_yticks(locs)
    ax.set_yticklabels(labels)

    ax.set_xlim(0, data.shape[0])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    return ax
