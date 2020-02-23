'''Utility Functions

1. `PSTH`: compute PSTH using spikes in a given window
2. `raster`: plot spikes of a neuron
'''
from warnings import warn
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')

def read_ND_spike_state(file_handle):
    '''Convert Spike State reading from new neurodriver output
    '''
    index=file_handle['spike_state']['data']['index'][:]
    time=file_handle['spike_state']['data']['time'][:]
    uids=file_handle['spike_state']['uids'][:]
    spikes = OrderedDict({uid.decode(): time[index==i] for i, uid in enumerate(uids)})
    return spikes

def raster(spikes_dict, ax=None, colors=None, force_yticks=False):
    '''Raster plot

    Paramters
    ---------
    spikes_dict:
        return from `read_ND_spike_state`
    ax: 
        matplotlib axis to plot raster into
    colors: callable
        used to find color of raster by calling `colors(n)` where `n` is index
        of neuron in `spikes_dict`,
    force_yticks: bool
        - False: yticklabels only reflect neuron ids if 

    Example
    ---------
    >>> spikes_dict = read_ND_spike_state(C.output.file_handle)
    >>> ax = raster(spikes_dict, colors=plt.cm.jet)
    '''
    if colors is None:
        colors = lambda x: 'k'
    if not callable(colors):
        warn('provided colors not callable, default to black')
        colors = lambda x: 'k'

    if ax is None:
        ax = plt.subplot(111)
    for n, (_id, _ss) in enumerate(spikes_dict.items()):
        if len(_ss) > 0:
            ax.plot(_ss, np.full((len(_ss),), n), '|', c=colors(n))

    names = list(spikes_dict.keys())
    if force_yticks:
        ytick_idx = np.arange(len(names))
    else:
        if len(names) > 8:
            ytick_idx = np.floor(np.linspace(0, len(names)-1, 8)).astype(int)
        else:
            ytick_idx = np.arange(len(names))
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(np.array(names)[ytick_idx])
    return ax

def convert_to_legacy_spike_state(spikes_dict, dt, steps):
    '''Convert New ND Spike State to Old format

    Returns a 2D array (steps, number_of_neurons) of binary values indicating 
    whether a spike occurs for a given neuron at corresponding index
    '''
    ss = np.zeros((steps, len(spikes_dict)), dtype=int)
    for n, tk in enumerate(spikes_dict.values()):
        ss[n, tk//dt] = 1
    return ss

def PSTH_legacy(spikes, d_t, window, interval):
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


def raster_legacy(data, ax=None, color=[(0, 0, 0)], offset=0, length=None, linewidth=0.2, dt=1, names=None):
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
