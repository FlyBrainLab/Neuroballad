'''Handle input files of NeuroDriver Sessions
'''
import os
import time
import h5py
from warnings import warn
import typing as tp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .io import IO
from ..utils import raster, PSTH

class Input(IO):
    def __init__(self, filename: str, data: tp.Dict, dt: tp.Union[np.float, tp.Any]=None, base_dir: str='./'):
        '''
        data = {variable:{
                 uids: []
                 data: []
        }
        '''
        super().__init__(filename, base_dir=base_dir)

        # coerce uids to be type 'S' as per required by h5 format
        for v in data:
            data[v]['uids'] = np.array(data[v]['uids'], dtype='S')

        self.vars = list(data.keys())
        self.uids = {v:data[v]['uids'].astype(str) for v in self.vars}
        self.shapes = {v:data[v]['data'].shape for v in self.vars}
        if dt is None:
            self.dt = 1
        else:
            self.dt = dt
        self.t = {v:np.arange(self.shapes[v][0])*dt for v in self.vars}

        with h5py.File(self.path, 'w') as f:
            for var in self.vars:
                _uids = np.array(data[var]['uids'], dtype='S')
                _data = data[var]['data']
                f.create_dataset('{}/uids'.format(var),
                                 data=_uids)
                f.create_dataset('{}/data'.format(var),
                                 shape=_data.shape,
                                 dtype=_data.dtype,
                                 data=_data)

    def read(self, nodes_ids, vars=None, prune_empty=True):
        ''' read everything associated with a node '''
        if vars is None:
            vars = self.vars
        else:
            vars = np.atleast_1d(vars)
            for v in vars:
                if v not in self.vars:
                    raise ValueErorr("Desired vars {} not found in output file vars{}".format(
                        vars, self.vars))
                for n in node_ids:
                    if n not in self.uids[v]:
                        raise ValueErorr("Desired var {} of node {} not found".format(
                            v, n))


        nodes_ids = np.atleast_1d(nodes_ids)
        data = {}
        self.open()
        for v in vars:
            data[v] = {}
            for n in nodes_ids:
                if n in self.uids[v]:
                    idx = np.where(self.uids[v] == n)[0]
                    data[v][n] = self.file_handle['{}/data'.format(v)][:, idx]

        if prune_empty: # keep only data fields that are not empty
            pruned_key = []
            for v in data: 
                if not data[v]:
                    pruned_key.append(v)
            for key in pruned_key:
                del data[key]
        return data

    def plot(self, nodes_ids, force_legend=False, cmap=plt.cm.jet,
             as_heatmap=False, show_ytick=True, fig_filename=None):
        '''Plot all input arrays related to a given set of nodes

        Parameters
        ----------
        nodes_ids: iterable or str
            ids of nodes to be plotted
        force_legend: bool
        cmap: matplotlib.cm
            cmap callable from matplotlib
        as_heatmap: bool
            - True: plot all traces as heat map
        show_ytick: bool
            - True: show yticks in heatmap labeled by node ids. Note that this could result in a lot of yticks
        fig_filename: None or str
            if not `None`, figure will be saved in `self.base_dir` under this name.

        Returns
        --------
        fig: matplotlib.figure.Figure or None
            - `None`: if the status of the input file is not `run`
        axes: iterable of matplotlib.figure.Axis or None
            - `None`: if the status of the input file is not `run`
        '''
        if self.status != 'run':
            warn("File stats = {}, session may not have completed running".format(self.status))
            return None, None

        # coerce nodes_ids to be iterable then read data
        nodes_ids = np.atleast_1d(nodes_ids)
        data = self.read(nodes_ids)

        fig, axes = plt.subplots(len(data), 1, figsize=(10,10))
        if not as_heatmap:
            colors = cmap(np.linspace(0,1,len(nodes_ids)))
            for var_idx, var in enumerate(data.keys()):
                _ax = axes[var_idx] if len(data) > 1 else axes
                if var == 'spike_state':
                    for idx, (node, var_val) in enumerate(data[var].items()):
                        raster(var_val, ax=_ax, dt=self.dt, offset=idx, length=0.5, color=colors[idx], names=node)
                else:
                    for idx, (node, var_val) in enumerate(data[var].items()):
                        _ax.plot(self.t[var], var_val, label=node, color=colors[idx])
                    if len(data[var]) < 8 or force_legend:
                        _ax.legend()
                _ax.set_title("{} - {}".format(self.filename, var))
        else:
            for var_idx, var in enumerate(data.keys()):
                _ax = axes[var_idx] if len(data) > 1 else axes
                if var == 'spike_state':
                    _spikes = np.zeros((len(data[var]), len(self.t[var])))
                    psth = []
                    _window = 2e-2
                    _interval = _window/2
                    fmt = lambda x, pos: '%.2f' % (float(x)*_interval)
                    labels = []
                    for idx, (node, var_val) in enumerate(data[var].items()):
                        _psth, _psth_t = PSTH(var_val, self.dt, 2e-2, _interval)
                        labels.append(node)
                        psth.append(_psth[:,np.newaxis])
                    psth = np.concatenate(psth, axis=-1)
                    im = _ax.imshow(psth.transpose(),
                                    cmap=cmap,
                                    aspect='auto')
                    _ax.set_ylim([-0.5, len(data[var])-0.5])
                    _ax.set_xlim([0, len(_psth_t)])
                    _ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
                    _ax.set_yticks(np.arange(len(data[var])))
                    if show_ytick:
                        _ax.set_yticklabels(labels)
                    plt.colorbar(ax=_ax, mappable=im)
                    _ax.set_title("{} - PSTH".format(self.filename))
                else:
                    vals = []
                    labels = []
                    for idx, (node, var_val) in enumerate(data[var].items()):
                        vals.append(var_val[:, np.newaxis])
                        labels.append(node)
                    vals = np.concatenate(vals, axis=-1)
                    fmt = lambda x, pos: '%.2f' % (float(x)*self.dt)
                    im = _ax.imshow(vals.transpose(),
                                    cmap=cmap,
                                    aspect='auto')
                    _ax.set_ylim([-0.5, len(data[var])-0.5])
                    _ax.set_xlim([0, len(self.t[var])])
                    _ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
                    _ax.set_yticks(np.arange(len(data[var])))
                    if show_ytick:
                        _ax.set_yticklabels(labels)
                    plt.colorbar(ax=_ax, mappable=im)
                    _ax.set_title("{} - {}".format(self.filename, var))
        if savefig:
            fig.savefig(os.path.join(RES_DIR, 'output_' + self.filename.split('.h5')[0]+'.png'), dpi=300)
        return fig, axes
