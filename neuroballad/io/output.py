'''Handle output files of NeuroDriver Sessions
'''
import os
import h5py
from warnings import warn

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .io import IO
from ..utils import raster, read_ND_spike_state

class Output(IO):
    def __init__(self, filename, uids, base_dir='./'):
        '''
        uids = {var: [uids]}
        '''
        super().__init__(filename, base_dir=base_dir)
        self._uids = uids
        self._dt = None
        self._t = None

    @property
    def isReady(self):
        if self.status != 'run':
            return False
        try:
            self.open()
            return True
        except Exception as e:
            self.close()
            return False

    @property
    def var_list(self):
        '''Return variables list

        This is in the format consistent with parameters for NeuroKernel.LPU.OutputProcessors
        e.g. [('I', None), ('g', None)]
        '''
        return [(v, self.uids[v]) for v in self.vars]

    @property
    def uids(self):
        if self.isReady:
            self.open()
            if any([self._uids[var] is None for var in self._uids]):
                for v, uid in self._uids.items():
                    if uid is None:
                        self._uids[v] = self.file_handle['{}/uids'.format(v)][()].astype(str)
        return self._uids

    @property
    def vars(self):
        return np.array(list(self.uids.keys()), dtype=str)

    @property
    def dt(self):
        if self._dt is not None:
            return self._dt
        if self.isReady:
            self.open()
            self._dt = self.file_handle['metadata'].attrs['dt']
        return self._dt

    @property
    def t(self):
        '''return a dictionary of time vector for all variables'''
        if self._t is not None:
            return self._t
        if not self.isReady:
            return self._t
        self.open()
        self._t = {}
        
        data = {}
        for v in self.vars:
            if v == 'spike_state':
                continue # this currently cannot be inferred from output data
            steps = self.file_handle['{}/data'.format(v)].shape[0]
            self._t[v] = np.arange(steps)*self.dt
        return self._t

    def read(self, nodes_ids, vars=None, prune_empty=True):
        ''' read all outputs associated with a given set of nodes'''
        if not self.isReady:
            return None

        if vars is None:
            vars = self.vars
        else:
            vars = np.atleast_1d(vars)
            assert all([k in self.vars for k in vars]), \
                "Desired vars {} not found in output file vars{}".format(vars, self.vars)
        nodes_ids = np.atleast_1d(nodes_ids)
        data = {}
        self.open()
        for v in vars:
            if v == 'spike_state':
                _ss = read_ND_spike_state(self.file_handle)
            data[v] = {}
            for n in nodes_ids:
                if n in self.uids[v]:
                    idx = np.where(self.uids[v] == n)[0]
                    if v == 'spike_state':
                        data[v][n] = _ss[n]
                    else:
                        data[v][n] = self.file_handle['{}/data'.format(v)][:, idx]

        if prune_empty:
            pruned_key = []
            for v in data: # keep only relevant data
                if not data[v]:
                    pruned_key.append(v)
            for key in pruned_key:
                del data[key]
        return data
 
    def plot(self, nodes_ids, force_legend=False,
             cmap=plt.cm.jet, as_heatmap=False, show_ytick=True,
             fig_filename=None, figsize=(10,8)):
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

        nodes_ids = np.atleast_1d(nodes_ids)
        data = self.read(nodes_ids)
        fig, axes = plt.subplots(len(data), 1, figsize=figsize)
        if not as_heatmap:
            colors = cmap(np.linspace(0,1,len(nodes_ids)))
            for var_idx, var in enumerate(data.keys()):
                _ax = axes[var_idx] if len(data) > 1 else axes
                if var == 'spike_state':
                    raster(data[var], ax=_ax, colors=cmap)
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
                    raster(data[var], ax=_ax, colors=cmap)
                    # _spikes = np.zeros((len(data[var]), len(self.t[var])))
                    # psth = []
                    # _window = 2e-2
                    # _interval = _window/2
                    # fmt = lambda x, pos: '%.2f' % (float(x)*_interval)
                    # labels = []
                    # for idx, (node, var_val) in enumerate(data[var].items()):
                    #     _psth, _psth_t = PSTH(var_val, self.dt, 2e-2, _interval)
                    #     labels.append(node)
                    #     psth.append(_psth[:,np.newaxis])
                    # psth = np.concatenate(psth, axis=-1)
                    # im = _ax.imshow(psth.transpose(),
                    #                 cmap=cmap,
                    #                 aspect='auto')
                    # _ax.set_ylim([-0.5, len(data[var])-0.5])
                    # _ax.set_xlim([0, len(_psth_t)])
                    # _ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
                    # _ax.set_yticks(np.arange(len(data[var])))
                    # if show_ytick:
                    #     _ax.set_yticklabels(labels)
                    # plt.colorbar(ax=_ax, mappable=im)
                    # _ax.set_title("{} - PSTH".format(self.filename))
                else:
                    vals = []
                    labels = []
                    for idx, (node, var_val) in enumerate(data[var].items()):
                        vals.append(var_val)
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

        if fig_filename is not None:
            output_figname = os.path.join(self.base_dir, fig_filename)
            fig.savefig(output_figname, dpi=300)
        return fig, axes
