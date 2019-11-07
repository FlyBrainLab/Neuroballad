#!/usr/bin/env python
"""
Neuroballad circuit class and components for simplifying Neurokernel/Neurodriver
workflow.
"""
from __future__ import absolute_import
import os
import copy
import json
import h5py
import time
import random
import pickle
import inspect
import argparse
import itertools
import subprocess
import numpy as np
import networkx as nx
import matplotlib as mpl
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from shutil import copyfile


mpl.use('agg')

class NeuroballadExecutor(object):
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.path = 'neuroballad_execute.py'

def get_neuroballad_path():
    return os.path.dirname(
        os.path.abspath(inspect.getsourcefile(NeuroballadExecutor)))

def NeuroballadModelGenerator(model_name, arg_names):
    def __init__(self, **kwargs):
        BaseNeuroballadModel.__init__(self, model_name)
        self.arg_names = arg_names + ['reverse']
        for key, value in kwargs.items():
            if key in arg_names:
                pass
            else:
                print("Warning: {} is not in the default parameters for {}: {}".format(
                    key, model_name, ', '.join(arg_names)))
            setattr(self, key, value)

    def nadd(self, G, i):
        name = 'uid' + str(i)
        node_name = self.__class__.__name__
        dict_struct = {'class': node_name}
        for key in self.arg_names:
            try:
                dict_struct[key] = getattr(self, key)
            except Exception as e:
                pass
                # Fill in better error handling here
                # print('Required parameter',key,'not found in the definition.')
        G.add_node(name, **dict_struct)
        return G
    return type(model_name, (BaseNeuroballadModel,),
                {"__init__": __init__, "nadd": nadd})

class BaseNeuroballadModel(object):
    def __init__(self, my_type):
        self._type = my_type

def populate_models():
    import os
    import inspect
    from importlib import import_module
    import neurokernel
    nk_path = inspect.getfile(neurokernel)
    ndcomponents_path = os.path.join(os.path.join(os.path.dirname(nk_path),
                                                  'LPU'),
                                     'NDComponents')
    comp_types = [f.path for f in os.scandir(ndcomponents_path) if \
                  f.is_dir() and 'Models' in os.path.basename(f.path)]
    for i in comp_types:
        models_path_py = '.'.join(i.split('/')[-4:])
        model_paths = [f.path for f in os.scandir(i) if \
                       not f.is_dir() and 'Base' not in os.path.basename(f.path) \
                       and '__init__' not in os.path.basename(f.path) \
                       and '.py' in os.path.basename(f.path)]
        for p in model_paths:
            model_name = os.path.basename(p).split('.')[0]
            from_path = models_path_py + '.' + model_name
            mod = import_module(from_path)
            model_class = getattr(mod, model_name)
            if hasattr(model_class, 'params'):
                params = model_class.params
            else:
                params = []
            if model_name not in globals():
                globals()[model_name] = NeuroballadModelGenerator(model_name, params)
            else:
                print(model_name, 'has been already defined in workspace.')
