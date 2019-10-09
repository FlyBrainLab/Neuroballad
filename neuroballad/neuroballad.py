#!/usr/bin/env python

"""
Neuroballad circuit class and components for simplifying Neurokernel/Neurodriver
workflow.
"""

from __future__ import absolute_import
import os
import copy
import h5py
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
import json

mpl.use('agg')

class NeuroballadExecutor(object):
    def __init__(self, config = {}):
        self.path = 'neuroballad_execute.py'

def get_neuroballad_path():
    return os.path.dirname(\
    os.path.abspath(inspect.getsourcefile(NeuroballadExecutor)))

def NeuroballadModelGenerator(model_name, arg_names):
    def __init__(self, **kwargs):
        BaseNeuroballadModel.__init__(self, model_name)
        self.arg_names = arg_names + ['reverse']
        for key, value in kwargs.items():
            if key in arg_names:
                pass
            else:
                print("Warning: ", key, 'is not in the default parameters for', model_name, ':', ', '.join(arg_names))
            setattr(self, key, value)
    def nadd(self, G, i):
        name = 'uid' + str(i)
        node_name = self.__class__.__name__
        dict_struct = {'class': node_name}
        for key in self.arg_names:
            try:
                dict_struct[key] = getattr(self, key)
            except:
                pass
                # Fill in better error handling here
                # print('Required parameter',key,'not found in the definition.')
        G.add_node(name, **dict_struct)
        return G
    return type(model_name, (BaseNeuroballadModel,),{"__init__": __init__, "nadd": nadd})

class BaseNeuroballadModel(object):
    def __init__(self, my_type):
        self._type = my_type

def merge_circuits(X, Y):
    XY = nx.compose(X, Y)
    return XY
        
def populate_models():
    from importlib import import_module
    import neurokernel
    import inspect
    import os
    nk_path = inspect.getfile(neurokernel)
    ndcomponents_path = os.path.join(os.path.join(os.path.dirname(nk_path),'LPU'),'NDComponents')
    comp_types = [f.path for f in os.scandir(ndcomponents_path) if f.is_dir() and 'Models' in os.path.basename(f.path)]    
    for i in comp_types:
        models_path_py = '.'.join(i.split('/')[-4:])
        model_paths = [f.path for f in os.scandir(i) if not f.is_dir() and 'Base' not in os.path.basename(f.path)  and '__init__' not in os.path.basename(f.path)  and '.py' in os.path.basename(f.path)]
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
                print(model_name,'has been already defined in workspace.')
                
class Circuit(object):
    """
    Create a Neuroballad circuit.

    Basic Example
    --------
    >>> from neuroballad import * #Import Neuroballad
    >>> C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
    >>> C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
    >>> C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]) #Join nodes together
    >>> C_in_a = InIStep(0, 40., 0.25, 0.50) #Create current input for node 0
    >>> C_in_b = InIStep(2, 40., 0.50, 0.75) #Create current input for node 2
    >>> C_in_c = InIStep(4, 40., 0.75, 0.50) #Create current input for node 4
    >>> C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c]) #Use the inputs and simulate
    """
    default_tags = {'species': 'None',
                'hemisphere': 'None',
                'neuropil': 'None',
                'circuit': 'None',
                'from': 'None',
                'to': 'None',
                'neurotransmitter': 'None',
                'experiment_id': 'None',
                'rid': 'None',
                'type': 'neuron'}
    def __init__(self, name = '', default_type = np.float64, experiment_name = ''):
        self.G = nx.MultiDiGraph() #Neurokernel graph definition
        self.results = {} #Simulation results
        self.ids = [] #Graph ID's
        self.tracked_variables = [] #Observable variables in circuit
        self.t_duration = 0
        self.t_step = 0
        self.inputs = []
        self.outputs = []
        self.experimentConfig = []
        self.experiment_name = experiment_name
        self.default_type = default_type
        self.ICs = []
        self.name = name


    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name
        Gc = copy.deepcopy(self.G)
        mapping = {}
        for i in self.G.nodes():
            ii = self.json_to_tags(i)
            ii['experiment_name'] = experiment_name
            mapping[i] = self.tags_to_json(ii)
        Gc = nx.relabel_nodes(Gc, mapping)
        self.G = Gc
        for i, val in self.G.nodes(data=True):
            val['name'] = i
        for i in range(len(self.ids)):
            self.ids[i][1] = experiment_name

    def copy(self):
        return copy.deepcopy(self)

    def merge(self, C):
        self.G = merge_circuits(self.G, C.G)
        self.ids += C.ids


    def tags_to_json(self, tags):
        """
        Turns the tags dictionary to a JSON string.
        """
        return json.dumps(tags)

    def json_to_tags(self, tags_str):
        """
        Turns a tags JSON string to the dictionary.
        """
        return json.loads(tags_str)

    def encode_name(self, i, experiment_name = None):
        i = str(i)
        try:
            i = i.decode('ascii')
        except:
            pass
        if experiment_name is None:
            experiment_name = self.experiment_name
        name_dict = {'name': str(i), 'experiment_name': experiment_name}
        name = self.tags_to_json(name_dict)
        return name

    def add(self, name, neuron):
        """
        Loops through a list of ID inputs and adds corresponding components
        of a specific type to the circuit.

        Example
        --------
        >>> C.add([1, 2, 3], HodgkinHuxley())
        """
        if neuron.ElementClass == "input":
            self.experimentConfig.append(neuron.addToExperiment)
        else:
            for i in name:
                if (i in self.ids):
                    raise ValueError('Don''t use the same ID for multiple neurons!')
            for i in name:
                neuron.nadd(self, i, self.experiment_name, self.default_tags)
                self.ids.append([str(i), self.experiment_name])
    def get_new_id(self):
        """
        Densely connects two arrays of circuit ID's.

        Example
        --------
        >>> C.dense_connect_via(cluster_a, cluster_b)
        """
        if self.ids == []:
            return '0'
        else:
            return str(len(self.ids)+1)
        #return next(filter(set(self.ids).__contains__, \
        #            itertools.count(0)))
    def add_cluster(self, number, neuron):
        """
        Creates a number of components of a specific type and returns their
        ID's.

        Example
        --------
        >>> id_list = C.add_cluster(256, HodgkinHuxley())
        """
        cluster_inds = []
        for i in range(number):
            i_toadd = self.get_new_id()
            neuron.nadd(self, i_toadd)
            self.ids.append(i_toadd)
            cluster_inds.append(i_toadd)
        return cluster_inds
    def dense_connect_via(self, in_array_a, in_array_b, neuron, delay = 0.0, via = '', tag = 0,
                          debug = 0):
        """
        Densely connects two arrays of circuit ID's, creating a layer of unique
        components of a specified type in between.

        Example
        --------
        >>> C.dense_join_via(cluster_a, cluster_b, AlphaSynapse())
        """
        for i in in_array_a:
            for j in in_array_b:
                i_toadd = self.get_new_id()
                if debug==1:
                    print('Added neuron ID: ' + str(i_toadd))
                neuron.nadd(self, i_toadd)
                self.ids.append(i_toadd)
                self.join([[i, i_toadd], [i_toadd, j]], delay = delay, via=via, tag = tag)
    def dense_connect(self, in_array_a, in_array_b, delay = 0.0):
        """
        Densely connects two arrays of circuit ID's.

        Example
        --------
        >>> C.dense_connect_via(cluster_a, cluster_b)
        """
        for i in in_array_a:
            for j in in_array_b:
                self.join([[i, j]], delay = delay)
    def dense_join(self, in_array_a, in_array_b, in_array_c, delay = 0.0):
        """
        Densely connects two arrays of circuit ID's, using a third array as the
        matrix of components that connects the two.

        Example
        --------
        >>> C.dense_join_via(cluster_a, cluster_b, cluster_c)
        """
        k = 0
        in_array_c = in_array_c.flatten()
        for i in in_array_a:
            for j in in_array_b:
                self.join([[i, in_array_c[k]], [in_array_c[k], j]]
                          , delay = delay)
                k += 1
    def join(self, in_array, delay = 0.0, via = None, tag = 0):
        """
        Processes an edge list and adds the edges to the circuit.

        Example
        --------
        >>> C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
        >>> C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
        >>> C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
        """
        in_array = np.array(in_array)
        #print(in_array)
        for i in range(in_array.shape[0]):
            if via is None:
                self.G.add_edge(self.encode_name(str(in_array[i,0])),
                                self.encode_name(str(in_array[i,1])), 
                                delay = delay, 
                                tag = tag)
            else:
                self.G.add_edge(self.encode_name(str(in_array[i,0])),
                                self.encode_name(str(in_array[i,1])), 
                                delay = delay,
                                via = via, 
                                tag = tag)
    def fit(self, inputs):
        """
        Attempts to find parameters to fit a certain curve to the output.
        Not implemented at this time.
        """
        pass
    def load_last(self, file_name = 'neuroballad_temp_model.gexf.gz'):
        """
        Loads the latest executed circuit in the directory.

        Example
        --------
        >>> C.load_last()
        """
        self.G = nx.read_gexf('neuroballad_temp_model.gexf.gz')
        self.ids = []
        for i in self.G.nodes():
            self.ids.append(i) # FIX
    def save(self, file_name = 'neuroballad_temp_model.gexf.gz'):
        """
        Saves the current circuit to a file.

        Example
        --------
        >>> C.save(file_name = 'my_circuit.gexf.gz')
        """
        nx.write_gexf(self.G, file_name)

    def compile(self, t_duration, t_step, in_list = None, record = ['V', 'spike_state', 'I']):
        self.t_duration = t_duration
        self.t_step = t_step
        if in_list is None:
            in_list = self.experimentConfig
        run_parameters = [self.t_duration, self.t_step]
        with open('run_parameters.pickle', 'wb') as f:
            pickle.dump(run_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_gexf(self.G, 'neuroballad_temp_model.gexf.gz')
        Nt = int(t_duration/t_step)
        t  = np.arange(0, t_step*Nt, t_step)
        uids = []
        for i in in_list:
            uids.append(self.encode_name(str(i.node_id), experiment_name=i.experiment_name))
        input_vars = []
        for i in in_list:
            input_vars.append(i.var)
        input_vars = list(set(input_vars))
        uids = np.array(list(set(uids)), dtype = 'S')
        Is = {}
        Inodes = {}
        for i in input_vars:
            Inodes[i] = []
        for i in in_list:
            in_name = self.encode_name(str(i.node_id), experiment_name=i.experiment_name)
            if in_name in list(self.G.nodes(data=False)):
                pass
            else:
                print('Not found in node names.')
            Inodes[i.var].append(self.encode_name(str(i.node_id), experiment_name=i.experiment_name))
        for i in input_vars:
            Inodes[i] = np.array(list(set(Inodes[i])), dtype = 'S')
        for i in input_vars:
            Is[i] = np.zeros((Nt, len(Inodes[i])), dtype = self.default_type)

        file_name = 'neuroballad_temp_model_input.h5'
        for i in in_list:
            Is[i.var] = i.add(self, Inodes[i.var], Is[i.var], t)

        with h5py.File(file_name, 'w') as f:
            for i in input_vars:
                # print(i + '/uids')
                i_nodes = Inodes[i]
                """
                try:
                    i_nodes = [i.decode('ascii') for i in i_nodes]
                except:
                    pass
                i_nodes = [self.encode_name(i) for i in i_nodes]
                """
                i_nodes = np.array(i_nodes, dtype = 'S')
                f.create_dataset(i + '/uids', data=i_nodes)
                f.create_dataset(i + '/data', (Nt, len(Inodes[i])),
                                dtype = self.default_type,
                                data = Is[i])
        recorders = []
        for i in record:
            recorders.append((i,None))
        with open('record_parameters.pickle', 'wb') as f:
            pickle.dump(recorders, f, protocol=pickle.HIGHEST_PROTOCOL)
    def sim(self, t_duration, t_step, in_list = None, record = ['V', 'spike_state', 'I'], preamble = []):
        """
        Simulates the circuit for a set amount of time, with a fixed temporal
        step size and a list of inputs.

        Example
        --------
        >>> C.sim(1., 1e-4, InIStep(0, 10., 1., 2.))
        """
        self.compile(t_duration, t_step, in_list, record)
        if not os.path.isfile('neuroballad_execute.py'):
            copyfile(get_neuroballad_path() + '/neuroballad_execute.py',\
                     'neuroballad_execute.py')
        subprocess.call(preamble + ['python','neuroballad_execute.py'])
    def collect_results(self):
        """
        Collects the latest results from the executor. Useful when loading
        a set of results after execution.

        Example
        --------
        >>> C.collect_results()
        """
        import neurokernel.LPU.utils.visualizer as vis
        self.V = vis.visualizer()
        self.V.add_LPU('neuroballad_temp_model_output.h5',
                  gexf_file = 'neuroballad_temp_model.gexf.gz',LPU = 'lpu')
        # print([self.V._uids['lpu']['V']])
    def visualize_video(self, name, config = {}, visualization_variable = 'V',
                        out_name = 'test.avi'):
        """
        Visualizes all ID's using a set visualization variable over time,
        saving them to a video file.

        Example
        --------
        >>> C.visualize_video([0, 2, 4], out_name='visualization.avi')
        """
        uids = []
        if config == {}:
            config = {'variable': visualization_variable, 'type': 'waveform',
                      'uids': [self.V._uids['lpu'][visualization_variable]]}
        for i in name:
            uids.append(i)
        config['uids'] = uids
        self.V.codec = 'mpeg4'
        self.V.add_plot(config, 'lpu')
        self.V.update_interval = 1e-4
        self.V.out_filename = out_name
        self.V.run()
    def visualize_circuit(self, prog = 'dot' , splines = 'line'):
        styles = {
            'graph': {
                'label': self.name,
                #'fontname': 'LM Roman 10',
                'fontsize': '16',
                'fontcolor': 'black',
                #'bgcolor': '#333333',
                #'rankdir': 'LR',
                'splines': splines,
                'model': 'circuit',
                'size': '250,250',
                'overlap': 'false',
            },
            'nodes': {
                #'fontname': 'LM Roman 10',
                'shape': 'box',
                'fontcolor': 'black',
                'color': 'black',
                'style': 'rounded',
                'fillcolor': '#006699',
                #'nodesep': '1.5',
            },
            'edges': {
                'style': 'solid',
                'color': 'black',
                'arrowhead': 'open',
                'arrowsize': '0.5',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'black',
                'splines': 'ortho',
                'concentrate': 'false',
            }
        }
        #G = nx.read_gexf('neuroballad_temp_model.gexf.gz')
        G = self.G
        # G.remove_nodes_from(nx.isolates(G))
        mapping = {}
        node_types = set()
        for n,d in G.nodes(data=True):
            node_types.add( d['name'].rstrip('1234567890') )
        node_nos = dict.fromkeys(node_types, 1)
        for n,d in G.nodes(data=True):
            node_type = d['name'].rstrip('1234567890')
            mapping[n] = d['name'].rstrip('1234567890') + str(node_nos[node_type])
            node_nos[node_type] += 1
        G = nx.relabel_nodes(G,mapping)
        A = nx.drawing.nx_agraph.to_agraph(G)
        #A.graph_attr['fontname']= 'LM Roman 10'
        #A.graph_attr['splines'] = 'ortho'
        #A.graph_attr['bgcolor'] = '#333333'
        A.graph_attr.update(styles['graph'])
        A.write('file.dot')
        for i in A.edges():
            e=A.get_edge(i[0],i[1])
            #e.attr['splines'] = 'ortho'
            e.attr.update(styles['edges'])
            if i[0][:-1] == 'Repressor':
                e.attr['arrowhead'] = 'tee'
        for i in A.nodes():
            n=A.get_node(i)
            print(n)
            #n.attr['shape'] = 'box'
            n.attr.update(styles['nodes'])
        A.layout(prog=prog)
        A.draw('neuroballad_temp_circuit.svg')
        A.draw('neuroballad_temp_circuit.eps')


### Component Definitions



class Element(object):
    ElementClass = 'None'
    states = {}
    params = {}
    
    def __init__(self, name = "", **kwargs):
        self.space = {'name': name}
        self.space.update(self.states)
        self.space.update(self.params)
        if 'initV' in self.space and 'threshold' in self.space:
            initV = self.space['initV']
            threshold = self.space['threshold']
            if initV > threshold:
                self.space['initV'] = np.random.uniform(self.space['reset_potential'], self.space['threshold'])
            else:
                self.space['initV'] = initV
        self.space['class'] = self.__class__.__name__
        
    def nadd(self, C, i, experiment_name, default_tags):
        # name_dict = {'name': i, 'experiment_name': experiment_name}
        # name = tags_to_json(name_dict)
        name = C.encode_name(i)
        self.space['name'] = name
        for i in default_tags.keys():
            if i not in self.space.keys():
                self.space[i] = default_tags[i]
        if 'selector' in self.space:
            self.space['selector'] += str(i)
        space = copy.deepcopy(self.space)
        C.G.add_node(name, **space)
        if 'n' in space:
            del space['n']
            attrs = {name: {'n': self.space['n']}}
            nx.set_node_attributes(C.G, attrs)
        return C.G


class HodgkinHuxley(Element):
    ElementClass = 'neuron'
    states = {'n': 0., 'm': 0., 'h': 1.0}
    params = {}

class ConnorStevens(Element):
    ElementClass = 'neuron'
    states = {'n': 0., 'm': 0., 'h': 1.0, 'a': 0., 'b': 0.}
    params = {}

class LeakyIAF(Element):
    ElementClass = 'neuron'
    states = {'initV': 10001.}
    params = {'resting_potential': 0., 
              'reset_potential': 0., 
              'threshold': 1.0, 
              'capacitance': 0., 
              'resistance': 0.}

class AlphaSynapse(Element):
    ElementClass = 'synapse'
    states = {}
    params = {'ar': 1.1*1e2, 
              'ad': 1.9*1e3, 
              'reverse': 65.0, 
              'gmax': 3*1e-6}

class PowerGPotGPot(Element):
    ElementClass = 'synapse'
    states = {}
    params = {'power': 1.0, 
              'slope': 0.02, 
              'saturation': 0.4, 
              'threshold': -55.0, 
              'reverse': 0.0, 
              'gmax': 0.4}

class MorrisLecar(Element):
    ElementClass = 'synapse'
    states = {'V1': -20.,
              'V2': 50.,
              'V3': -40.,
              'V4': 20.0,
              'initV': -46.080,
              'initn': 0.3525}
    params = {'phi': 0.4,
              'offset': 0., 
              'V_L': -40., 
              'V_Ca': 120., 
              'V_K': -80., 
              'g_L': 3., 
              'g_Ca': 4., 
              'g_K': 16.}

class Activator(Element):
    ElementClass = 'abstract'
    states = {}
    params = {'beta': 1.0, 
              'K': 1.0, 
              'n': 1.0}

class Repressor(Element):
    ElementClass = 'abstract'
    states = {}
    params = {'beta': 1.0, 
              'K': 1.0, 
              'n': 1.0}

class Integrator(Element):
    ElementClass = 'abstract'
    states = {}
    params = {'gamma': 0.0}

class CurrentModulator(Element):
    ElementClass = 'abstract'
    states = {}
    params = {'A': 1.0,
              'shift': 0.0}

class OutPort(Element):
    ElementClass = 'port'
    states = {}
    params = {'port_type': 'gpot',
              'class': 'Port',
              'port_io': 'out',
              'lpu': 'lpu',
              'selector': '/%s/out/%s/' % ('lpu', 'gpot')}

class InPort(Element):
    ElementClass = 'port'
    states = {}
    params = {'port_type': 'spike',
              'class': 'Port',
              'port_io': 'in',
              'lpu': 'lpu',
              'selector': '/%s/in/%s/' % ('lpu', 'gpot')}

### Input Processors

class Input(Element):
    ElementClass = 'input'
    states = {}
    params = {}

    def __init__(self, name = "", experiment_name='', **kwargs):
        self.experiment_name = experiment_name
        self.space = {'name': name}
        self.space.update(self.states)
        self.space.update(self.params)
    
    def add(self, C, uids, I, t):
        # name_dict = {'name': i, 'experiment_name': experiment_name}
        # name = tags_to_json(name_dict)
        name = C.encode_name(i)
        self.space['name'] = name
        if 'selector' in self.space:
            self.space['selector'] += str(i)
        space = copy.deepcopy(self.space)
        C.G.add_node(name, **space)
        if 'n' in space:
            del space['n']
            attrs = {name: {'n': self.space['n']}}
            nx.set_node_attributes(C.G, attrs)
        return C.G
    def addToExperiment(self):
        return self.params

class InIBoxcar(Input):
    ElementClass = 'input'
    def __init__(self, node_id, I_val, t_start, t_end, var = 'I', experiment_name=''):
        self.experiment_name = experiment_name
        self.node_id = node_id
        self.I_val = I_val
        self.t_start = t_start
        self.t_end = t_end
        self.var = var
        a = {}
        a['name'] = 'InIBoxcar'
        a['node_id'] = node_id
        a['I_val'] = I_val
        a['t_start'] = t_start
        a['t_end'] = t_end
        self.params = a
    def add(self, C, uids, I, t):
        try:
            uids = [i.decode('ascii') for i in uids]
        except:
            pass
        step_range = [self.t_start, self.t_end]
        step_intensity = self.I_val
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == (C.encode_name(str(self.node_id), experiment_name=self.experiment_name)) for i in uids])] += step_intensity
        return I
    def addToExperiment(self):
        return self.params

class InIStep(InIBoxcar):
    ElementClass = 'input'

class InIGaussianNoise(Input):
    ElementClass = 'input'
    def __init__(self, node_id, mean, std, t_start, t_end, var='I'):
        self.node_id = node_id
        self.mean = mean
        self.std = std
        self.t_start = t_start
        self.t_end = t_end
        self.var = var
        a = {}
        a['name'] = 'InIGaussianNoise'
        a['node_id'] = node_id
        a['mean'] = mean
        a['std'] = std
        a['t_start'] = t_start
        a['t_end'] = t_end
        self.params = a
    def add(self, uids, I, t):
        step_range = [self.t_start, self.t_end]
        uids = [i.decode("utf-8") for i in uids]
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] += self.mean + self.std*\
        np.array(np.random.randn(len(np.where(np.logical_and(t>step_range[0], \
        t<step_range[1])))))
        return I
    def addToExperiment(self):
        return self.params

class InISinusoidal(object):
    ElementClass = 'input'
    def __init__(self, 
                 node_id, 
                 amplitude, 
                 frequency, 
                 t_start, 
                 t_end, 
                 mean = 0, 
                 shift = 0., 
                 frequency_sweep = 0.0, 
                 frequency_sweep_frequency = 1.0, 
                 threshold_active = 0, 
                 threshold_value = 0.0,
                 var = 'I'):
        self.node_id = node_id
        self.amplitude = amplitude
        self.frequency = frequency
        self.mean = mean
        self.t_start = t_start
        self.t_end = t_end
        self.shift = shift
        self.threshold_active = threshold_active
        self.threshold_value = threshold_value
        self.frequency_sweep_frequency = frequency_sweep_frequency
        self.frequency_sweep = frequency_sweep
        self.var = var
        a = {}
        a['name'] = 'InISinusoidal'
        a['node_id'] = node_id
        a['amplitude'] = amplitude
        a['frequency'] = frequency
        a['t_start'] = t_start
        a['t_end'] = t_end
        a['mean'] = mean
        a['shift'] = shift
        a['frequency_sweep'] = frequency_sweep
        a['frequency_sweep_frequency'] = frequency_sweep_frequency
        a['threshold_active'] = threshold_active
        a['threshold_value'] = threshold_value
        self.params = a
    def add(self, uids, I, t):
        step_range = [self.t_start, self.t_end]
        uids = [i.decode("utf-8") for i in uids]
        sin_wave = np.sin(2 * np.pi * t * (self.frequency + self.frequency_sweep * np.sin(2 * np.pi * t * self.frequency_sweep_frequency)) + self.shift)
        values_to_add = self.mean + self.amplitude * \
                sin_wave[np.logical_and(t>step_range[0], t<step_range[1])]
        if self.threshold_active>0:
            values_to_add[values_to_add>self.threshold_value] = np.max(values_to_add)
            values_to_add[values_to_add<=self.threshold_value] = np.min(values_to_add)
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] += values_to_add
        return I
    def addToExperiment(self):
        return self.params
