'''Neuroballad Circuit class, which contains the utilities that simplify circuit specification.
'''
import os
import copy
import json
import time
import warnings
import pickle
from collections import namedtuple
from shutil import copyfile
import inspect
import subprocess

import h5py
import numpy as np
import networkx as nx

from neuroballad.models.element import Element, Input
from .visualizer import visualize_circuit, visualize_video

SimConfig = namedtuple('SimConfig',
                       ['dt', 'duration', 'steps', 't', 'device'])

class NeuroballadExecutor(object):
    def __init__(self, config = {}):
        self.path = 'neuroballad_execute.py'

def get_neuroballad_path():
    return os.path.dirname(\
    os.path.abspath(inspect.getsourcefile(NeuroballadExecutor)))

class Circuit(object):
    """
    Create a Neuroballad circuit.

    Basic Example
    --------
    >>>  from neuroballad import * #Import Neuroballad
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

    def __init__(self, name='', dtype=np.float64, experiment_name='', config=None):
        self.G = nx.MultiDiGraph()  # Neurokernel graph definition
        if config is None:
            self.config = SimConfig(duration=None, steps=None,
                                    dt=1e-4, t=None, device=0)
        else:
            self.config = config
        self.node_ids = []  # Graph ID's
        self.tracked_variables = []  # Observable variables in circuit
        self._inputs = None  # input nodes
        self._outputs = None # output nodes
#        self.experiment_config = []
        self.experiment_name = experiment_name
        self.dtype = dtype
#        self.ICs = []
        self.name = name
        self.manager = None  # LPU Manager
        self.logger = None # logger

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
        for i in range(len(self.node_ids)):
            self.node_ids[i][1] = experiment_name

    def get_new_id(self):
        """Generate new ID
        """
        if self.node_ids == []:
            return '1'
        else:
            return str(len(self.node_ids)+1)

    @property
    def nodes(self):
        return self.G.nodes

    def copy(self):
        '''Return Copy of Circuit Instance'''
        return copy.deepcopy(self)

    def merge(self, C):
        '''Merge circuit `C` with current circuit'''
        self.G = nx.compose(self.G, C.G)
        self.node_ids += C.node_ids

    def find_neurons(self, **tags_tofind):
        tags_tofind['type'] = 'neuron'
        return self.filter_nodes_by_tags(tags_tofind)

    def find_synapses(self, **tags_tofind):
        tags_tofind['type'] = 'synapse'
        return self.filter_nodes_by_tags(tags_tofind)

    def find_ports(self, **tags_tofind):
        tags_tofind['type'] = 'port'
        return self.filter_nodes_by_tags(tags_tofind)

    def filter_nodes_by_tags(self, tags_tofind):
        output = []
        for i, val in self.G.nodes(data=True):
            skip = False
            for j in tags_tofind.keys():
                if j not in val:
                    skip = True
                else:
                    if tags_tofind[j] in val[j]:
                        pass
                    else:
                        skip = True
            if not skip:
                output.append(i)
        return output

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

    def encode_name(self, i, experiment_name=None):
        '''Encode node id into json format

        Example
        -------
        >>> a = C.encode_name(0, experiment_name='test')
        >>> a
        '{"name": "0", "experiment_name": "test"}'
        '''
        i = str(i)
        try:
            i = i.decode('ascii')
        except Exception as e:
            pass
            # TODO: use raise ValueError('ASCII decode failed for {}, error {}'.format(i, e))
        if experiment_name is None:
            experiment_name = self.experiment_name

        name_dict = {'name': str(i), 'exp': experiment_name}
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
        if isinstance(neuron, Input):
            self._inputs.append(neuron.addToExperiment)
        else:
            for i in name:
                if i in self.node_ids:
                    raise ValueError(
                        'Don''t use the same ID for multiple neurons!')
            for i in name:
                neuron.nadd(self, i, self.experiment_name, self.default_tags)
                self.node_ids.append([str(i), self.experiment_name])

    def add_cluster(self, number, neuron, name=None):
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
            _id = '{}-{}'.format(name, i_toadd)
            neuron.nadd(self, _id, self.experiment_name, self.default_tags)
            self.node_ids.append(_id)
            cluster_inds.append(_id)
        return cluster_inds

    def dense_connect_variable(self, in_array_a, in_array_b, neuron,
                               delay=0.0, variable='', tag=0, debug=False):
        """
        Densely connects two arrays of circuit ID's, creating a layer of unique
        components of a specified type in between.

        Example
        --------
        >>> C.dense_join_variable(cluster_a, cluster_b, AlphaSynapse())
        """
        for i in in_array_a:
            for j in in_array_b:
                i_toadd = self.get_new_id()
                if debug:
                    print('Added neuron ID: ' + str(i_toadd))
                neuron.nadd(self, i_toadd, self.experiment_name,
                            self.default_tags)
                self.node_ids.append(i_toadd)
                self.join([[i, i_toadd], [i_toadd, j]],
                          delay=delay, variable=variable, tag=tag)

    def dense_connect(self, in_array_a, in_array_b, delay=0.0):
        """Densely connect clusters
        Densely connects two arrays of circuit ID's.

        Example
        --------
        >>> C.dense_connect_variable(cluster_a, cluster_b)
        """
        for i in in_array_a:
            for j in in_array_b:
                self.join([[i, j]], delay=delay)

    def dense_join_variable(self, in_array_a, in_array_b, in_array_c, delay=0.0, variable=None):
        """Densely connect clusters variable intermediary elements
        Densely connects two arrays of circuit ID's, using a third array as the
        matrix of components that connects the two.

        TODO
        ----
        1. Currently only support scalar delay, add component-dependent
            delay

        Example
        --------
        >>> C.dense_join_variable(cluster_a, cluster_b, cluster_c)
        """
        if np.isscalar(delay):
            delay = [delay]*2
        if isinstance(variable, str):
            variable = [variable]*2
        k = 0
        in_array_c = in_array_c.flatten()
        for i in in_array_a:
            for j in in_array_b:
                if variable is not None:
                    self.join([i, in_array_c[k]], delay=delay[0],
                              variable=variable[0])
                    self.join([in_array_c[k], j], delay=delay[1],
                              variable=variable[1])
                else:
                    self.join([i, in_array_c[k]], delay=delay[0])
                    self.join([in_array_c[k], j], delay=delay[1])
                k += 1

    def join(self, in_array, delay=0.0, variable=None, tag=0):
        """
        Processes an edge list and adds the edges to the circuit.

        Example
        --------
        >>> C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
        >>> C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
        >>> C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
        """
        in_array = np.array(in_array)
        # print(in_array)
        for i in range(in_array.shape[0]):
            if variable is None:
                self.G.add_edge(self.encode_name(str(in_array[i, 0])),
                                self.encode_name(str(in_array[i, 1])),
                                delay=delay,
                                tag=tag)
            else:
                self.G.add_edge(self.encode_name(str(in_array[i, 0])),
                                self.encode_name(str(in_array[i, 1])),
                                delay=delay,
                                variable=variable,
                                tag=tag)

    def fit(self, inputs):
        """
        Attempts to find parameters to fit a certain curve to the output.
        Not implemented at this time.
        """
        raise NotImplementedError

    def load_last(self, file_name='neuroballad_temp_model.gexf.gz'):
        """
        Loads the latest executed circuit in the directory.

        Example
        --------
        >>> C.load_last()
        """
        self.G = nx.read_gexf(file_name)
        self.node_ids = []
        for i in self.G.nodes():
            self.node_ids.append(i)  # FIX

    def save(self, file_name='neuroballad_temp_model.gexf.gz'):
        """
        Saves the current circuit to a file.

        Example
        --------
        >>> C.save(file_name = 'my_circuit.gexf.gz')
        """
        nx.write_gexf(self.G, file_name)

    def compile(self, duration, dt=None, steps=None, in_list=None,
                record=('V', 'spike_state', 'I'), extra_comps=None,
                input_filename='neuroballad_temp_model_input.h5',
                output_filename='neuroballad_temp_model_output.h5',
                graph_filename='neuroballad_temp_graph.gexf.gz',
                device=0, sample_interval=1, execute_in_same_thread=True):
        """
        Compiles a neuroballad circuit before execution.

        # Arguments
            duration (float): Simulation duration.
            dt (float): Time step size.
            steps (int): Number of steps to use in simulation. Optional; don't use dt if provided.
            in_list (list): List of inputs to use during compilation.
            record (tuple): Tuple of variables to record. Defaults to ('V', 'spike_state', 'I').
            extra_comps (list): List of new, custom components to include for your simulation.
            input_filename (str): The .h5 file name to use for the input.
            output_filename (str): The .h5 file name to use for recording the output.
            graph_filename (str): Name of the graph file to save the circuit to. Uses the .gexf format.
            device (int): Device to use for execution.
            sample_interval (int): Sampling interval for recording simulation output.
            execute_in_same_thread (bool): Whether to execute the circuit in the current thread.
        """
        if dt is not None:
            if steps is not None:
                warnings.warn("Both 'steps' and 'duration' arguments were specified. 'steps' argument is ignored.")
                steps = int(duration/dt)

            else:
                steps = int(duration/dt)
            t = np.linspace(0, duration, steps)
        else:
            if steps is not None:
                t = np.linspace(0, duration, steps)
                dt = t[1] - t[0]
            else:
                raise ValueError('dt and step cannot both be None')
        self.config = self.config._replace(duration=duration,
                                           steps=steps,
                                           dt=dt,
                                           t=t,
                                           device=device)

        run_parameters = [duration, dt]
        with open('run_parameters.pickle', 'wb') as f:
            pickle.dump(run_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Compile inputs
        if in_list is None:
            in_list = self._inputs
        uids = []
        for i in in_list:
            uids.append(self.encode_name(str(i.node_id),
                                         experiment_name=i.experiment_name))
        input_vars = []
        for i in in_list:
            if isinstance(i.var, list):
                for j in i.var:
                    input_vars.append(j)
            else:
                input_vars.append(i.var)
        input_vars = list(set(input_vars))
        uids = np.array(list(set(uids)), dtype='S')
        Is = {}
        Inodes = {}
        for i in input_vars:
            Inodes[i] = []
        for i in in_list:
            in_name = self.encode_name(str(i.node_id),
                                       experiment_name=i.experiment_name)
            if in_name in list(self.G.nodes(data=False)):
                pass
            else:
                raise ValueError(
                    'Input node {} not found in Circuit.'.format(in_name))

            if isinstance(i.var, list):
                for j in i.var:
                    Inodes[j].append(
                        self.encode_name(str(i.node_id),
                                         experiment_name=i.experiment_name))
            else:
                Inodes[i.var].append(
                    self.encode_name(str(i.node_id),
                                     experiment_name=i.experiment_name))
        for i in input_vars:
            Inodes[i] = np.array(list(set(Inodes[i])), dtype='S')
        for i in input_vars:
            Is[i] = np.zeros((self.config.steps, len(Inodes[i])),
                             dtype=self.dtype)

        for i in in_list:
            if isinstance(i.var, list):
                for j in i.var:
                    Is[j] = i.add(self, Inodes[j], Is[j], t, var=j)
            else:
                Is[i.var] = i.add(self, Inodes[i.var], Is[i.var], t, var=i.var)

        with h5py.File(input_filename, 'w') as f:
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
                i_nodes = np.array(i_nodes, dtype='S')
                f.create_dataset(i + '/uids', data=i_nodes)
                f.create_dataset(i + '/data', (self.config.steps, len(Inodes[i])),
                                 dtype=self.dtype,
                                 data=Is[i])

        recorders = []
        for i in record:
            recorders.append((i,None))
        with open('record_parameters.pickle', 'wb') as f:
            pickle.dump(recorders, f, protocol=pickle.HIGHEST_PROTOCOL)

        if graph_filename is not None:
            nx.write_gexf(self.G, graph_filename)

        if execute_in_same_thread:
            from neurokernel.core_gpu import Manager
            from neurokernel.LPU.LPU import LPU
            # import neurokernel.mpi_relaunch
            from neurokernel.LPU.InputProcessors.FileInputProcessor import  \
                FileInputProcessor
            from neurokernel.LPU.OutputProcessors.FileOutputProcessor import \
                FileOutputProcessor

            input_processor = FileInputProcessor(input_filename)
            (comp_dict, conns) = LPU.graph_to_dicts(self.G)
            output_processor = FileOutputProcessor([(i, None) for i in list(record)],
                                                output_filename,
                                                sample_interval=sample_interval)
            self.manager = Manager()
            self.manager.add(LPU, self.experiment_name, self.config.dt,
                            comp_dict, conns,
                            device=self.config.device,
                            input_processors=[input_processor],
                            output_processors=[output_processor],
                            debug=False,
                            extra_comps=extra_comps if extra_comps is not None else [])

    def sim(self, duration, dt, steps=None, in_list=None,
            record=('V', 'spike_state', 'I'), log=None,
            device=0, sample_interval=1,
            input_filename='neuroballad_temp_model_input.h5',
            output_filename='neuroballad_temp_model_output.h5',
            graph_filename='neuroballad_temp_graph.gexf.gz',
            log_filename='neuroballad_temp_log.log',
            extra_comps=None, preamble=[], args=[], execute_in_same_thread=True):
        """
        Simulates the circuit for a set amount of time, with a fixed temporal
        step size and a list of inputs.

        TODO
        ----
        1. use preamble and args for slurm

        Example
        --------
        >>> C.sim(1., 1e-4, InIStep(0, 10., 1., 2.))
        """
        from neurokernel.tools.logging import setup_logger
        if log is not None:
            screen = False
            file_name = None
            if log.lower() in ['file', 'both']:
                file_name = log_filename
            if log.lower() in ['screen', 'both']:
                screen = True
            self.logger = setup_logger(file_name=file_name, screen=screen)

        self.compile(duration, dt, steps=steps,
                     in_list=in_list, record=record, extra_comps=extra_comps,
                     input_filename=input_filename, output_filename=output_filename,
                     graph_filename=graph_filename,
                     device=device, sample_interval=sample_interval, 
                     execute_in_same_thread=execute_in_same_thread)
        if execute_in_same_thread:
            self.manager.spawn()
            self.manager.start(self.config.steps)
            self.manager.wait()
        else:
            if not os.path.isfile('neuroballad_execute.py'):
                copyfile(get_neuroballad_path() + '/neuroballad_execute.py',\
                        'neuroballad_execute.py')
            subprocess.call(preamble + ['python','neuroballad_execute.py'])

    def collect(self):
        data = {'in': {}, 'out': {}}
        uids = {'in': {}, 'out': {}}
        time.sleep(1.)
        with h5py.File('neuroballad_temp_model_input.h5', 'r') as f:
            for k in f.keys():
                data['in'][k] = f[k]['data']
                uids['in'][k] = f[k]['uids'].astype(str)

        with h5py.File('neuroballad_temp_model_output.h5', 'r') as f:
            for k in f.keys():
                if k != 'metadata':
                    if k == 'spike_state':
                        print(f[k]['data'].keys())
                        data['out'][k] = f[k]['data']['time'][()]
                    else:
                        data['out'][k] = f[k]['data'][()]
                    uids['out'][k] = f[k]['uids'][()].astype(str)
        return uids, data

    def collect_results(self):
        """
        Collects the latest results from the executor. Useful when loading
        a set of results after execution.

        Example
        --------
        >>> C.collect_results()
        """
        import neurokernel.LPU.utils.visualizer as vis
        self.Viz = vis.visualizer()
        self.Viz.add_LPU('neuroballad_temp_model_output.h5',
                       gexf_file='neuroballad_temp_model.gexf.gz', LPU='lpu')


    def visualize_circuit(self, prog='dot', splines='line',
                      filename='neuroballad_temp_circuit.svg'): 
        return visualize_circuit(self,
                                 prog=prog,
                                 splines=splines,
                                 filename=filename)

    def visualize_video(self, name, config={}, visualization_variable='V',
                    out_name='test.avi'):
        visualize_video(self,
                        name=name,
                        config=config,
                        visualization_variable=visualization_variable,
                        out_name=out_name)
