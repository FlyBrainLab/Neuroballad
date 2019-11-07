'''
Parent class definition for `Elements` in `Circuit`
'''
import copy
import numpy as np
import networkx as nx


class Element(object):
    element_class = 'None'
    states = {}
    params = {}

    def __init__(self, name="", **kwargs):
        self.space = {'name': name}
        self.space.update(self.states)
        self.space.update(self.params)
        self.space.update(kwargs)
        self.space['type'] = self.element_class
        if 'initV' in self.space and 'threshold' in self.space:
            initV = self.space['initV']
            threshold = self.space['threshold']
            if initV > threshold:
                self.space['initV'] = np.random.uniform(
                    self.space['reset_potential'],
                    self.space['threshold']
                )
            else:
                self.space['initV'] = initV
        self.space['class'] = self.__class__.__name__

    def __repr__(self):
        '''Return ClassName on repr() call
        TODO: maybe use `self.space['class']`
        '''
        clsname = str(self.__class__)
        return repr(clsname).split(".")[-1].split("'")[0]

    def nadd(self, C, _id, experiment_name=None, default_tags=None):
        '''Add multiple component to circuit'''
        name = C.encode_name(_id, experiment_name=experiment_name)
        self.space['name'] = name
        for key, item in default_tags.items():
            if key not in self.space.keys():
                self.space[key] = item
        if 'selector' in self.space:
            self.space['selector'] += str(_id)  # DEBUG: is this right?
        space = copy.deepcopy(self.space)  # TOOD: is this necessary?
        C.G.add_node(name, **space)
        if 'n' in space:  # TODO: that is this for?
            del space['n']
            attrs = {name: {'n': self.space['n']}}
            nx.set_node_attributes(C.G, attrs)
        return C.G

    @property
    def accesses(self):
        '''Mirror NDComponent accesses'''
        if self._ndcomp is not None:
            return self._ndcomp.accesses
        return None

    @property
    def updates(self):
        '''Mirror NDComponent updates'''
        if self._ndcomp is not None:
            return self._ndcomp.updates
        return None


class Input(Element):
    '''Input Processors Wrapper
    '''
    element_class = 'input'
    states = {}
    params = {}

    def __init__(self, name='', experiment_name='', **kwargs):
        self.experiment_name = experiment_name
        self.space = {'name': name}
        self.space.update(self.states)
        self.space.update(self.params)

    def add(self, C, uids, I, t, var=None):
        '''Add input to circuit
        Parameters
        ----------
        C: Circuit
        uids: array of dtype 'S'
        I: numpy ndarray
        t:  time vector
        var: variable
        '''
        raise NotImplementedError('To be implemented by child class')
        # name_dict = {'name': i, 'experiment_name': experiment_name}
        # name = tags_to_json(name_dict)

        # name = C.encode_name(i)  # DEBUG: `i` is not defined
        # self.space['name'] = name
        # if 'selector' in self.space:
        #     self.space['selector'] += str(i)
        # space = copy.deepcopy(self.space)
        # C.G.add_node(name, **space)
        # if 'n' in space:
        #     del space['n']
        #     attrs = {name: {'n': self.space['n']}}
        #     nx.set_node_attributes(C.G, attrs)
        # return C.G
