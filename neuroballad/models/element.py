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

class Input(Element):
    '''Input Processors Wrapper
    '''
    element_class = 'input'
    states = {}
    params = {}

    def __init__(self, name="", experiment_name='', **kwargs):
        self.experiment_name = experiment_name
        self.space = {'name': name}
        self.space.update(self.states)
        self.space.update(self.params)

    def add(self, C, uids, I, t, var=None):
        # name_dict = {'name': i, 'experiment_name': experiment_name}
        # name = tags_to_json(name_dict)
        name = C.encode_name(i)  # DEBUG: `i` is not defined
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
