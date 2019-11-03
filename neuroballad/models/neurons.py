from ..neuroballad import Element

class HodgkinHuxley(Element):
    element_class = 'neuron'
    states = {'n': 0., 'm': 0., 'h': 1.0}
    params = {}

class ConnorStevens(Element):
    element_class = 'neuron'
    states = {'n': 0., 'm': 0., 'h': 1.0, 'a': 0., 'b': 0.}
    params = {}

class LeakyIAF(Element):
    element_class = 'neuron'
    states = {'initV': 10001.}
    params = {'resting_potential': 0., 
              'reset_potential': 0., 
              'threshold': 1.0, 
              'capacitance': 0., 
              'resistance': 0.}
