from ..neuroballad import Element

class Activator(Element):
    element_class = 'abstract'
    states = {}
    params = {'beta': 1.0, 
              'K': 1.0, 
              'n': 1.0}

class Repressor(Element):
    element_class = 'abstract'
    states = {}
    params = {'beta': 1.0, 
              'K': 1.0, 
              'n': 1.0}

class Integrator(Element):
    element_class = 'abstract'
    states = {}
    params = {'gamma': 0.0}

class CurrentModulator(Element):
    element_class = 'abstract'
    states = {}
    params = {'A': 1.0,
              'shift': 0.0}
