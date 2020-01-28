from .element import Element
class AlphaSynapse(Element):
    element_class = 'synapse'
    states = {}
    params = {'ar': 1.1*1e2, 
              'ad': 1.9*1e3, 
              'reverse': 65.0, 
              'gmax': 3*1e-6}

class PowerGPotGPot(Element):
    element_class = 'synapse'
    states = {}
    params = {'power': 1.0, 
              'slope': 0.02, 
              'saturation': 0.4, 
              'threshold': -55.0, 
              'reverse': 0.0, 
              'gmax': 0.4}

class MorrisLecar(Element):
    element_class = 'synapse'
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
