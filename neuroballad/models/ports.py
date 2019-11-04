from .element import Element, Input

class OutPort(Element):
    element_class = 'port'
    states = {}
    params = {'port_type': 'gpot',
              'class': 'Port',
              'port_io': 'out',
              'lpu': 'lpu',
              'selector': '/%s/out/%s/' % ('lpu', 'gpot')}

class InPort(Element):
    element_class = 'port'
    states = {}
    params = {'port_type': 'spike',
              'class': 'Port',
              'port_io': 'in',
              'lpu': 'lpu',
              'selector': '/%s/in/%s/' % ('lpu', 'gpot')}
