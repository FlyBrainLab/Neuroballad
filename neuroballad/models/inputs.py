import numpy as np
from ..neuroballad import Element, Input

class InIBoxcar(Input):
    element_class = 'input'
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
    def add(self, C, uids, I, t, var=None):
        try:
            uids = [i.decode('ascii') for i in uids]
        except:
            pass
        step_range = [self.t_start, self.t_end]
        step_intensity = self.I_val
        I[np.logical_and(t > step_range[0], t < step_range[1]),
          np.where([i == (C.encode_name(str(self.node_id), experiment_name=self.experiment_name)) for i in uids])] += step_intensity
        return I

    def addToExperiment(self):
        return self.params

class InArray(Input):
    element_class = 'input'
    def __init__(self, node_id, I_vals, experiment_name=''):
        self.experiment_name = experiment_name
        self.node_id = node_id
        self.I_vals = I_vals
        a = {}
        a['name'] = 'InArray'
        a['node_id'] = node_id
        a['I_vals'] = I_vals
        self.var = list(I_vals.keys())
        self.params = a

    def add(self, C, uids, I, t, var=None):
        try:
            uids = [i.decode('ascii') for i in uids]
        except:
            pass
        a = I[:, np.where([i == (C.encode_name(str(self.node_id), experiment_name=self.experiment_name)) for i in uids])]
        X = self.params['I_vals'][var].reshape(a.shape)
        I[:, np.where([i == (C.encode_name(str(self.node_id), experiment_name=self.experiment_name)) for i in uids])] += X
        return I

    def addToExperiment(self):
        return self.params

class InIStep(InIBoxcar):
    element_class = 'input'

class InIGaussianNoise(Input):
    element_class = 'input'
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
    def add(self, uids, I, t, var=None):
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
    element_class = 'input'
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
    def add(self, uids, I, t, var=None):
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
