from neuroballad import LeakyIAF, AlphaSynapse, Circuit, InIStep
C = Circuit() #Create a circuit
N = 5
neuron_ids = C.add_cluster(N, LeakyIAF(initV=-60.,
                                       reset_potential=-67.,
                                       resting_potential=0.,
                                       threshold=-25.1,
                                       resistance=1e3,
                                       capacitance=0.07),
                           name='iaf') #Create three neurons
synapse_ids = C.add_cluster(N, AlphaSynapse(), name='alpha') #Create three synapses
C.join(list(zip(neuron_ids, synapse_ids))) #Join nodes together

C_in = []
t_start = 0.2
t_end = 0.8
_chunk = 1./len(neuron_ids)*(t_end-t_start)
_start = t_start
for n, _id in enumerate(neuron_ids):
    _end = _start + _chunk
    C_in.append(InIStep(node_id=_id, I_val=0.6, t_start=_start, t_end=_end)) #Create current input for node 0
    _start = _end

C.sim(duration=1., dt=1e-4, in_list=C_in, record=('I', 'g', 'V', 'spike_state'), log='screen', graph_filename=None) #Use the three inputs and simulate

C.input.plot(neuron_ids, fig_filename='test_in.png')
C.output.plot(neuron_ids, fig_filename='test_out_neurons.png')
C.output.plot(synapse_ids, fig_filename='test_out_synapses.png')
