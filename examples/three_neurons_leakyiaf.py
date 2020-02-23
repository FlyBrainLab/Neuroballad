from neuroballad import LeakyIAF, AlphaSynapse, Circuit, InIStep
C = Circuit() #Create a circuit
neuron_ids = C.add_cluster(3, LeakyIAF(), name='iaf') #Create three neurons
synapse_ids = C.add_cluster(3, AlphaSynapse(), name='alpha') #Create three synapses
C.join(list(zip(neuron_ids, synapse_ids))) #Join nodes together

C_in_a = InIStep(neuron_ids[0], 40., 0.20, 0.40) #Create current input for node 0
C_in_b = InIStep(neuron_ids[1], 40., 0.40, 0.60) #Create current input for node 2
C_in_c = InIStep(neuron_ids[2], 40., 0.60, 0.80) #Create current input for node 4
C.compile(duration=1., dt=1e-4, in_list=[C_in_a, C_in_b, C_in_c])
C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c], log='none') #Use the three inputs and simulate

C.input.plot(neuron_ids, fig_filename='test_in.png')
C.output.plot(neuron_ids, fig_filename='test_out.png')
