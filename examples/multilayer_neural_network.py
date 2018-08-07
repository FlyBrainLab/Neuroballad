from neuroballad import * #Import Neuroballad
# Create a circuit
C = Circuit()
# Create 784 LeakyIAF neurons and get their ID's
in_neurons = C.add_cluster(784, LeakyIAF())
# Create 32 Hodgkin-Huxley neurons and get their ID's
middle_neurons = C.add_cluster(32, HodgkinHuxley())
# Join nodes together via alpha synapses
C.dense_connect_via(in_neurons, middle_neurons, AlphaSynapse())
# Create 10 more Hodgkin-Huxley neurons and get their ID's
out_neurons = C.add_cluster(10, HodgkinHuxley())
# Join nodes together via alpha synapses
C.dense_connect_via(middle_neurons, out_neurons, AlphaSynapse())
# Create inputs for the first set of neurons
input_list = []
for i in in_neurons:
    input_list.append(InIStep(i, 40., 0.25, 0.50))
# Simulate the circuit
C.sim(1., 1e-4, input_list)
sim_results = C.collect_results() #Get simulation results
