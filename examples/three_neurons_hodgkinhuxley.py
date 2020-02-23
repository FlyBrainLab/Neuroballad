from neuroballad import Circuit
from neuroballad.models.neurons import HodgkinHuxley
from neuroballad.models.synapses import AlphaSynapse
from neuroballad.models.inputs import InIStep

C = Circuit() #Create a circuit
neuron_ids = C.add_cluster(3, HodgkinHuxley(), name='hhn') #Create three neurons
synapse_ids = C.add_cluster(3, AlphaSynapse(), name='alpha') #Create three synapses
all_ids = neuron_ids + synapse_ids
edge_list = list(zip(all_ids, all_ids[1:]+[all_ids[-1]]))
C.join(edge_list) # Join nodes together

C_in_a = InIStep(node_id=neuron_ids[0], I_val=40., t_start=0.20, t_end=0.40) #Create current input for node 0
C_in_b = InIStep(node_id=neuron_ids[0], I_val=40., t_start=0.40, t_end=0.60) #Create current input for node 2
C_in_c = InIStep(node_id=neuron_ids[0], I_val=40., t_start=0.60, t_end=0.80) #Create current input for node 4

#Use the four inputs and simulate the circuit for 1 second with a dt of 1e-4
#C.compile(duration=1., dt=1e-4, in_list=[C_in_a, C_in_b, C_in_c])
C.sim(duration=1., dt=1e-4, in_list=[C_in_a, C_in_b, C_in_c], log='screen') #Use the three inputs and simulate

C.input.plot(neuron_ids, fig_filename='test_in.png')
C.output.plot(neuron_ids, fig_filename='test_out.png')
