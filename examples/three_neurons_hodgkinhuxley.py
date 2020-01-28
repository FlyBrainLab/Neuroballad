from neuroballad import Circuit, Experiment
from neuroballad.models.neurons import HodgkinHuxley
from neuroballad.models.synapses import AlphaSynapse
from neuroballad.models.inputs import InIStep

C = Circuit() #Create a circuit
C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
edge_list = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [4, 5],
             [5, 0]]
C.join(edge_list) #Join nodes together

Cb = C.copy() #Copy the circuit
Cb.set_experiment('hello') #Rename the experiment
C.merge(Cb) #Merge them together

C_in_a = InIStep(node_id=0, I_val=40., t_start=0.20, t_end=0.40) #Create current input for node 0
C_in_b = InIStep(node_id=2, I_val=40., t_start=0.40, t_end=0.60) #Create current input for node 2
C_in_c = InIStep(node_id=4, I_val=40., t_start=0.60, t_end=0.80) #Create current input for node 4
C_in_d = InIStep(node_id=4, I_val=40., t_start=0.60, t_end=0.80,
                 experiment_name='hello') # Create current input for node 4 in the new experiment

#Use the four inputs and simulate the circuit for 1 second with a dt of 1e-4
C.sim(duration=1., dt=1e-4, in_list=[C_in_a, C_in_b, C_in_c, C_in_d])
#Important: Remove the preamble argument if not in a slurm session
