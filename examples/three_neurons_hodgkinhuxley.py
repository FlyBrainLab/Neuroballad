from neuroballad import * #Import Neuroballad
C = Circuit() #Create a circuit
C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]) #Join nodes together
Cb = C.copy() #Copy the circuit
Cb.set_experiment('hello') #Rename the experiment
C.merge(Cb) #Merge them together

C_in_a = InIStep(0, 40., 0.20, 0.40) #Create current input for node 0
C_in_b = InIStep(2, 40., 0.40, 0.60) #Create current input for node 2
C_in_c = InIStep(4, 40., 0.60, 0.80) #Create current input for node 4
C_in_d = InIStep(4, 40., 0.60, 0.80, experiment_name = 'hello') #Create current input for node 4 in the new experiment

C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c, C_in_d], preamble=['srun'])  #Use the four inputs and simulate the circuit for 1 second with a dt of 1e-4
#Important: Remove the preamble argument if not in a slurm session