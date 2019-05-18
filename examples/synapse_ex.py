from neuroballad import * #Import Neuroballad
C = Circuit() #Create a circuit
C.add([0, 2, 4], LeakyIAF()) #Create three neurons
C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
C.join([[1,2],[2,3],[3,4]]) #Join nodes together

C_in_a = InIStep(1, 1., 0.20, 0.21, var='spike_state') #Create current input for node 0
C.sim(1e-2, 1e-4, [C_in_a]) #Use the three inputs and simulate

sim_results = C.collect_results() #Get simulation results
C.visualize_video([0, 2, 4]) #Visualize the neurons into a video

# C.visualize_circuit() #Calculate a 2D Layout for circuit visualization
