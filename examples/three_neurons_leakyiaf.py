from neuroballad import * #Import Neuroballad
C = Circuit() #Create a circuit
C.add([0, 2, 4], LeakyIAF()) #Create three neurons
C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
C.join([[0,1],[1,2],[2,3],[3,4]]) #Join nodes together

C_in_a = InIStep(0, 40., 0.20, 0.40) #Create current input for node 0
C_in_b = InIStep(2, 40., 0.40, 0.60) #Create current input for node 2
C_in_c = InIStep(4, 40., 0.60, 0.80) #Create current input for node 4
C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c]) #Use the three inputs and simulate
#C.plot(0) #Plot the first neuron
sim_results = C.collect_results() #Get simulation results
C.visualize_video([0, 2, 4]) #Visualize the neurons into a video
#C.calculate_2d_layout() #Calculate a 2D Layout for circuit visualization
