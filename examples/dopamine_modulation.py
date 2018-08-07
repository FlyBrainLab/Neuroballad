from neuroballad import * #Import Neuroballad
C = Circuit() #Create a circuit
C.add([0, 3], HodgkinHuxley()) #Create two Hodgkin-Huxley neurons
C.add([1], DopamineModulatedAlphaSynapse()) #Create a dopamine-modulated synapse
C.add([2], DopamineLeakyIAF()) #Create three neurons
C.join([[0,1],[2,1],[1,3]]) #Join nodes together

C_in_a = InIStep(0, 40., 0.25, 0.50) #Create current input for node 0
C_in_b = InIStep(2, 40., 0.50, 0.75) #Create current input for node 2
C.sim(1., 1e-4, [C_in_a, C_in_b]) #Use the three inputs and simulate
sim_results = C.collect_results() #Get simulation results
