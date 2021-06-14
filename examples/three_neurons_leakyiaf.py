from neuroballad import * #Import Neuroballad
dt = 1e-4
dur = 1.
t = np.arange(int(dur/dt))*dt

C = Circuit() #Create a circuit
C.add([0, 2, 4], LeakyIAF(capacitance=1000.,resistance=1.)) #Create three neurons
C.add([1, 3, 5], AlphaSynapse(gmax = 1e1, ad = 0.19, ar = 0.11, reverse=20.)) #Create three excitatory synapses
C.join([[0,1],[1,2],[2,3],[3,4]]) #Join nodes together

C_in_a = InIStep(0, 40., 0.20, 0.40) #Create current input for node 0
C_in_b = InIStep(2, 40., 0.40, 0.60) #Create current input for node 2
C_in_c = InIStep(4, 40., 0.60, 0.80) #Create current input for node 4

C.sim(1., dt, in_list = [C_in_a, C_in_b, C_in_c], execute_in_same_thread=False, preamble=['srun']) #Use the three inputs and simulate
sim_results = C.collect() #Get simulation results