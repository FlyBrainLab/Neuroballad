from neuroballad import * #Import Neuroballad
C = Circuit() #Create a circuit
C.add([0], HodgkinHuxley()) #Create a neuron
t_duration = 1. # Specify duration
t_step = 1e-4 # Specify dt
Nt = int(t_duration/t_step) # Calculate simulation timeframe
I_in = 30. * np.ones((Nt,1)) # Generate the input
C_in = InArray(0, {'I': I_in}) #Create current input for node 0
C.sim(t_duration, t_step, [C_in], preamble=['srun'])  #Simulate
#Important: Remove the preamble argument if not in a slurm session
uids, data = C.collect() #Collect the results

# Plot our results:

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, t_duration, t_step)
plt.figure()
plt.plot(t, data['out']['V'])
plt.title('Hodgkin-Huxley Neuron')
plt.xlabel('Time [t]')
plt.ylabel('Membrane Potential [mV]')
