# This library contains all the necessary functions to create a dipolar modulation in the simulated data
import numpy as np

class dipole:

    d10 = 0.049
    beta = 0.97

    def dipole_dependence(self, E):
        # energy in EeV
        return self.d10*np.power(E/10, self.beta)
    