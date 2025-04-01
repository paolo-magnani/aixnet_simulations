# This library contains all the necessary functions to define the dipole distribution along energy, for all the mass compositions
# analysis done with Sybill, numbers from Emily's thesis  
import numpy as np
import scipy.stats as ss

names = {'p': 1, 'He': 4, 'CNO': 16, 'Fe': 56}
colors = {'p': 'red', 'He': 'orange', 'CNO': 'green', 'Fe': 'blue'}

energies = np.array([4e18, 8e18, 16e18, 32e18, 200e18])
mid_ene = energies[:-1]
dipole_values = [0.017, 0.065, 0.094, 0.17]
dipole_errors = [[0.005, 0.009, 0.019, 0.04], [0.008, 0.012, 0.026, 0.05]]


class simulated_dipole:

    d10 = 0.049
    beta = 0.97

    # values for Sybill2.3d
    d_r = 0.0097
    beta_r = 1.3
    d_max = [1, 3]

    charges = {'p': 1, 'He': 2, 'CNO': 8, 'Fe': 26}


    def overall_dipole_dependence(self, E):
        # energy in EeV
        return self.d10*np.power(E/10, self.beta)
    

    def dipole_dependence(self, name, E, d_max):
        # energy in EeV
        dip = self.d_r*np.power(E/(self.charges[name]), self.beta_r)
        cutoff = np.zeros_like(E) + d_max

        return np.min([dip, cutoff], axis=0)
    

    def dipole_rigidity(self, charge, E, d_max):
        # energy in EeV
        dip = self.d_r*np.power(E/(charge), self.beta_r)
        cutoff = np.zeros_like(E) + d_max

        return np.min([dip, cutoff], axis=0)


class SMD_method:

    sd = simulated_dipole()

    def dipole_value(self, energy, name, d_max):

        dipole = np.sum(self.sd.dipole_rigidity(name, energy, d_max))/len(energy)
        err = np.sqrt(2/len(energy))

        return dipole, err


    def quantify_SMD(self, light_ene, light_name, heavy_ene, heavy_name, d_max):

        dip_l, sigma_l = self.dipole_value(light_ene, light_name, d_max)
        dip_h, sigma_h = self.dipole_value(heavy_ene, heavy_name, d_max)

        SMD = np.abs(dip_l-dip_h)/np.sqrt(np.power(sigma_l, 2)+np.power(sigma_h, 2))

        return SMD
