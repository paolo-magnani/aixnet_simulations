## This library contains all the necessary functions to define the dipole distribution along energy, for all the mass compositions
## The original analysis was done with Sybill, numbers from Emily's thesis 
## EPOS-LHC is now also supported
import numpy as np


## ----- All global definitions go here -----

names = {'p': 1, 'He': 4, 'CNO': 16, 'Fe': 56}
charges = {'p': 1, 'He': 2, 'CNO': 8, 'Fe': 26}
colors = {'p': 'red', 'He': 'orange', 'CNO': 'green', 'Fe': 'blue'}

## dipole energies (in eV)
energies = np.array([4e18, 8e18, 16e18, 32e18, 200e18])
mid_ene = energies[:-1] # for energy binning

## values for plotting the measured value of the dipole from ICRC2023  
dipole_values = (0.017, 0.065, 0.094, 0.17)
dipole_errors = ([0.005, 0.009, 0.019, 0.04], [0.008, 0.012, 0.026, 0.05])


## ----- UTILITIES ------

def dict_cutter(dict, mask):
    new_dict = {}

    for key, value in dict.items():
        new_dict[key] = value[mask]

    return new_dict


def dict_paster(arr_of_dicts):
    new_dict = arr_of_dicts[0].copy()

    for i in range(1, len(arr_of_dicts)):
        for key, value in arr_of_dicts[i].items():
            new_dict[key] = np.concatenate([new_dict[key], value])

    return new_dict


def generate_xmax19(data_dict, xmax_key, ene_key):
    # Energy in EeV!
    data_dict["xmax19"] = data_dict[xmax_key] - 58.*np.log10(data_dict[ene_key]/10.)
    return  # dd


def flatten_distr(energy, seed=1312):
    ## README: energy in EeV

    np.random.seed(seed)
    hist, bin_edges = np.histogram(np.log10(energy))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fraction = np.mean(hist[bin_centers>=1])/np.mean(hist[bin_centers<1])
    labels = np.random.uniform(size=len(energy))

    data_mask = (energy<10)*(labels <= fraction) + (energy>=10)

    return data_mask


## ----- Classes -----

class simulated_dipole:

    ## global dipole
    d10 = 0.049
    beta = 0.97

    ## values for Sybill2.3d (to get EPOS, use the constructor)
    d_r = 0.0097
    beta_r = 1.3
    d_max = [1, 3] # this is the same for both Sybill and EPOS

    ## constructor: model is either Sybill or EPOS, default is Sybill
    def __init__(self, model='Sybill'):

        if model=='EPOS':
            self.beta_r=2.1
            self.d_r=0.0018


    def overall_dipole_dependence(self, E):
        ## energy in EeV
        return self.d10*np.power(E/10, self.beta)
    

    def dipole_dependence(self, name, E, d_max):
        ## energy in EeV
        dip = self.d_r*np.power(E/(charges[name]), self.beta_r)
        cutoff = np.zeros_like(E) + d_max

        return np.min([dip, cutoff], axis=0)
    

    def dipole_rigidity(self, charge, E, d_max):
        ## energy in EeV
        dip = self.d_r*np.power(E/(charge), self.beta_r)
        cutoff = np.zeros_like(E) + d_max

        return np.min([dip, cutoff], axis=0)



# this class calculates 
class SMD_method:

    sd = simulated_dipole()

    ## constructor: model is either Sybill or EPOS, default is Sybill
    def __init__(self, model='Sybill'):
        if model=='EPOS':
            self.sd = simulated_dipole(model='EPOS')

    

    def dipole_value(self, energy, name, d_max):

        if len(energy)!=0:
            dipole = np.sum(self.sd.dipole_rigidity(name, energy, d_max))/len(energy)
            # print(len(energy))
            err = np.sqrt(2/len(energy))

            return dipole, err
        else: return 0, 0 # in case of empty bin


    def quantify_SMD(self, light_ene, light_name, heavy_ene, heavy_name, d_max):

        dip_l, sigma_l = self.dipole_value(light_ene, light_name, d_max)
        dip_h, sigma_h = self.dipole_value(heavy_ene, heavy_name, d_max)

        if dip_l==0 or dip_h==0:
            return -1
        else:
            SMD = np.abs(dip_l-dip_h)/np.sqrt(np.power(sigma_l, 2)+np.power(sigma_h, 2))
            return SMD




class mass_fractions:
    ## README: based on the value given by Emily, this class works with log10 of energy!

    ## masses names
    masses = {'p': 1, 'He': 4, 'CNO': 16, 'Fe': 56}
    
    ## mean values
    mu = {'p': 18.19, 'He': 18.79, 'CNO': [17.83, 19.39]}

    ## standard deviations
    std = {'p': 0.38, 'He': 0.57, 'CNO': [0.22, 0.34]}

    ## constant
    amp = {'p': 0.39, 'He': 0.93, 'CNO': [0.2, 0.37]}

    ## CNO offset
    CNO_c = 0.1

    ## fixed random seed
    seed = 42023

    
    ## constructor: model is either Sybill or EPOS
    def __init__(self, model='Sybill'):

        if model=='EPOS':
            # ...
            self.mu = {'p': 18.24, 'He': 18.85, 'CNO': [17.71, 19.45]}
            self.std = {'p': 0.45, 'He': 0.4, 'CNO': [0.35, 0.34]}
            self.amp = {'p': 0.74, 'He': 0.54, 'CNO': [0.4, 0.61]}
            self.CNO_c = 0


    def fraction_func(self, name, energy):
        if name == 'CNO':
            gauss1 = self.amp[name][0]*1./(np.sqrt(2.*np.pi)*self.std[name][0])*np.exp(-np.power((energy - self.mu[name][0])/self.std[name][0], 2.)/2.)
            gauss2 = self.amp[name][1]*1./(np.sqrt(2.*np.pi)*self.std[name][1])*np.exp(-np.power((energy - self.mu[name][1])/self.std[name][1], 2.)/2.)

            return gauss1 + gauss2 + self.CNO_c
        
        else:
            return self.amp[name]*1./(np.sqrt(2.*np.pi)*self.std[name])*np.exp(-np.power((energy - self.mu[name])/self.std[name], 2.)/2.)


    def extract_all_fractions(self, energy, mass):

        np.random.seed(self.seed)
        final_mask = [False]*len(mass)

        for name in self.masses.keys():

            mass_mask = (mass==self.masses[name])

            if name=='Fe':
                cumulative_frac = (1 - (self.fraction_func('p', energy) + self.fraction_func('He', energy) + self.fraction_func('CNO', energy)))
                labels = np.random.uniform(0, 1, size=len(energy))
                mass_mask = (labels<=cumulative_frac)*mass_mask

            else:
                fractions = self.fraction_func(name, energy)
                labels = np.random.uniform(0, 1, size=len(energy))
                mass_mask =  (labels<=fractions)*mass_mask

            final_mask = final_mask + mass_mask

        return final_mask
    


## this class is used to have data that resemble the measured energy spectrum 
class energy_spectrum:
    ## README: energy in eV!

    # spectrum parameters
    J0 = 1.315e-18
    g1 = 3.29
    g2 = 2.51
    g3 = 3.05
    g4 = 5.1
    E12 = 5e18 # ankle
    E23 = 13e18
    E34 = 46e18 # suppression

    omega = 0.05


    seed = 69 # this can be changed in case is needed

    def spectrum_func(self, energy):

        prod1 = np.power((1+np.power(energy/self.E12, 1/self.omega )), (self.g1-self.g2)*self.omega )
        prod2 = np.power((1+np.power(energy/self.E23, 1/self.omega )), (self.g2-self.g3)*self.omega )
        prod3 = np.power((1+np.power(energy/self.E34, 1/self.omega )), (self.g3-self.g4)*self.omega )

        func = self.J0*np.power(energy/(10**(18.5)), -self.g1)*prod1*prod2*prod3

        return func
    

    ## extract mass fractions
    def spectrum_fraction(self, energy):

        spectrum = self.spectrum_func(energy)*(energy) # the multiplication for energy is necessary because data is already distributed as energy^-1
        ## accept-reject method
        labels = np.random.uniform(np.min(spectrum), np.max(spectrum), size=len(energy))
        mask = (labels<=spectrum)

        return  mask