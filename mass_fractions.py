### Case study: EPOS-LHC
### Define the functions to extract fractions of elements to reconstruct the expected mass spectrum
### The fractions are taken as gaussians a*N(mu, std) + c (offset is always 0)
### For CNO there are two gaussians
### Iron is defined as the remainder of the other fractions, with offsets and constants
### Values are takem from Emily's thesis, chapter 5, page 50
import numpy as np


### UTILITIES
def dict_cutter(dict, mask):
    new_dict = {}

    for key, value in dict.items():
        new_dict[key] = value[mask]

    return new_dict


def dict_paster(arr_of_dicts):
    new_dict = arr_of_dicts[0].copy()

    for i in range(1, len(arr_of_dicts)):
        for key, value in arr_of_dicts[i].items():
            new_dict[key] = np.concatenate(new_dict[key], value)

    return new_dict


### CLASSES
class mass_fractions:
    
    # mean values
    mu = {'p': 18.24, 'He': 18.85, 'CNO': [17.71, 19.45]}

    # standard deviations
    std = {'p': 0.45, 'He': 0.4, 'CNO': [0.35, 0.34]}

    # constant
    amp = {'p': 0.74, 'He': 0.54, 'CNO': [0.4, 0.61]}


    # fixed random seed
    seed = 420
    np.random.seed(seed)


    def fraction_func(self, name, energy):
        if name == 'CNO':
            gauss1 = self.amp[name][0]*1./(np.sqrt(2.*np.pi)*self.std[name][0])*np.exp(-np.power((energy - self.mu[name][0])/self.std[name][0], 2.)/2.)
            gauss2 = self.amp[name][1]*1./(np.sqrt(2.*np.pi)*self.std[name][1])*np.exp(-np.power((energy - self.mu[name][1])/self.std[name][1], 2.)/2.)

            return gauss1 + gauss2
        
        else:
            return self.amp[name]*1./(np.sqrt(2.*np.pi)*self.std[name])*np.exp(-np.power((energy - self.mu[name])/self.std[name], 2.)/2.)


    def extract_frac(self, name, energy):
        # using an accept-reject method, seed is fixed a priori

        if name == 'Fe':
            labels = np.random.uniform(size=len(energy))
            cumulative_frac = self.fraction_func('p', energy) + self.fraction_func('He', energy) + self.fraction_func('CNO', energy)
            fraction_mask = (labels<=cumulative_frac)

        else:     
            labels = np.random.uniform(size=len(energy))
            fractions = self.fraction_func(name, energy)
            fraction_mask = (labels<=fractions)

        return fraction_mask
    

# this class is used to have data that resemble the measured energy spectrum 
class energy_spectrum:

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


    seed = 69
    np.random.seed(seed)

    def spectrum_func(self, energy):

        prod1 = np.power((1+np.power(energy/self.E12, 1/self.omega )), (self.g1-self.g2)*self.omega )
        prod2 = np.power((1+np.power(energy/self.E23, 1/self.omega )), (self.g2-self.g3)*self.omega )
        prod3 = np.power((1+np.power(energy/self.E34, 1/self.omega )), (self.g3-self.g4)*self.omega )

        func = self.J0*np.power(energy/(10**(18.5)), -self.g1)*prod1*prod2*prod3

        return func
    
    def spectrum_fraction(self, energy):

        mf = mass_fractions()

        spectrum = self.spectrum_func(energy)
        # accept-reject method
        labels = np.random.uniform(np.min(spectrum), np.max(spectrum), size=len(energy))
        mask = (labels<=spectrum)

        return  mask



