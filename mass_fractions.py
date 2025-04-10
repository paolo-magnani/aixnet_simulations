### Case study: SYBILL
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
            new_dict[key] = np.concatenate([new_dict[key], value])

    return new_dict


def generate_xmax19(data_dict, xmax_key, ene_key):
    # Energy in EeV!
    data_dict["xmax19"] = data_dict[xmax_key] - 58.*np.log10(data_dict[ene_key]/1e1)
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

### CLASSES
class mass_fractions:
    ## README: based on the value given by Emily, this class works with log10 of energy!

    # masses names
    masses = {'p': 1, 'He': 4, 'CNO': 16, 'Fe': 56}
    
    # mean values
    mu = {'p': 18.19, 'He': 18.79, 'CNO': [17.83, 19.39]}

    # standard deviations
    std = {'p': 0.38, 'He': 0.57, 'CNO': [0.22, 0.34]}

    # constant
    amp = {'p': 0.39, 'He': 0.93, 'CNO': [0.2, 0.37]}

    # CNO offset
    CNO_c = 0.1

    # fixed random seed
    seed = 42023


    def fraction_func(self, name, energy):
        if name == 'CNO':
            gauss1 = self.amp[name][0]*1./(np.sqrt(2.*np.pi)*self.std[name][0])*np.exp(-np.power((energy - self.mu[name][0])/self.std[name][0], 2.)/2.)
            gauss2 = self.amp[name][1]*1./(np.sqrt(2.*np.pi)*self.std[name][1])*np.exp(-np.power((energy - self.mu[name][1])/self.std[name][1], 2.)/2.)

            return gauss1 + gauss2 + self.CNO_c
        
        else:
            return self.amp[name]*1./(np.sqrt(2.*np.pi)*self.std[name])*np.exp(-np.power((energy - self.mu[name])/self.std[name], 2.)/2.)

    '''
    def extract_frac(self, name, energy):
        # using an accept-reject method, seed is fixed a priori

        if name == 'Fe':
            labels = np.random.uniform(size=len(energy))
            cumulative_frac = (self.fraction_func('p', energy) + self.fraction_func('He', energy) + self.fraction_func('CNO', energy)) #*energy # data is distributed as E^-1
            fraction_mask = (labels<=cumulative_frac)

        else:     
            labels = np.random.uniform(size=len(energy))
            fractions = self.fraction_func(name, energy) #*energy # data is distributed as E^-1
            fraction_mask = (labels<=fractions)

        return fraction_mask
    '''

    def extract_all_fractions(self, energy, mass):

        np.random.seed(self.seed)
        final_mask = [False]*len(mass)

        for name in self.masses.keys():

            mass_mask = (mass==self.masses[name])
            # print(name, np.where(mass_mask==False))

            if name=='Fe':
                cumulative_frac = (1 - (self.fraction_func('p', energy) + self.fraction_func('He', energy) + self.fraction_func('CNO', energy)))
                labels = np.random.uniform(0, 0.845, size=len(energy))
                mass_mask = (labels<=cumulative_frac)*mass_mask
                # print(np.max(cumulative_frac))

            else:
                fractions = self.fraction_func(name, energy)
                labels = np.random.uniform(0, 0.845, size=len(energy))
                mass_mask =  (labels<=fractions)*mass_mask

            final_mask = final_mask + mass_mask

        #print(final_mask)
        return final_mask



# this class is used to have data that resemble the measured energy spectrum 
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


    seed = 69

    def spectrum_func(self, energy):

        prod1 = np.power((1+np.power(energy/self.E12, 1/self.omega )), (self.g1-self.g2)*self.omega )
        prod2 = np.power((1+np.power(energy/self.E23, 1/self.omega )), (self.g2-self.g3)*self.omega )
        prod3 = np.power((1+np.power(energy/self.E34, 1/self.omega )), (self.g3-self.g4)*self.omega )

        func = self.J0*np.power(energy/(10**(18.5)), -self.g1)*prod1*prod2*prod3

        return func
    

    # extract mass fractions
    def spectrum_fraction(self, energy):

        #np.random.seed(self.seed)

        spectrum = self.spectrum_func(energy)*(energy) # the multiplication for energy is necessary because data is already having an energy^-1 trend
        # accept-reject method
        labels = np.random.uniform(np.min(spectrum), np.max(spectrum), size=len(energy))
        mask = (labels<=spectrum)

        return  mask

    '''
    def spectrum_distr(self, energy):
        # P(log(x))=P(x)*dx/d(log(x))=P(x)*x*ln(10)
        return self.spectrum_func(energy)*energy*np.log(10) # dependence on log(energy instead of energy)

    # I use Metropolis acceptance to extract data
    # the idea is that the spectrum is not a super-sharp 
    def metropolis_spectrum_fraction(self, energy):
        # using log(energy) so E^-1 distribution is constant
        np.random.seed(self.seed)

        # integral, err = si.quad(self.spectrum_distr, np.log10(3e18), np.log10(200e18))
        # print('Intergral value:', integral,'pm', err)
        prob_spectrum = self.spectrum_distr(energy)# /integral
        prob_original = np.log(10)/np.log(200/3) # this is not the most efficient, but I need a clear visualization!
        
        prob_frac = prob_spectrum/prob_original
        metropolis = np.min([np.ones_like(energy), prob_frac], axis=0)# choose the minimum between the two
        # print(prob_frac)
        labels = np.random.uniform(size=len(energy))

        mask = (labels <= metropolis)

        return mask
    '''


