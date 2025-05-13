## This library contains all the necessary functions to define the dipole distribution along energy, for all the mass compositions
## The original analysis was done with Sybill, numbers from Emily's thesis 
## EPOS-LHC is now also supported
import numpy as np
import pandas as pd
import pickle


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

def file_loader(estimator): # load files choosing the right estimator; further changes are needed to use one specific data set (e.g., using EPOS-LHC)

    dd = {}

    if estimator=='AixNet':
        input_path = '/mnt/c/Users/paolo/Desktop/LAVORO/data_files/aixnet_sib_all_data.npz'
        data = np.load(input_path, allow_pickle=True)
        print(data.files)
        print(len(data['dnn_xmax']))
        dd = {key: data[key] for key in data.files}

    elif estimator=='KAne':
        input_kane = '../berenika_dipole/dataset/KAne_EPOS_sims.csv'
        kf = pd.read_csv(input_kane)
        dd = {}
        dd['dnn_xmax'] = np.array(kf['Xmax'])
        dd['zenith'] = np.array(np.pi/2 - kf['Zenith']) # I use the same angle definition 
        dd['energy'] = np.array(kf['Energy_MC']/1e18)
        dd['primary'] = np.array(kf['Primary'])
        dd['mass'] = np.empty_like(dd['energy'])
        dd['mass'][np.where(dd['primary']==0)] = 1
        dd['mass'][np.where(dd['primary']==1)] = 4
        dd['mass'][np.where(dd['primary']==2)] = 16
        dd['mass'][np.where(dd['primary']==3)] = 56

    elif estimator=='AixNet_EPOS':
        input_aix = '../berenika_dipole/dataset/AixNet_EPOSoldP_sims.csv'
        kf = pd.read_csv(input_aix)
        dd = {}
        dd['dnn_xmax'] = np.array(kf['Xmax'])
        # dd['zenith'] = np.array(np.pi/2 - kf['Zenith'])
        dd['energy'] = np.array(kf['mc_energy'])
        dd['primary'] = np.array(kf['Primary'])
        dd['mass'] = np.empty_like(dd['energy'])
        dd['mass'][np.where(dd['primary']==0)] = 1
        dd['mass'][np.where(dd['primary']==1)] = 4
        dd['mass'][np.where(dd['primary']==2)] = 16
        dd['mass'][np.where(dd['primary']==3)] = 56

    elif estimator=='AixNet_SYBILL':
        input_aix = '../berenika_dipole/dataset/AixNet_SIBYLL23d_sims.csv'
        kf = pd.read_csv(input_aix)
        dd = {}
        dd['dnn_xmax'] = np.array(kf['aix_xmax_raw'])
        dd['zenith'] = np.array(kf['sd_zenith']) # np.array(np.pi/2 - kf['Zenith'])
        dd['energy'] = np.array(kf['mc_energy'])
        dd['primary'] = np.array(kf['Primary'])
        dd['mass'] = np.empty_like(dd['energy'])
        dd['mass'][np.where(dd['primary']==0)] = 1
        dd['mass'][np.where(dd['primary']==1)] = 4
        dd['mass'][np.where(dd['primary']==2)] = 16
        dd['mass'][np.where(dd['primary']==3)] = 56
        print(dd.keys())
        print(len(dd['energy']))

    elif estimator=='Universality':
        input_univ = '/mnt/c/Users/paolo/Desktop/LAVORO/data_files/pickle/pickle/'
        filenames = ['proton_phase1.pickle', 'helium_phase1.pickle', 'oxygen_phase1.pickle', 'iron_phase1.pickle']
        filemasses = [1, 4, 16, 56]
        temp_dict = [{} for i in range(4)]

        for i in range(4):

            with open(input_univ +filenames[i], 'rb') as f:
                temp_dict[i] = pickle.load(f)
            
            temp_dict[i]['mass'] = np.zeros_like(temp_dict[i]['univ_xmax']) + filemasses[i]
            temp_dict[i]['energy'] = temp_dict[i]['sd_e']/1e18
            temp_dict[i]['univ_rmu'] = temp_dict[i]['univ_rmu']

        dd = dict_paster(temp_dict)

    else:
        print('ERROR: mass estimator not given')
        return

    return dd 


def remove_nan_entries(d): # ChatGPT wrote this for me
    # Create a mask of valid (non-NaN) entries based on one key (e.g., the first one)
    keys = list(d.keys())
    if not keys:
        return d  # empty dict
    
    # Start with all entries being valid
    mask = ~np.isnan(d[keys[0]])

    # Combine masks if any other key has NaNs
    for key in keys[1:]:
        mask &= ~np.isnan(d[key])

    # Apply mask to all entries
    return {key: np.array(d[key])[mask] for key in d}


## this function cuts dictionaries according to a specific mask
def dict_cutter(dict, mask):
    new_dict = {}

    for key, value in dict.items():
        new_dict[key] = value[mask]

    return new_dict

## this function pastes dictionaries together (I tuse this to get the overall data set when I extract bin per bin)
def dict_paster(arr_of_dicts):
    new_dict = arr_of_dicts[0].copy()

    for i in range(1, len(arr_of_dicts)):
        for key, value in arr_of_dicts[i].items():
            new_dict[key] = np.concatenate([new_dict[key], value])

    return new_dict


def generate_xmax19(data_dict, xmax_key, ene_key):
    ## Energy in EeV!
    data_dict["xmax19"] = data_dict[xmax_key] - 58.*np.log10(data_dict[ene_key]/10.)
    return 

def generate_lnA(data_dict, xmax_key, rmu_key, ene_key, values='Emily'):
    ## Energy in EeV!

    if values=='Emily':
        ## to be further completed
        lam0 = 16.45
        bet0 = 0.013
        xmax0 = 797.16
        ln_rmu0 = 0.26
        sigma_xm0 = 68.2
        sigma_rmu0 = 0.214


    lam = 16.45 + 0.38*np.log10(data_dict[ene_key]/10)
    bet = 0.013 + 0.006*np.log10(data_dict[ene_key]/10)

    xmax_p = 797.16 + 47.1*np.log10(data_dict[ene_key]/10)
    ln_rmu = 0.26 - 0.0176*np.log10(data_dict[ene_key]/10)

    sigma_xm = 68.2 + (data_dict[xmax_key]-xmax_p)*23.1/(lam*np.log(56))
    sigma_rmu = 0.214 + (data_dict[rmu_key]-ln_rmu)*(-0.057)/(bet*np.log(56))

    phi_0 = (bet*sigma_xm/sigma_rmu)**2/lam

    lnA = (phi_0*(data_dict[rmu_key]-ln_rmu)-bet*(data_dict[xmax_key]-xmax_p))/(bet*(lam+phi_0))

    data_dict['lnA'] = lnA
    return


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



## this class calculates the standardized mean difference
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
    ## README: based on the values given by Emily, this class works with log10 of energy!

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

    ## spectrum parameters
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
    

    def spectrum_func_simpl(self, energy): # this is the simplified version, to have less heavy calculation
        # energy in EeV

        J0 = 1 # this is just a constant in the end
        E12 = 5 # ankle
        E23 = 13
        E34 = 46 # suppression

        prod1 = np.power((1+np.power(energy/E12, 1/self.omega )), (self.g1-self.g2)*self.omega )
        prod2 = np.power((1+np.power(energy/E23, 1/self.omega )), (self.g2-self.g3)*self.omega )
        prod3 = np.power((1+np.power(energy/E34, 1/self.omega )), (self.g3-self.g4)*self.omega )

        func = J0*np.power(energy/np.sqrt(10), -self.g1)*prod1*prod2*prod3

        return func
    

    ## extract mass fractions (using now simplified spectrum)
    def spectrum_fraction(self, energy):

        spectrum = self.spectrum_func_simpl(energy)*(energy) # the multiplication for energy is necessary because data is already distributed as energy^-1
        ## accept-reject method
        labels = np.random.uniform(np.min(spectrum), np.max(spectrum), size=len(energy))
        mask = (labels<=spectrum)

        return  mask

    
class dipole_parameters:
    # this class is given by f(Z,E)*J(E)*d(Z,E), and together they should minimize distance with the dipole values

    def __init__(self, model='EPOS', dmax=3, e_steps=1000):
        # this is used to evaluate the fraction f(Z,E)
        self.mf = mass_fractions(model)
        self.dmax = dmax
        self.steps = e_steps
        print('Model chosen:', model)
        print('dipole cutoff:', self.dmax)
        print('Number of energy steps in each bin:', self.steps)

    # this is used to evaluate spectrum
    es = energy_spectrum()

    vec_d_r = np.arange(0.0001, 0.0201, step=0.0001)
    vec_b_r = np.arange(0.1, 3.1, step=0.1)

    def dipole_func(self, name, E, d_r, beta_r, d_max):
        # energy in EeV
        dip = d_r*np.power((E/(charges[name])), beta_r)
        cutoff = np.zeros_like(E) + d_max

        return np.min([dip, cutoff], axis=0)

    def evaluate(self):
        print('Best parameter evaluation ...')
        enebins = energies/1e18
        results = {'d_r': [], 'beta_r': [], 'chisq':[], '4':[], '8': [], '16': [], '32': []}
        counter = 0

        # maybe this cycle can be optimized, Idk
        for d_r in self.vec_d_r:
            counter += 1
            print('progress:', counter/2, '%', flush=True)

            for beta_r in self.vec_b_r:
                results['d_r'].append(d_r)
                results['beta_r'].append(beta_r)

                chisq = 0
                for i in range(len(enebins[:-1])):
                    E = np.linspace(enebins[i], enebins[i+1], self.steps)
                    Z_sum = np.zeros_like(E)

                    for name in charges:
                        if name == 'Fe':
                            Z_sum += self.dipole_func(name, E, d_r, beta_r, self.dmax)*(1 - (self.mf.fraction_func('p', np.log10(E)+18) + self.mf.fraction_func('He', np.log10(E)+18) + self.mf.fraction_func('CNO', np.log10(E)+18)))
                        else:
                            Z_sum += self.dipole_func(name, E, d_r, beta_r, self.dmax)*self.mf.fraction_func(name, np.log10(E)+18)
                    
                    dip = np.sum(Z_sum*self.es.spectrum_func_simpl(E))/np.sum(self.es.spectrum_func_simpl(E))

                    val = (dip - dipole_values[i])
                    if i>0: # Do not account the first bin!
                        if val>0:
                            chisq += (val/dipole_errors[1][i])**2
                        else:
                            chisq += (val/dipole_errors[0][i])**2

                    results[f'{enebins[i]:.0f}'].append(dip)

                
                
                results['chisq'].append(chisq)

        for key, value in results.items():
            results[key] = np.array(value)

        print('Best d_r:', results['d_r'][results['chisq']==np.min(results['chisq'])])
        print('Best beta_r:', results['beta_r'][results['chisq']==np.min(results['chisq'])])

        return results
