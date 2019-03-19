import npmc.dihedral_ff_functions as dihedral_ff_functions
import numpy as np
from math import *
from functools import partial

class ForceField(object):
    """This class holds the key values of a Pair, Bond, Angle, or Dihedral Forcefield.

    Parameters
    ----------
    ff_function : function
        The forecefield function with input arguments being the ff_parameters as well as the radial distance for pair forcefields, bond distance for bond forcefields, bond angle for bond forcefields, and dihedral angle for dihedral forcefield.  The function should return the energy in kcal/mol.

    ff_parameters : list of type float
        A list of constants needed in the given forcefield function.
    """
    def __init__(self,ff_function,ff_parameters):
        self.ff_function = ff_function
        self.ff_parameters = ff_parameters



class DihedralForceField(ForceField):
    """This class is used to represent dihedral force fields.  Currently the only force field suported is OPLS, but it can be extended to others by adding to the dihedral_ff_functions module.

    Parameters
    ----------
    settings_filename : str
        A string containing the path to the file holding the settings file of the LAMMPS system.  This is the system.in.settings file created when using Moltemplate.

    dihedral_type : int
        The integer representing the dihedral type in the simulation.  This number is used to find the correct coeffs from the settings file.
    """
    def __init__(self,settings_filename,dihedral_type):
        self.dihedral_type = dihedral_type
        (ff_type,ff_params)=get_ff_params(settings_filename,dihedral_type)
        ff_function = get_ff_function(settings_filename,dihedral_type)
        super(DihedralForceField,self).__init__(ff_function,ff_params) 

    def get_pdf(self,temp):
        kb = 0.0019872041
        beta = 1./(kb*temp)
        (thetas,dtheta) = np.linspace(0,2*pi,num=500,retstep=True)
        energies = np.array([self.ff_function(theta) for theta in thetas])
        unnorm_probs = np.exp(-beta*energies)
        norm_probs = unnorm_probs/(sum(unnorm_probs)*dtheta)
        return((thetas,norm_probs))

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.dihedral_type==other.dihedral_type,self.ff_parameters==other.ff_parameters])
        else:
            return False


def get_ff_params(settings_filename,dihedral_type):
    dihedral_coeffs=read_ff_coeffs(settings_filename,coeff_style='dihedral')
    coeff = [coeff for coeff in dihedral_coeffs if int(coeff[1])==dihedral_type][0]
    params = [float(param) for param in coeff[3:]]
    ff_type = coeff[2]
    return (ff_type,params)

def get_ff_function(settings_filename,dihedral_type):
    (dih_ff_type,params)=get_ff_params(settings_filename,dihedral_type)
    dih_function = getattr(dihedral_ff_functions,dih_ff_type)
    dih_function_w_parameters = partial(dih_function,parameters=params)
    return dih_function_w_parameters

def read_ff_coeffs(settings_filename,coeff_style):
    with open(settings_filename,mode='r') as settings:
        coeffs = [str.split(coeff) for coeff in settings if coeff.strip()]
    return [coeff for coeff in coeffs if coeff[0]==(coeff_style+'_coeff')]


def initialize_dihedral_ffs(settings_filename):
    coeffs = read_ff_coeffs(settings_filename,coeff_style='dihedral')
    return([DihedralForceField(settings_filename,int(coeff[1])) for coeff in coeffs])
        
