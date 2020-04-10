import npmc.ff_functions as ff_functions
import numpy as np
from scipy import integrate as si
from scipy import optimize as so
from math import *
from functools import partial
import time
import pdb
import matplotlib.pyplot as plt
import pprint,pickle

class ForceField(object):
    """This class holds the key values of a Pair, Bond, Angle, or Dihedral Forcefield.

    Parameters
    ----------
    ff_function : function
        The forecefield function with input arguments being the ff_parameters as well as the radial distance for pair forcefields, bond distance for bond forcefields, 
        bond angle for bond forcefields, and dihedral angle for dihedral forcefield.  The function should return the energy in kcal/mol.
    ff_parameters : list of type float
        A list of constants needed in the given forcefield function.
    """
    def __init__(self,ff_function,ff_parameters):
        self.ff_function = ff_function
        self.ff_parameters = ff_parameters

class DihedralForceField(ForceField):
    """This class is used to represent dihedral force fields.  Currently the only force fields supported are OPLS and TraPPE-UA, 
    but it can be extended to others by adding to the ff_functions module.

    Parameters
    ----------
    settings_filename : str
        A string containing the path to the file holding the settings file of the LAMMPS system.  This is the system.in.settings file created when using Moltemplate.
    dihedral_type : int
        The integer representing the dihedral type in the simulation.  This number is used to find the correct coeffs from the settings file.
    """
    def __init__(self,settings_filename,dihedral_type):
        self.dihedral_type = dihedral_type
        ff_type,ff_params = get_ff_params(settings_filename,dihedral_type,ff_style='dihedral')
        self.ff_type = ff_type
        ff_function = get_ff_function(settings_filename,dihedral_type,ff_style='dihedral')
        super(DihedralForceField,self).__init__(ff_function,ff_params) 
        
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.dihedral_type==other.dihedral_type,self.ff_parameters==other.ff_parameters])
        else:
            return False
    
    def get_pdf(self,temp):
        """Calculates the probability distribution function for dihedral angles of the given type, based on energies determined from the associated force field function and force field
        parameters.
        
        Parameters
        ----------
        temp : float
            The temperature at which to calculated the PDF. 
            
        Returns
        -------
        thetas : Numpy array of floats
            1x500 array of the dihedral angles in radians which corresponed to the returned probabilities.   
        norm_probs : Numpy array of floats
            1x500 array of the normalized probabilities of the returned dihedral angles.
        """
        kb = 0.0019872041
        beta = 1./(kb*temp)
        thetas,dtheta = np.linspace(0,2*pi,num=500,retstep=True)
        energies = self.ff_function(thetas)
        unnorm_probs = np.exp(-beta*energies)
        norm_probs = unnorm_probs/(np.sum(unnorm_probs)*dtheta)
        return thetas,norm_probs

            
class AngleForceField(ForceField):
    """This class is used to represent bond angle force fields.  Currently the only force field supported is harmonic, 
    but it can be extended to others by adding to the ff_functions module.

    Parameters
    ----------
    settings_filename : str
        A string containing the path to the file holding the settings file of the LAMMPS system.  This is the system.in.settings file created when using Moltemplate.
    angle_type : int
        The integer representing the bond angle type in the simulation.  This number is used to find the correct coeffs from the settings file.
    """
    def __init__(self,settings_filename,angle_type):
        self.angle_type = angle_type
        ff_type,ff_params = get_ff_params(settings_filename,angle_type,ff_style='angle')
        ff_function = get_ff_function(settings_filename,angle_type,ff_style='angle')
        super(AngleForceField,self).__init__(ff_function,ff_params) 

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.angle_type==other.angle_type,self.ff_parameters==other.ff_parameters])
        else:
            return False
            
    def get_pdf(self,temp):
        """Calculates the probability distribution function for bond angles of the given type, based on energies determined from the associated force field function and force field
        parameters.
        
        Parameters
        ----------
        temp : float
            The temperature at which to calculated the PDF. 
            
        Returns
        -------
        thetas : Numpy array of floats
            1x500 array of the bond angles in radians which corresponed to the returned probabilities.   
        norm_probs : Numpy array of floats
            1x500 array of the normalized probabilities of the returned bond angles.
        """
        kb = 0.0019872041
        beta = 1./(kb*temp)
        (thetas,dtheta) = np.linspace(0,2*pi,num=500,retstep=True)
        energies = self.ff_function(thetas)
        unnorm_probs = np.exp(-beta*energies)
        norm_probs = unnorm_probs/(np.sum(unnorm_probs)*dtheta)
        return thetas,norm_probs


class BranchPDF():
    """This class is used to handle the PDF of energy at branch points. It currently handles only single split branches, using two dihedral force fields and one bond angle force field to generate the PDF.
    
    Parameters
    ----------
    dihFF1,dihFF2 : DihedralForceField
        DihedralForceField objects corresponding to the two dihedral angles at the branch point, where the third atom of both dihedrals is the atom to which both branches connect.       
    angleFF : AngleForceField
        The AngleForceField object corresponding to the bond angle between the third atom of both dihedrals and the fourth atoms of both dihedrals.   
    bond_angles : Numpy array of floats
        A 1x2 array of the bond angles between the second, third, and fourth atoms of each dihedral.   
    temp : float
        The temperature at which to calculated the PDF.       
    read : Boolean
        A Boolean that determines whether the PDF is pickled and written to a .pkl file.   
    write : Boolean
        A Boolean that determines whether the PDF is read from an already existing .pkl file.
    """
    def __init__(self,dihFF1,dihFF2,angleFF,bond_angles,temp,read=False):
        self.dihFF1 = dihFF1; self.dihFF2 = dihFF2; self.angleFF = angleFF
        self.bond_angles = bond_angles
        kb = 0.0019872041
        self.beta = 1./(kb*temp)
        self.Q = si.dblquad(self.unnorm_prob, 0,2*pi, 0,2*pi)[0]
        self.weighted = True
        if read: self.pdf,self.weights = self.read_pdf(self.weighted)
        elif self.weighted: self.pdf,self.weights = self.tabulate_pdf_weighted(250)
        else: self.pdf = self.tabulate_pdf(100,100)
          
    def tabulate_pdf_weighted(self,intervals):
        """Generate equally spaced intervals for the two dihedral angles along with corresponding probabilities for each interval based on the potential energy of the branch point.
            
        Parameters
        ----------
        intervals : int
            The number of intervals into which to divide dihedral angle phase space.     
        write : Boolean
            A Boolean that determines whether the PDF is read from an already existing .pkl file.
            
        Returns
        -------
        pdf : Numpy array of floats
            An "intervals x intervals" array of dihedral angles which correspond to the limits of the intervals in the PDF.          
        weights : Numpy array of floats 
            An "intervals x intervals" array of the probabilities of each interval in the PDF.
        """
        phis = np.linspace(0,2*pi,intervals+1)
        pdf = np.empty([intervals**2,4]); weights = np.empty(intervals**2)
        for i in range(intervals):
            for j in range(intervals): 
                pdf[i*intervals+j,0] = phis[i]; pdf[i*intervals+j,1] = phis[i+1]
                pdf[i*intervals+j,2] = phis[j]; pdf[i*intervals+j,3] = phis[j+1]
                weights[i*intervals+j] = si.dblquad(self.unnorm_prob,pdf[i*intervals+j,2],pdf[i*intervals+j,3],lambda x:pdf[i*intervals+j,0],lambda x:pdf[i*intervals+j,1])[0]
        weights = weights/np.sum(weights)
        self.write_pdf(pdf,True,weights)
        return pdf,weights
  
    def tabulate_pdf(self,intervals1,intervals2):
        """Generate equal probability intervals for the two dihedral angles along with corresponding probabilities for each interval based on the potential energy of the branch point.
            
        Parameters
        ----------
        intervals1,intervals2 : int
            The number of intervals into which to divide each dihedral angle phase space.       
        write : Boolean
            A Boolean that determines whether the PDF is read from an already existing .pkl file.
            
        Returns
        -------
        pdf : Numpy array of floats
            An "intervals x intervals" array of dihedral angles which correspond to the limits of the intervals in the PDF. Each interval contians equal probability.
        """
        pdf_1D = self.get_pdf_1D(intervals1)
        pdf = np.zeros([intervals1,intervals2,4]); pdf[:,-1,3] = 2*pi; probs = np.empty([intervals1,intervals2])
        for i in range(intervals1):
            pdf[i,:,0] = pdf_1D[i]; pdf[i,:,1] = pdf_1D[i+1]
            for j in range(intervals2-1):
                upper_bracket = self.get_upper_bracket(i,j,pdf,1/(intervals1*intervals2)*1.5)
                boundary=so.brenth(lambda phi:si.dblquad(self.unnorm_prob,pdf[i,j,2],phi,pdf[i,j,0],pdf[i,j,1])[0]/self.Q-(1/intervals1)*(1/intervals2),pdf[i,j,2],upper_bracket)
                pdf[i,j,3] = boundary; pdf[i,j+1,2] = boundary
                probs[i,j] = si.dblquad(self.unnorm_prob,pdf[i,j,2],pdf[i,j,3],lambda x:pdf[i,j,0],lambda x:pdf[i,j,1])[0]/self.Q                
            probs[i,-1] = si.dblquad(self.unnorm_prob,pdf[i,-1,2],pdf[i,-1,3],lambda x:pdf[i,-1,0],lambda x:pdf[i,-1,1])[0]/self.Q
        self.write_pdf(pdf,False,None)
        return pdf
    
    def get_probs_1D(self,phi1s):
        """Get the probabilities for the full phase space of one dihedral angle, where the full phase space of the other dihedral angle is integrated.
        
        Parameters
        ----------
        phi1s : Numpy array of floats
            An 1xN array of the dihedral angles which specify the limits of the intervals into which the phase space is divided.
            
        Returns
        -------
        probs_1D : Numpy array of floats
            An 1xN-1 array of the probability corresponding to each phase space interval.
        """
        probs_1D = np.zeros(len(phi1s))
        phi2s,dphi2 = np.linspace(0,2*pi,1000,retstep=True)
        for i,phi1 in enumerate(phi1s): 
            probs_1D[i] = self.norm_flatten_probs(phi2s,phi1,dphi2)
        return probs_1D  
    
    def norm_flatten_probs(self,phi1,phi2,dphi):
        """Get the normalized probabilities for the phase space contained in phi1 of one dihedral angle, where the phase space contained in phi2 of the other dihedral angle is integrated step-wise by dphi.
        
        Parameters
        ----------
        phi1 : Numpy array of floats
            1xN array of dihedral angles for the dihedral angle of interest.         
        phi2 : Numpy array of floats
            1xM array of dihedral angles for the secondary dihedral angle to be "flattened".           
        dphi : float
            Width of the step for the step-wise integration.
            
        Returns
        -------
        flatten_probs : Numpy array of floats
            1xN array of normalized probabilities for the dihedral angle phase space given by phi1.
        """
        flatten_probs = np.sum(self.unnorm_prob(phi1,phi2))*dphi/self.Q
        return flatten_probs
    
    def unnorm_prob(self,phi1,phi2):
        """Get the unnormalized probabilities for the phase space contained in phi1 and phi2.
        
        Parameters
        ----------
        phi1 : Numpy array of floats
            1xN array of dihedral angles for the first dihedral angle.          
        phi2 : Numpy array of floats
            1xM array of dihedral angles for the second dihedral angle.
            
        Returns
        -------
        unnorm_probs : Numpy array of floats
            1xN array of unnormalized probabilities for the dihedral angle phase space given by phi1 and phi2.
        """
        theta = central_angle_Vincenty(phi1,phi2,self.bond_angles[0],self.bond_angles[1])
        unnorm_probs = np.exp(-self.beta*(self.dihFF1.ff_function(phi1)+self.dihFF2.ff_function(phi2)+self.angleFF.ff_function(theta)))
        return unnorm_probs
       
    def get_upper_bracket(self,i,j,pdf,threshold):
        """Acquire an upper limit for the brenth integration method.
        
        Parameters
        ----------
        i,j : int
            indices of the PDF array for the interval for which the upper limit is desired.      
        pdf : Numpy array of floats
            An "intervals x intervals" array of dihedral angles which correspond to the limits of the intervals in the PDF.           
        threshold : float
            The expected probability contained in equal probability intervals.
            
        Returns
        -------
        upper_bracket : float
            The upper limit for the step of the brenth integration method given by i and j.
        """
        upper_bracket = pdf[i,j,2]; total_prob = 0; slice = 2*pi/1000
        phi1s,dphi1 = np.linspace(pdf[i,j,0],pdf[i,j,1],100,retstep=True)
        while total_prob < threshold:
            total_prob += self.norm_flatten_probs(phi1s,upper_bracket,dphi1)*slice
            upper_bracket += slice
        return upper_bracket
        
    def read_pdf(self,weighted):
        """Read the PDF from the correct .pkl file.
        
        Parameters
        ----------
        weighted : Boolean
            A Boolean which indicated whether weights need to be read along with the PDF.
        
        Returns
        -------
        pdf : Numpy array of floats
            An "intervals x intervals" array of dihedral angles which correspond to the limits of the intervals in the PDF.            
        weights : Numpy array of floats 
            An "intervals x intervals" array of the probabilities of each interval in the PDF; returns None if the PDF is unweighted.
        """
        if weighted: type = 'weighted'
        else: type = 'unweighted'
        pdf_file = open(f'pdf_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}_{type}.pkl', 'rb')
        weights_file = open(f'weights_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}.pkl', 'rb')
        pdf = pickle.load(pdf_file)
        if weighted: weights = pickle.load(weights_file)
        else: weights = None
        pdf_file.close(); weights_file.close()
        return pdf,weights
    
    def write_pdf(self,pdf,weighted,weights):
        """Write the PDF to a .pkl file.
        
        Parameters
        ----------
        weighted : Boolean
            A Boolean which indicated whether weights need to be read along with the PDF.       
        pdf : Numpy array of floats
            An "intervals x intervals" array of dihedral angles which correspond to the limits of the intervals in the PDF.           
        weights : Numpy array of floats 
            An "intervals x intervals" array of the probabilities of each interval in the PDF; this can be None if weighted is False.       
        """
        if weighted: type = 'weighted'
        else: type = 'unweighted'
        pdf_file = open(f'pdf_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}_{type}.pkl', 'wb')
        weights_file = open(f'weights_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}.pkl', 'wb')
        pickle.dump(pdf,pdf_file)
        if weighted: pickle.dump(weights,weights_file)
        pdf_file.close(); weights_file.close()
   
def get_ff_params(settings_filename,ff_type,ff_style):
    """Returns the forcefield parameters for ff_type and ff_style from settings_filename.
    
    Parameters
    ----------
    settings_filename : str
        The name of the file containing the forcefield parameters (e.g. system.in.settings)
    ff_type : int
        An integer indicating which coefficients to return; it corresponds to a specific forcefield.
    ff_style : str
        The name of the style of forcefield; 'angle' for bond angles and 'dihedral' for dihedral angles are currently supported.
        
    Returns
    -------
    ff_method : str
        The forcefield method used from 'ff_functions.py'; 'harmonic', 'opls', and 'fourier' are currently supported.
    params : list of floats
        The forcefield parameters for the specific type, style, and method.
    """
    ff_coeffs=read_ff_coeffs(settings_filename,ff_style)
    coeff = [coeff for coeff in ff_coeffs if int(coeff[1])==ff_type][0]
    params = [float(param) for param in coeff[3:]]
    ff_method = coeff[2]
    return ff_method,params

def get_ff_function(settings_filename,forcefield_type,ff_style):
    """Constructs a partial forcefield function using the appropriate form from 'ff_functions.py' and the parameters from settings_filename.
    
    Parameters
    ----------
    settings_filename : str
        The name of the file containing the forcefield parameters (e.g. system.in.settings)
    ff_type : int
        An integer indicating which coefficients to return; it corresponds to a specific forcefield.
    ff_style : str
        The name of the style of forcefield; 'angle' for bond angles and 'dihedral' for dihedral angles are currently supported.
        
    Returns
    -------
    ff_function_w_parameters : partial object
        The forcefield function from 'ff_functions.py' with the forcefield parameters substituted in.
    """
    (ff_type,params)=get_ff_params(settings_filename,forcefield_type,ff_style)
    ff_function = getattr(ff_functions,ff_type)
    ff_function_w_parameters = partial(ff_function,parameters=params)
    return ff_function_w_parameters

def read_ff_coeffs(settings_filename,ff_style):
    """Reads all the forcefield parameters for ff_style from settings_filename.
    
    Parameters
    ----------
    settings_filename : str
        The name of the file containing the forcefield parameters (e.g. system.in.settings)
    ff_style : str
        The name of the style of forcefield; 'angle' for bond angles and 'dihedral' for dihedral angles are currently supported.
        
    Returns
    -------
    params : list of list of floats
        All the forcefield parameters for ff_style from settings_filename.
    """
    with open(settings_filename,mode='r') as settings:
        coeffs = [str.split(coeff) for coeff in settings if coeff.strip()]
    params = [coeff for coeff in coeffs if coeff[0]==(ff_style+'_coeff')]
    return params

def initialize_dihedral_ffs(settings_filename):
    """Initiliaze the DihedralForceFields found in settings_filename.
    
    Parameters
    ----------
    settings_filename : str
        The name of the file containing the forcefield parameters (e.g. system.in.settings)
        
    Returns
    -------
    dihedral_ffs : list of DihedralForceFields
        A list of the DihedralForceField objects for the dihedral force fields found in settings_filename.
    """
    coeffs = read_ff_coeffs(settings_filename,ff_style='dihedral')
    dihedral_ffs = [DihedralForceField(settings_filename,int(coeff[1])) for coeff in coeffs]
    return dihedral_ffs

def initialize_angle_ffs(settings_filename):
    """Initiliaze the AngleForceFields found in settings_filename.
    
    Parameters
    ----------
    settings_filename : str
        The name of the file containing the forcefield parameters (e.g. system.in.settings)
        
    Returns
    -------
    angle_ffs : list of AngleForceFields
        A list of the AngleForceField objects for the bond angle force fields found in settings_filename.
    """
    coeffs = read_ff_coeffs(settings_filename,ff_style='angle')
    angle_ffs = [AngleForceField(settings_filename,int(coeff[1])) for coeff in coeffs]
    return angle_ffs
    
def initialize_branch_pdfs(molecules,dihedral_ffs,angle_ffs,T,read=False,write=False):
    """Initiliaze the PDF's for all the branch points in the ligands.
    
    Parameters
    ----------
    molecules : list of Molecules
        A list of the Molecule objects in the simulation.
    dihedral_ffs : list of DihedralForceFields
        A list of the dihedral force fields, in the form of DihedralForceField objects, that are relevant for the simulation.
    angle_ffs : list of AngleForceFields
        A list of the bond angle force fields, in the form of AngleForceField objects, that are relevant for the simulation.
    read : Boolean
        A Boolean that determines whether the PDF is pickled and written to a .pkl file.   
    write : Boolean
        A Boolean that determines whether the PDF is read from an already existing .pkl file.    
        
    Returns
    -------
    branchPDFs : list of BranchPDFs
        A list of BranchPDFs corresponding to the branch points in all the ligands.
    """
    branch_points = [branch_point for molecule in molecules for branch_point in molecule.findBranchPoints()]
    branchPDFs = []; known_types = []
    for branch_point in branch_points:
        types = branch_point[0]
        if types in known_types: continue
        dihedral_ff1 = [dihedral_ff for dihedral_ff in dihedral_ffs if dihedral_ff.dihedral_type==types[0]][0]
        dihedral_ff2 = [dihedral_ff for dihedral_ff in dihedral_ffs if dihedral_ff.dihedral_type==types[1]][0]           
        angle_ff = [angle_ff for angle_ff in angle_ffs if angle_ff.angle_type==types[2]][0]
        branchPDFs.append(BranchPDF(dihedral_ff1,dihedral_ff2,angle_ff,branch_point[1],T,read=read,write=write))
        known_types.append(types)
    return branchPDFs
    
def central_angle_Vincenty(phi1,phi2,lambda1,lambda2):
    """Calculates the central angle, corresponding to the bond angle at a branch point, using the Vincenty formula, the two dihedral angle at the branch point,
    and the two bond angles between the second, third, and fourth atoms of the dihedral angles.
    
    Parameters
    ----------
    phi1,phi2 : float
        The dihedral angles.
    lambda1,lambda2 : float
        The bond angles corresponding to the dihedral angles, respectively.
        
    Returns
    -------
    central_angle : float
        The bond angle at the branch point which includes phi1 and phi2 as dihedral angles.
    """
    lambda1 = lambda1-np.pi/2; lambda2 = lambda2-np.pi/2
    dphi = np.absolute(phi1-phi2)
    numerator = ((np.cos(lambda2)*np.sin(dphi))**2 + (np.cos(lambda1)*np.sin(lambda2) - np.sin(lambda1)*np.cos(lambda2)*np.cos(dphi))**2)**0.5
    denominator = np.sin(lambda1)*np.sin(lambda2) + np.cos(lambda1)*np.cos(lambda2)*np.cos(dphi)
    theta = np.arctan(numerator/denominator)
    central_angle = np.where(theta<0,pi+theta,theta) 
    return central_angle
