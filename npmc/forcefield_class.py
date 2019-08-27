import npmc.ff_functions as ff_functions
import numpy as np
from scipy import integrate as si
from scipy import optimize as so
from math import *
from functools import partial
import time
import pdb
import pprint,pickle

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
    """This class is used to represent dihedral force fields.  Currently the only force field suported is OPLS, but it can be extended to others by adding to the ff_functions module.

    Parameters
    ----------
    settings_filename : str
        A string containing the path to the file holding the settings file of the LAMMPS system.  This is the system.in.settings file created when using Moltemplate.

    dihedral_type : int
        The integer representing the dihedral type in the simulation.  This number is used to find the correct coeffs from the settings file.
    """
    def __init__(self,settings_filename,dihedral_type):
        self.dihedral_type = dihedral_type
        (ff_type,ff_params)=get_ff_params(settings_filename,dihedral_type,ff_style='dihedral')
        self.ff_type = ff_type
        ff_function = get_ff_function(settings_filename,dihedral_type,ff_style='dihedral')
        super(DihedralForceField,self).__init__(ff_function,ff_params) 
    
    def get_pdf(self,temp):
        kb = 0.0019872041
        beta = 1./(kb*temp)
        (thetas,dtheta) = np.linspace(0,2*pi,num=500,retstep=True)
        energies = self.ff_function(thetas)
        unnorm_probs = np.exp(-beta*energies)
        norm_probs = unnorm_probs/(np.sum(unnorm_probs)*dtheta)
        return((thetas,norm_probs))

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.dihedral_type==other.dihedral_type,self.ff_parameters==other.ff_parameters])
        else:
            return False
            
class AngleForceField(ForceField):

    def __init__(self,settings_filename,angle_type):
        self.angle_type = angle_type
        (ff_type,ff_params)=get_ff_params(settings_filename,angle_type,ff_style='angle')
        ff_function = get_ff_function(settings_filename,angle_type,ff_style='angle')
        super(AngleForceField,self).__init__(ff_function,ff_params) 

    def get_pdf(self,temp):
        kb = 0.0019872041
        beta = 1./(kb*temp)
        (thetas,dtheta) = np.linspace(0,2*pi,num=500,retstep=True)
        energies = self.ff_function(thetas)
        unnorm_probs = np.exp(-beta*energies)
        norm_probs = unnorm_probs/(np.sum(unnorm_probs)*dtheta)
        return((thetas,norm_probs))

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.angle_type==other.angle_type,self.ff_parameters==other.ff_parameters])
        else:
            return False

class BranchPDF():

    def __init__(self,dihFF1,dihFF2,angleFF,bond_angles,temp,read=False,write=False):
        self.dihFF1 = dihFF1; self.dihFF2 = dihFF2; self.angleFF = angleFF
        self.bond_angles = bond_angles
        kb = 0.0019872041
        self.beta = 1./(kb*temp)
        self.Q = si.dblquad(self.unnorm_prob, 0,2*pi, 0,2*pi)[0]
        self.weighted = True      
        if read: self.pdf,self.weights = self.read_pdf(self.weighted)
        elif self.weighted: self.pdf,self.weights = self.tabulate_pdf_weighted(250,write)
        else: self.pdf = self.tabulate_pdf(100,100,write)
          
    def tabulate_pdf_weighted(self,intervals,write):
        phis = np.linspace(0,2*pi,intervals+1)
        pdf = np.empty([intervals**2,4]); weights = np.empty(intervals**2)
        for i in range(intervals):
            for j in range(intervals): 
                pdf[i*intervals+j,0] = phis[i]; pdf[i*intervals+j,1] = phis[i+1]
                pdf[i*intervals+j,2] = phis[j]; pdf[i*intervals+j,3] = phis[j+1]
                weights[i*intervals+j] = si.dblquad(self.unnorm_prob,pdf[i*intervals+j,0],pdf[i*intervals+j,1],lambda x:pdf[i*intervals+j,2],lambda x:pdf[i*intervals+j,3])[0]
        weights = weights/np.sum(weights)
        if write: self.write_pdf(pdf,True,weights)
        return pdf,weights
  
    def tabulate_pdf(self,intervals1,intervals2,write):
        pdf_1D = self.get_pdf_1D(intervals1)
        pdf = np.zeros([intervals1,intervals2,4]); pdf[:,-1,3] = 2*pi; probs = np.empty([intervals1,intervals2])
        for i in range(intervals1):
            pdf[i,:,0] = pdf_1D[i]; pdf[i,:,1] = pdf_1D[i+1]
            for j in range(intervals2-1):
                upper_bracket = self.get_upper_bracket(i,j,pdf,1/(intervals1*intervals2)*1.5)
                boundary=so.brenth(lambda phi:si.dblquad(self.unnorm_prob,pdf[i,j,0],pdf[i,j,1],lambda x:pdf[i,j,2],phi)[0]/self.Q-(1/intervals1)*(1/intervals2),lambda x:pdf[i,j,2],upper_bracket)
                pdf[i,j,3] = boundary; pdf[i,j+1,2] = boundary
                probs[i,j] = si.dblquad(self.unnorm_prob,pdf[i,j,0],pdf[i,j,1],lambda x:pdf[i,j,2],lambda x:pdf[i,j,3])[0]/self.Q                
            probs[i,-1] = si.dblquad(self.unnorm_prob,pdf[i,-1,0],pdf[i,-1,1],lambda x:pdf[i,-1,2],lambda x:pdf[i,-1,3])[0]/self.Q
        if write: self.write_pdf(pdf,False,None)
        return pdf
    
    def get_pdf_1D(self,intervals):
        pdf_1D = np.zeros(intervals+1); pdf_1D[-1] = 2*pi
        phi2s,dphi2 = np.linspace(0,2*pi,1000,retstep=True)
        for i in range(intervals-1): 
            pdf_1D[i+1]=so.brenth(lambda phi:si.quad(self.norm_flatten_probs,pdf_1D[i],phi,args=(phi2s,dphi2))[0]-1/intervals,pdf_1D[i],2*pi)    
        return pdf_1D  
    
    def norm_flatten_probs(self,phi1,phi2,dphi):
        flatten_probs = np.sum(self.unnorm_prob(phi1,phi2))*dphi/self.Q
        return flatten_probs
    
    def unnorm_prob(self,phi1,phi2):
        theta = central_angle_Vincenty(phi1,phi2,self.bond_angles[0],self.bond_angles[1])
        unnorm_probs = np.exp(-self.beta*(self.dihFF1.ff_function(phi1)+self.dihFF2.ff_function(phi2)+self.angleFF.ff_function(theta)))
        return unnorm_probs
       
    def get_upper_bracket(self,i,j,pdf,threshold):
        upper_bracket = pdf[i,j,2]; total_prob = 0; slice = 2*pi/1000
        phi1s,dphi1 = np.linspace(pdf[i,j,0],pdf[i,j,1],100,retstep=True)
        while total_prob < threshold:
            total_prob += self.norm_flatten_probs(phi1s,upper_bracket,dphi1)*slice
            upper_bracket += slice
        return upper_bracket
        
    def read_pdf(self,weighted):
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
        if weighted: type = 'weighted'
        else: type = 'unweighted'
        pdf_file = open(f'pdf_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}_{type}.pkl', 'wb')
        weights_file = open(f'weights_{self.dihFF1.dihedral_type}_{self.dihFF2.dihedral_type}_{self.angleFF.angle_type}.pkl', 'wb')
        pickle.dump(pdf,pdf_file)
        if weighted: pickle.dump(weights,weights_file)
        pdf_file.close(); weights_file.close()
   
def get_ff_params(settings_filename,ff_type,ff_style):
    ff_coeffs=read_ff_coeffs(settings_filename,ff_style)
    coeff = [coeff for coeff in ff_coeffs if int(coeff[1])==ff_type][0]
    params = [float(param) for param in coeff[3:]]
    ff_type = coeff[2]
    return (ff_type,params)

def get_ff_function(settings_filename,forcefield_type,ff_style):
    (ff_type,params)=get_ff_params(settings_filename,forcefield_type,ff_style)
    ff_function = getattr(ff_functions,ff_type)
    ff_function_w_parameters = partial(ff_function,parameters=params)
    return ff_function_w_parameters

def read_ff_coeffs(settings_filename,ff_style):
    with open(settings_filename,mode='r') as settings:
        coeffs = [str.split(coeff) for coeff in settings if coeff.strip()]
    return [coeff for coeff in coeffs if coeff[0]==(ff_style+'_coeff')]

def initialize_dihedral_ffs(settings_filename):
    coeffs = read_ff_coeffs(settings_filename,ff_style='dihedral')
    return([DihedralForceField(settings_filename,int(coeff[1])) for coeff in coeffs])

def initialize_angle_ffs(settings_filename):
    coeffs = read_ff_coeffs(settings_filename,ff_style='angle')
    return([AngleForceField(settings_filename,int(coeff[1])) for coeff in coeffs])
    
def initialize_branch_pdfs(molecules,dihedral_ffs,angle_ffs,T,read=False,write=False):
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
    lambda1 = lambda1-np.pi/2; lambda2 = lambda2-np.pi/2
    dphi = np.absolute(phi1-phi2)
    numerator = ((np.cos(lambda2)*np.sin(dphi))**2 + (np.cos(lambda1)*np.sin(lambda2) - np.sin(lambda1)*np.cos(lambda2)*np.cos(dphi))**2)**0.5
    denominator = np.sin(lambda1)*np.sin(lambda2) + np.cos(lambda1)*np.cos(lambda2)*np.cos(dphi)
    theta = np.arctan(numerator/denominator)
    return np.where(theta<0,pi+theta,theta)   
