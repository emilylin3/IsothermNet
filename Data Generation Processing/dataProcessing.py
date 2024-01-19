import os
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem import Draw, AllChem
from rdkit import Chem

import networkx as nx

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

"""
Data pre-processing and reshaping 

Input Features: textural properties, nodal info., bond info., angle, connectivity matrix
    N       maximum number of nodes (across all crystal structures)
    B       maximum number of bonds (across all crystal structures)
    A       maximum number of bond pairs (across all crystal structures)
    P_N     number of nodal properties/features
    P_B     number of bond properties/features
    P_A     number of angle properties/features
    P_T     number of textural properties
    S       number of crystal structures 

Outputs: isotherm fitting parameters, enthalpy (& errors) fitting parameters
    n_iso   number of isotherm fitting parameters
    n_H     number of enthalpy fitting parameters 
"""

def calcBondDist(x1, x2, y1, y2, z1, z2):
    dist = np.sqrt(((x2-x1)**2) + ((y2-y1)**2) + ((z2-z1)**2));
    return dist

def get_coordination_number(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return sum(1 for bond in atom.GetBonds())

def get_atom_properties(atom):
    electronegativity = AllChem.GetPeriodicTable().GetElementContrib(atom.GetAtomicNum(), "EN")
    return electronegativity

def mol_nx(mol):
    nxGraph = nx.Graph();
    for atom in mol.GetAtoms():
        nxGraph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), is_aromatic=atom.GetIsAromatic(), atom_symbol=atom.GetSymbol(), charge=atom.GetFormalCharge());
    for bond in mol.GetBonds():
        nxGraph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType(), in_ring=bond.IsInRing());
    return nxGraph

electronegativity_dict = {
    1: 2.20,   # Hydrogen
    2: 0.98,   # Helium
    3: 0.93,   # Lithium
    4: 1.31,   # Beryllium
    5: 1.61,   # Boron
    6: 2.01,   # Carbon
    7: 2.66,   # Nitrogen
    8: 2.02,   # Oxygen
    9: 2.05,   # Fluorine
    10: 0.82,  # Neon
    11: 0.93,  # Sodium
    12: 1.31,  # Magnesium
    13: 1.61,  # Aluminum
    14: 1.90,  # Silicon
    15: 2.19,  # Phosphorus
    16: 2.58,  # Sulfur
    17: 3.16,  # Chlorine
    18: 3.44,  # Argon
    19: 0.82,  # Potassium
    20: 1.00,  # Calcium
    21: 1.36,  # Scandium
    22: 1.54,  # Titanium
    23: 1.63,  # Vanadium
    24: 1.66,  # Chromium
    25: 1.55,  # Manganese
    26: 1.83,  # Iron
    27: 1.88,  # Cobalt
    28: 1.91,  # Nickel
    29: 1.90,  # Copper
    30: 1.65,  # Zinc
    31: 1.81,  # Gallium
    32: 2.01,  # Germanium
    33: 2.18,  # Arsenic
    34: 2.55,  # Selenium
    35: 3.04,  # Bromine
    36: 3.98,  # Krypton
    37: 0.79,  # Rubidium
    38: 0.89,  # Strontium
    39: 1.10,  # Yttrium
    40: 1.12,  # Zirconium
    41: 1.13,  # Niobium
    42: 1.17,  # Molybdenum
    43: 1.20,  # Technetium
    44: 1.20,  # Ruthenium
    45: 1.14,  # Rhodium
    46: 1.13,  # Palladium
    47: 1.17,  # Silver
    48: 1.20,  # Cadmium
    49: 1.22,  # Indium
    50: 1.23,  # Tin
    51: 1.24,  # Antimony
    52: 1.25,  # Tellurium
    53: 1.27,  # Iodine
    54: 1.30,  # Xenon
    55: 0.79,  # Cesium
    56: 0.89,  # Barium
    57: 1.10,  # Lanthanum
    58: 1.12,  # Cerium
    59: 1.13,  # Praseodymium
    60: 1.14,  # Neodymium
    61: 1.13,  # Promethium
    62: 1.17,  # Samarium
    63: 1.20,  # Europium
    64: 1.20,  # Gadolinium
    65: 1.20,  # Terbium
    66: 1.22,  # Dysprosium
    67: 1.23,  # Holmium
    68: 1.24,  # Erbium
    69: 1.25,  # Thulium
    70: 1.10,  # Ytterbium
    71: 1.27,  # Lutetium
    72: 1.10,  # Hafnium
    73: 1.12,  # Tantalum
    74: 1.13,  # Tungsten
    75: 1.14,  # Rhenium
    76: 1.13,  # Osmium
    77: 1.17,  # Iridium
    78: 1.20,  # Platinum
    79: 1.20,  # Gold
    80: 1.22,  # Mercury
    81: 1.23,  # Thallium
    82: 1.24,  # Lead
    83: 1.25,  # Bismuth
    84: 1.27,  # Polonium
    85: 2.04,  # Astatine0
    86: 2.55,  # Radon
    87: 0.70,  # Francium
    88: 0.89,  # Radium
    89: 1.10,  # Actinium
    90: 1.30,  # Thorium
    91: 1.50,  # Protactinium
    92: 1.38,  # Uranium
    93: 1.36,  # Neptunium
    94: 1.28,  # Plutonium
    95: 1.30,  # Americium
    96: 1.30,  # Curium
    97: 1.30,  # Berkelium
    98: 1.30,  # Californium
    99: 1.30,  # Einsteinium
    100: 1.30, # Fermium
}

# INPUTS    
dataDir = "C:/Users/opbir/CGCNN/"
texturalProp_data = pd.read_excel(dataDir + "texturalProperties.xlsx")  

LJ_params =  pd.read_csv("LJ_params.txt", sep="\s+", header=None)
LJ_element = LJ_params.iloc[:, 0]
LJ_eps_kb = LJ_params.iloc[:, 2]
LJ_sigma = LJ_params.iloc[:, 3]

uniqueElements = []
for s in texturalProp_data.iloc[:,0]:
    uniqueElements = uniqueElements + list(list((pd.read_csv(dataDir+"mol/"+str(s)+".mol", sep="\s+", skiprows=[0,1,2]).iloc[:, 0:4].dropna()).iloc[:, 3].to_numpy()))
uniqueElements = np.unique(uniqueElements)

print(uniqueElements)

uniqueSpacegroup = []
for i, s in enumerate(texturalProp_data.iloc[:,0]):
    structure = Structure.from_file(dataDir+"cif/"+str(s)+".cif")
    analyzer = SpacegroupAnalyzer(structure)
    uniqueSpacegroup.append(analyzer.get_crystal_system())
uniqueSpacegroup = list(np.unique(uniqueSpacegroup))

print(uniqueSpacegroup)

atomNum_metals = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]

# Max. number of nodes across all crystal structures
N = max([mol_nx(Chem.MolFromMolFile(dataDir+"mol/"+str(s)+".mol", sanitize=False)).number_of_nodes() for s in texturalProp_data.iloc[:,0]])  
# Max. number of bonds across all crystal structures
B = max([mol_nx(Chem.MolFromMolFile(dataDir+"mol/"+str(s)+".mol", sanitize=False)).number_of_edges() for s in texturalProp_data.iloc[:,0]])   
        
nodeFeat = []
IMFeat = []
strucGlobalFeat = []
bondFeat = []

for i, s in enumerate(texturalProp_data.iloc[:,0]):
    m = Chem.MolFromMolFile(dataDir+"mol/"+str(s)+".mol", sanitize=False)
    G = mol_nx(m)
    n = G.number_of_nodes()            
    
    structure = Structure.from_file(dataDir+"cif/"+str(s)+".cif")
    lattice_parameters = structure.lattice.abc
    lattice_parameters = torch.tensor([float(lattice_parameters[indx])for indx in range(len(lattice_parameters))])
    lattice_dim = lattice_parameters.repeat(n, 1)
    
    analyzer = SpacegroupAnalyzer(structure)
    crystal_system = analyzer.get_crystal_system()

    symmetry = torch.zeros(len(uniqueSpacegroup))
    symmetry[uniqueSpacegroup.index(crystal_system)] = 1
    symmetry = symmetry.repeat(n, 1)
    
    xyz = ((pd.read_csv(dataDir+"mol/"+str(s)+".mol", sep="\s+", skiprows=[0,1,2])).iloc[:, 0:4].dropna()).iloc[:, 0:3].to_numpy()
    xyz_tensor = torch.tensor([[float(xyz[k][indx])for indx in range(len(xyz[k]))] for k in range(xyz.shape[0])])

    elements = ((pd.read_csv(dataDir+"mol/"+str(s)+".mol", sep="\s+", skiprows=[0,1,2])).iloc[:, 0:4].dropna()).iloc[:, 3].to_numpy();
    
    # Node Features: Elements [one-hot encoding], shape: (n, len(uniqueElements))  |   Rings [one-hot encoding], shape: (n, )
    P = torch.Tensor(np.array([1e3,5e3,1e4,5e4,1e5,2e5,3e5,4e5,5e5,7e5,1e6,1.5e6,2e6,2.5e6,3e6,3.5e6,4e6,4.5e6,5e6])).unsqueeze(0).repeat(n, 1)
    P = P*0.00001

    nodeOHE = torch.zeros((n, len(uniqueElements)))
    nodeRingOHE = torch.zeros(n)

    for i in range(n):
        indx = list(uniqueElements).index(elements[i])
        nodeOHE[i, indx] = 1;
        
    nodePairs_tot = []
    ring = nx.get_edge_attributes(G, "in_ring")
    for i in range(len(ring.values())):
        if list(ring.values())[i] == True:
            nodePairs_tot = nodePairs_tot + str(list(ring.keys())[i]).replace("(","").replace(")","").replace(" ","").split(",")    
    indx = [int(x) for x in np.unique(nodePairs_tot)]
    nodeRingOHE[indx] = 1
    nodeRingOHE = nodeRingOHE.view(-1, 1);            # convert from 1D to 2D tensor
    
    coordination_num = []
    for atoms in m.GetAtoms():
        atom_idx = atoms.GetIdx()
        coord_num = get_coordination_number(m, atom_idx)
        coordination_num.append(coord_num)
    coordination_num = torch.Tensor(coordination_num)
        
    atom_num = torch.tensor(list(nx.get_node_attributes(G, "atomic_num").values()), dtype=torch.float32)   
    
    T = torch.ones(n) * 298
    sigma_CO2 = torch.ones(n) * 3.04
    eps_CO2 = torch.ones(n) * 125.3
    P_crit_CO2 = torch.ones(n) * 73.773
    
    is_metal = torch.zeros(n)
    electronegativity = torch.zeros(n)
    sigma = torch.zeros(n)
    eps = torch.zeros(n)
    for k, atomNum_val in enumerate(atom_num):
        if int(atomNum_val) in atomNum_metals:
            is_metal[k] = 1
        else:
            is_metal[k] = 0
            
        EN_index = [atomic_num for atomic_num, electronegativity in electronegativity_dict.items()].index(int(atomNum_val))
        electronegativity[k] = list(electronegativity_dict.values())[EN_index]
        
        atom = list(nx.get_node_attributes(G, "atom_symbol").values())[k]
        indx = list(LJ_element).index(atom)
        sigma[k] = LJ_sigma[indx]
        eps[k] = LJ_eps_kb[indx]
                    
    sigma = sigma.unsqueeze(-1)
    eps = eps.unsqueeze(-1)
    is_metal = is_metal.unsqueeze(-1)
    coordination_num = coordination_num.unsqueeze(-1)
    electronegativity = electronegativity.unsqueeze(-1)
    T = T.unsqueeze(-1)
    sigma_CO2 = sigma_CO2.unsqueeze(-1)
    eps_CO2 = eps_CO2.unsqueeze(-1)
    P_crit_CO2 = P_crit_CO2.unsqueeze(-1)
    
    distMat = torch.zeros((n, n));
    for i in range(n):
        xi = float(xyz[i][0]); yi = float(xyz[i][1]); zi = float(xyz[i][2]);  
        for j in range(n):
            xj = float(xyz[j][0]); yj = float(xyz[j][1]); zj = float(xyz[j][2]);  
            r = calcBondDist(xi,xj,yi,yj,zi,zj)
            if r == 0:
                distMat[i,j] = 1
            else:
                distMat[i,j] = r
                
    connectivity = Tensor(nx.to_numpy_array(G)) 

    r_inv = (1/distMat) * connectivity
            
    nodeFeat_cat = torch.cat((nodeOHE, nodeRingOHE, is_metal, coordination_num, electronegativity, xyz_tensor, P), dim=1)
    nodeFeat_pad = F.pad(input=nodeFeat_cat, pad=(0, 0, 0, N - n), mode="constant", value=0)    # Pad with zeros: shape (N, len(uniqueElements) + 1)
    nodeFeat.append(nodeFeat_pad)
        
    IMFeat_cat = torch.cat((sigma, eps, r_inv), dim=1)
    IMFeat_pad =  F.pad(input=IMFeat_cat, pad=(0, N - n, 0, N - n), mode="constant", value=0)    # Pad with zeros: shape (N, len(uniqueElements) + 1)
    IMFeat.append(IMFeat_pad)
    
    strucGlobalFeat_cat = torch.cat((lattice_dim, symmetry, T, sigma_CO2, eps_CO2, P_crit_CO2), dim=1)
    strucGlobalFeat_pad =  F.pad(input=strucGlobalFeat_cat, pad=(0, 0, 0, N - n), mode="constant", value=0)    # Pad with zeros: shape (N, len(uniqueElements) + 1)
    strucGlobalFeat.append(strucGlobalFeat_pad)
       
    # Bond Features: Bond index (begin), shape: (b, ) | Bond index (end), shape: (b, ) | Bond distance, shape: (b, )
    b = G.number_of_edges()        
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        
    bondDist = []
    U_r = []
    for k in range(edge_index.size(1)):
        i = edge_index[0, k]
        j = edge_index[1, k]
        
        atom_i = list(nx.get_node_attributes(G, "atom_symbol").values())[i]
        atom_j = list(nx.get_node_attributes(G, "atom_symbol").values())[j]
        
        indx_i = list(LJ_element).index(atom_i)
        indx_j = list(LJ_element).index(atom_j)
        
        sigma_ij = 0.5*(LJ_sigma[indx_i] + LJ_sigma[indx_j])
        eps_ij = np.sqrt(LJ_eps_kb[indx_i] * LJ_eps_kb[indx_j])
        
        xi = float(xyz[i][0]); yi = float(xyz[i][1]); zi = float(xyz[i][2]);  
        xj = float(xyz[j][0]); yj = float(xyz[j][1]); zj = float(xyz[j][2]);  
        
        r = calcBondDist(xi, xj, yi, yj, zi, zj)
        bondDist.append(r)
        
        U = 4*eps_ij*(((sigma_ij/r)**12) - ((sigma_ij/r)**6))          # Lennard-Jones potential        
        U_r.append(U)
                        
    bondFeat_cat = torch.cat((edge_index, torch.tensor(bondDist).view(1, -1)), dim=0)
    bondFeat_pad = F.pad(input=bondFeat_cat, pad=(0, (2*B)-(2*b), 0, 0), mode="constant", value=0)      # Pad with zeros: shape (B, 3)   
    bondFeat.append(bondFeat_pad)

    
nodeFeat = torch.stack(nodeFeat, dim=0)                        # shape: (S, N, P_N)
IMFeat = torch.stack(IMFeat, dim=0)                        # shape: (S, N, P_N)
strucGlobalFeat = torch.stack(strucGlobalFeat, dim=0)                        # shape: (S, N, P_N)
bondFeat = torch.stack(bondFeat, dim=0)                        # shape: (S, B, 3)

print(nodeFeat.shape)
print(IMFeat.shape)
print(strucGlobalFeat.shape)
print(bondFeat.shape)

# OUTPUTS
y_data = pd.read_excel(dataDir + "outputData.xlsx")                                      # shape: (S, n_iso+(3*n_H))
qm, K, n = y_data.iloc[:, 1], y_data.iloc[:, 2], y_data.iloc[:, 3]
y1_H, y2_H, y3_H = y_data.iloc[:, 4], y_data.iloc[:, 5], y_data.iloc[:, 6]
y1_H_LB, y2_H_LB, y3_H_LB = y_data.iloc[:, 7], y_data.iloc[:, 8], y_data.iloc[:, 9]
y1_H_UB, y2_H_UB, y3_H_UB = y_data.iloc[:, 10], y_data.iloc[:, 11], y_data.iloc[:, 12]

# Save all data
data_dict = {
    "nodeFeat": nodeFeat,
    "IMFeat": IMFeat,
    "strucGlobalFeat": strucGlobalFeat,
    "bondFeat": bondFeat 
}
torch.save(data_dict, dataDir+"X_dataset_electro_xyz_bond_struc.pth")

# Load and access all data
loadData = torch.load(dataDir+"X_dataset_electro_xyz_bond_struc.pth")

# Access the tensors and variables
nodeFeat = loadData["nodeFeat"]
IMFeat = loadData["IMFeat"]
strucGlobalFeat = loadData["strucGlobalFeat"]
bondFeat = loadData["bondFeat"]