from enum import Enum
import torch
from torch_geometric.loader import DataLoader

class QM9Properties(Enum):
    MU = 0 # Dipole moment in Debye
    ALPHA = 1 # Isotropic polarizability in cubic Bohr radii
    HOMO = 2 # Highest occupied molecular orbital energy in eV
    LOMO = 3 # Lowest unoccupied molecular orbital energy in eV
    DELTA = 4 # Gap between HOMO and LUMO energies in eV
    R2 = 5 # Electronic spatial extent in square Bohr radii
    ZVPE = 6 # Zero-point vibrational energy in eV
    U_0 = 7 # Internal energy at 0K in eV
    U = 8 # Internal energy at 298.15K in eV
    H = 9 # Enthalpy at 298.15K in eV
    G = 10 # Free energy at 298.15K in eV
    C_V = 11 # Heat capacity at 298.15K in cal/mol K
    U_0_ATOM = 12 # Atomization energy at 0K in eV
    U_ATOM = 13 # Atomization energy at 298.15K in eV
    H_ATOM = 14 # Atomization enthalpy at 298.15K in eV
    G_ATOM = 15 # Atomization free energy at 298.15K in eV
    A = 16 # Rotational constant A in GHz
    B = 17 # Rotational constant B in GHz
    C = 18 # Rotational constant C in GHz




    
        

