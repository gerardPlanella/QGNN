from abc import ABC, abstractmethod
import random
import platform
from enum import Enum
import warnings
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch

import quimb.tensor as qtn
import quimb as qu

import tenpy
from tenpy.networks.site import SpinHalfSite
from tenpy.models.tf_ising import TFIChain
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

from torch_geometric.data import Data


class NonParametrizedAlgorithm():
    def parameters(self):
        return 0

class TensorNetworkAlgorithm(ABC):
    @abstractmethod
    def __init__(self, params):
        pass
    @abstractmethod
    def set_datapoint(self, data_point):
        pass
    @abstractmethod
    def create_model(self):
        pass
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def getEnergy(self):
        pass
    @abstractmethod
    def getRDMs(self):
        pass
    @abstractmethod
    def getGroundState(self):
        pass
    @staticmethod
    def isMPS(data_point):
        return len(data_point.grid_extent) == 1 or data_point.grid_extent[1] == 1

class SimpleUpdate(TensorNetworkAlgorithm, NonParametrizedAlgorithm):
    def __init__(self, params):
        self.params = params
        self.chi = params['chi']
        self.bond_dim = params['bond_dim']
        self.num_iters = params['num_iters']
        self.tau = params['tau']
        self.psi = None
        self.hamiltonian = None
        self.psi = None
        self.energy = None
        self.data_point = None
        self.psi0 = None
        self.tn_type = "PEPS"

    @staticmethod
    def pos(node, n):
        # calculate x,y coordinates from node index
        y = node % n[1]
        x = (node - y) // n[1]
        return x,y

    def set_datapoint(self, data_point):
        self.n = tuple(data_point.grid_extent)
        self.data_point = data_point
        if TensorNetworkAlgorithm.isMPS(data_point):
            raise ValueError("SimpleUpdate does not support MPS")
        self.psi0 = qtn.PEPS.rand(*self.n, bond_dim=self.bond_dim)
    
    def create_model(self):

        local_one_site_hamiltonians = {}  # dict for qtn.LocalHam2D H1
        for i, (h,g) in enumerate(self.data_point.x_nodes):
            h, g = float(h), float(g)  # convert from torch to float
            local_one_site_hamiltonians[SimpleUpdate.pos(i, self.n)] = qu.spin_operator('Z') * h * 2
            local_one_site_hamiltonians[SimpleUpdate.pos(i, self.n)] += qu.spin_operator('X') * g * 2

        local_two_site_hamiltonians = {}  # dict for qtn.LocalHam2D H2
        for (a,b), J_ab in zip(self.data_point.edge_index.T, self.data_point.x_edges):
            a, b, J_ab = int(a), int(b), float(J_ab)  # convert from torch to int/float
            if J_ab != 0:
                local_two_site_hamiltonians[SimpleUpdate.pos(a, self.n), SimpleUpdate.pos(b, self.n)] = qu.ham_heis(2, j=(0, 0, 4*J_ab))  # factor of 4 because of different convention

        ham_local = qtn.LocalHam2D(*self.n, H2=local_two_site_hamiltonians, H1=local_one_site_hamiltonians)
        self.hamiltonian = ham_local
    
    def run(self):
        su = qtn.SimpleUpdate(
            psi0 = self.psi0,
            ham = self.hamiltonian,
            chi = self.chi,
            compute_energy_every = None,
            compute_energy_per_site = True,
            keep_best = True,
            progbar = False
        )
        for tau in self.tau:
            su.evolve(self.num_iters, tau=tau)
        self.psi = su.best['state']
        self.energy = su.best['energy'] * np.prod(self.n)
    
    def getEnergy(self):
        return self.energy
    
    def getRDMs(self):
        one_rdms = []
        two_rdms = []

        m, n = tuple(self.data_point.grid_extent)
        dims = [[2] * n] * m
            

        def compute_rdm(peps, sites, dims):
            """
            Compute the RDM for a list of sites in a PEPS.

            Parameters:
            peps (PEPS): The PEPS representing the quantum state.
            sites (List of tuples): The coordinates of the sites.
            dims (list): The dimensions of the Hilbert space at each site.

            Returns:
            numpy.ndarray: The 2-RDM of the specified sites.
            """          
            return qu.normalize(qu.partial_trace(peps, dims=dims, keep=sites))

        ground_state = self.psi.to_dense()

        for node_idx in range(self.data_point.x_nodes.shape[0]):
            one_rdms.append(compute_rdm(ground_state, [SimpleUpdate.pos(node_idx, self.n)], dims))

        for edge_idx in range(self.data_point.edge_index.shape[1]):
            edge = self.data_point.edge_index[:, edge_idx]
            two_rdms.append(compute_rdm(ground_state, [SimpleUpdate.pos(edge[0], self.n), SimpleUpdate.pos(edge[1], self.n)], dims))
        
        return one_rdms, two_rdms
    
    def getGroundState(self):
        return self.psi


class FullUpdate(SimpleUpdate, NonParametrizedAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.is_gpu = False#platform.system() != 'Windows'
        self.tn_type = "PEPS"

    def run(self):
        super().run()

        self.psi0 = self.psi

        if self.is_gpu:
            def to_backend(x):
                import cupy as cp
                return cp.asarray(x).astype('float32')
            self.hamiltonian.apply_to_arrays(to_backend)
            self.psi0.apply_to_arrays(to_backend)

        fu = qtn.FullUpdate(
            psi0 = self.psi0,
            ham = self.hamiltonian,
            chi = self.chi,
            compute_energy_every = None,
            compute_energy_per_site = True,
            keep_best = True,
            progbar = False
        )
        for tau in self.tau:
            fu.evolve(self.num_iters, tau=tau)
        self.psi = fu.best['state']
        self.energy = fu.best['energy'] * np.prod(self.n)


class SimpleUpdateGen(SimpleUpdate, NonParametrizedAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.edges = []
        self.max_bond = params['max_bond']
        self.tn_type = "Generic"
        self.phys_dim = params['bond_dim']


    def set_datapoint(self, data_point):
        self.n = tuple(data_point.grid_extent)
        if len(self.n) == 1:
            self.n = (self.n[0], 1)
        self.data_point = data_point
        self.edges = []
        for (a,b), J_ab in zip(self.data_point.edge_index.T, self.data_point.x_edges):
            a, b, J_ab = int(a), int(b), float(J_ab)
            if J_ab != 0:
                self.edges.append((a, b))
        self.psi0 = qtn.TN_from_edges_rand(self.edges, D=self.bond_dim, phys_dim=self.phys_dim)

    def run(self):
        su = qtn.SimpleUpdateGen(
            psi0 = self.psi0,
            ham = self.hamiltonian,
            compute_energy_every = None,
            compute_energy_per_site = True,
            keep_best = True,
            progbar = True
        )
                
        for tau in self.tau:
            su.evolve(self.num_iters, tau=tau)
            

        self.psi = su.best['state']
        self.energy = su.best['energy'] * np.prod(self.n)

    def create_model(self):

        local_one_site_hamiltonians = {}  # dict for qtn.LocalHam2D H1
        for i, (h,g) in enumerate(self.data_point.x_nodes):
            h, g = float(h), float(g)  # convert from torch to float
            local_one_site_hamiltonians[i] = qu.spin_operator('Z') * h * 2
            local_one_site_hamiltonians[i] += qu.spin_operator('X') * g * 2

        local_two_site_hamiltonians = {}  # dict for qtn.LocalHam2D H2
        for (a,b), J_ab in zip(self.data_point.edge_index.T, self.data_point.x_edges):
            a, b, J_ab = int(a), int(b), float(J_ab)  # convert from torch to int/float
            if J_ab != 0:
                local_two_site_hamiltonians[a, b] = qu.ham_heis(2, j=(0, 0, 4*J_ab))  # factor of 4 because of different convention

        ham_local = qtn.LocalHamGen(H2=local_two_site_hamiltonians, H1=local_one_site_hamiltonians)
        self.hamiltonian = ham_local

    def getRDMs(self):
        one_rdms = []
        two_rdms = []

        m, n = self.n
        dims = [[2] * n] * m
        
        def pos(node, n):
            # calculate x,y coordinates from node index
            y = node % n[1]
            x = (node - y) // n[1]
            # return x, y
            return node

        for node_idx in tqdm(range(self.data_point.x_nodes.shape[0]), desc="1-RDMs", leave=False):
            one_rdms.append(self.psi.partial_trace([pos(node_idx, self.n)], self.max_bond, "auto", normalized=True))

        for edge_idx in tqdm(range(self.data_point.edge_index.shape[1]), desc="2-RDMs", leave=False):
            edge = self.data_point.edge_index[:, edge_idx]
            two_rdms.append(self.psi.partial_trace([pos(edge[0], self.n), pos(edge[1], self.n)], self.max_bond, "auto", normalized=True))
        
        return one_rdms, two_rdms

class CustomIsingMPOModel(CouplingMPOModel):
    default_lattice = "Chain"
    force_default_lattice = False
    

    def init_sites(self, model_param):
        site = SpinHalfSite(conserve=None)
        return site

    def init_lattice(self, model_param):
        sites = self.init_sites(model_param)
        bc = "open" if model_param["bc_MPS"] == "open" else "periodic"
        lat = Chain(model_param["L"], sites, bc=bc, bc_MPS=model_param["bc_MPS"])
        self.L = lat.N_sites
        return lat

    def init_terms(self, model_params):
        # Add local field terms
        for i, (h, g) in model_params["local_fields"]:
            self.add_onsite_term(-h, i, 'Sigmaz')
            self.add_onsite_term(-g, i, 'Sigmax')

        # Add coupling terms
        for (i, j), J in model_params["couplings"]:
            if i > j:
                t = i
                i = j
                j = t
    
            if J != 0:
                self.add_coupling_term(float(J), int(i), int(j),  'Sigmaz', 'Sigmaz')


class DMRG_QUIMB(TensorNetworkAlgorithm, NonParametrizedAlgorithm):
    def __init__(self, params):
        self.params = params
        self.max_bond = params.get('max_bond', 30)  # Maximum bond dimension
        self.cutoff = params.get('cutoff', 1e-10)  # Truncation cutoff for SVD
        self.tolerance = params.get('tol', 1e-6)  # Tolerance for convergence
        self.verbosity = params.get('verbosity', 0)  # Verbosity level
        self.psi = None  # Ground state MPS
        self.energy = None
        self.data_point = None
        self.pbc = False
        self.tn_type = "MPS"

    def set_datapoint(self, data_point):
        if not TensorNetworkAlgorithm.isMPS(data_point):
            raise ValueError("DMRG_QUIMB only supports MPS formatted data points.")
        self.data_point = data_point
        self.L = data_point.grid_extent[0]  # Assuming a 1D system
        self.pbc = data_point.pbc  # Periodic boundary conditions flag
        assert not self.pbc, "Periodic boundary conditions are not supported."

    def create_model(self):
        assert self.data_point is not None
        # Initialize the Hamiltonian builder
        ham_builder = qtn.SpinHam1D(S=1/2)  # Spin-1/2 Hamiltonian

        # Add single-site terms
        for i, (h, g) in enumerate(self.data_point.x_nodes):
            h, g = float(h), float(g)
            ham_builder[i] += 2*h, 'Z'
            ham_builder[i] += 2*g, 'X'

        # Add interaction terms, respecting PBC if necessary
        for (a, b), J_ab in zip(self.data_point.edge_index.T, self.data_point.x_edges):
            J_ab = float(J_ab)
            if J_ab != 0 or (abs(a - b) == 1 or (self.pbc and (abs(a - b) == n[0] - 1))):
                ham_builder[int(a), int(b)] +=  4*J_ab, 'Z', 'Z'
                ham_builder[int(b), int(a)] +=  4*J_ab, 'Z', 'Z'

        # Build the Hamiltonian MPO
        self.H_mpo = ham_builder.build_mpo(self.L)

    def run(self):
        # Setup and run DMRG
        dmrg = qtn.DMRG2(self.H_mpo)
        dmrg.opts['max_bond'] = self.max_bond
        dmrg.opts['cutoff'] = self.cutoff
        dmrg.solve(tol = self.tolerance, verbosity=self.verbosity)
        self.psi = dmrg.state
        self.energy = dmrg.energy


    def getEnergy(self):
        return self.energy.real if self.energy else None
    
    def getRDMs(self):
        one_rdms = []
        two_rdms = []

        dims = [2] * self.L
            

        def compute_rdm(peps, sites, dims):
            """
            Compute the RDM for a list of sites in a PEPS.

            Parameters:
            peps (PEPS): The PEPS representing the quantum state.
            sites (List of tuples): The coordinates of the sites.
            dims (list): The dimensions of the Hilbert space at each site.

            Returns:
            numpy.ndarray: The 2-RDM of the specified sites.
            """          
            return qu.normalize(qu.partial_trace(peps, dims=dims, keep=sites))

        ground_state = self.psi.to_dense()

        for node_idx in range(self.data_point.x_nodes.shape[0]):
            one_rdms.append(compute_rdm(ground_state, [node_idx], dims))

        for edge_idx in range(self.data_point.edge_index.shape[1]):
            edge = self.data_point.edge_index[:, edge_idx]
            two_rdms.append(compute_rdm(ground_state, [edge[0], edge[1]], dims))
        
        return one_rdms, two_rdms

    def getGroundState(self):
        return self.psi


class DMRG(TensorNetworkAlgorithm, NonParametrizedAlgorithm):
    def __init__(self, params):
        self.params = params
        self.max_E_err = params['max_E_err']
        self.chi_max = params['chi_max']
        self.svd_min = params['svd_min']
        self.psi = None
        self.energy = None
        self.data_point = None
        self.psi0 = None
        self.model = None
        self.tn_type = "MPS"
    
    def set_datapoint(self, data_point):
        self.data_point = data_point
        if not TensorNetworkAlgorithm.isMPS(data_point):
            raise ValueError("DMRG only supports MPS")
        self.L = int(self.data_point.grid_extent[0])
        self.bc_MPS = 'infinite' if self.data_point.pbc else 'finite'
        self.bc = 'periodic' if self.data_point.pbc else 'open'
        self.psi = None
        self.energy = None
        self.psi0 = None
        self.model = None
        
        
    def create_model(self):
        local_fields = [(i, (float(h), float(g))) for i, (h, g) in enumerate(self.data_point.x_nodes)]
        couplings = [((int(a), int(b)), float(J_ab)) for (a, b), J_ab in zip(self.data_point.edge_index.T, self.data_point.x_edges)]
        
        # Define model parameters
        model_params = {
            'L': self.L,
            'local_fields': local_fields,
            'couplings': couplings,
            'conserve': None,
            'bc_MPS': self.bc_MPS
        }

        self.model = CustomIsingMPOModel(model_params)
        seq = [random.choice(["up", "down"]) for _ in range(self.model.lat.N_sites)]
        self.psi0 = MPS.from_product_state(self.model.lat.mps_sites(), seq, bc=self.bc_MPS)
        
    def run(self):
        dmrg_params = {
            'mixer': True,
            'max_E_err': self.max_E_err,
            'trunc_params': {
                'chi_max': self.chi_max,
                'svd_min': self.svd_min
            },
            'combine': True,
        }

        if self.L > 2:
            eng = dmrg.TwoSiteDMRGEngine(self.psi0, self.model, dmrg_params)
        else:
            eng = dmrg.SingleSiteDMRGEngine(self.psi0, self.model, dmrg_params)
            
        self.energy, self.psi = eng.run()
        if self.data_point.pbc:
            self.energy *= self.model.lat.N_sites


    def getEnergy(self):
        return self.energy
    
    def getRDMs(self):
        #Use TeNPy function
        one_rdms = []
        two_rdms = []

        def compute_RDM_TeNPy(psi, indices):

            one_rdm = len(indices) == 1
            two_rdm = len(indices) == 2

            if not one_rdm and not two_rdm:
                raise ValueError("The 'indices' argument must be a tuple of one or two integers.")
            
            if one_rdm:
                rdm = psi.get_rho_segment(indices)
                rdm = rdm.to_ndarray()
                rdm = rdm[::-1,::-1]
                off = np.eye(rdm.shape[0]) != 1
                rdm[off] *= -1
            elif two_rdm:
                rdm = psi.get_rho_segment(indices)
                rdm = rdm.to_ndarray()
                rdm = rdm.reshape(4, 4)
                rdm = rdm[::-1, ::-1]
                off = np.logical_and(np.eye(4) != 1, np.eye(4)[::-1] != 1)
                rdm[off] *= -1
                # if indices are not adjacent, we need to transform the 2-RDM
                def swap(a, i, j):
                    a[i], a[j] = a[j], a[i]
                if abs(indices[0] - indices[1]) != 1:
                    # swap indices to match the label
                    swap(rdm, (0,2), (1,0))
                    swap(rdm, (0,3), (1,1))
                    swap(rdm, (2,2), (3,0))
                    swap(rdm, (2,3), (3,1))

            return rdm

        for node_idx in range(self.data_point.x_nodes.shape[0]):
            one_rdms.append(compute_RDM_TeNPy(self.psi, [node_idx]))

        for edge_idx in range(self.data_point.edge_index.shape[1]):
            edge = self.data_point.edge_index[:, edge_idx]
            two_rdms.append(compute_RDM_TeNPy(self.psi, [edge[0], edge[1]]))
    
        return one_rdms, two_rdms        

    def getGroundState(self):
        return self.psi

class DataPointError(Exception):
    def __init__(self, index, message="Error in datapoint"):
        self.index = index
        self.message = message
        super().__init__(f"{message} at batch index {index}")
    
class TensorNetworkRunner():
    def __init__(self, algorithm, params) -> None:
        assert issubclass(algorithm, TensorNetworkAlgorithm)
        self.algorithm = algorithm(params)
        self.tn_type = self.algorithm.tn_type

    def run(self, data_point):
        self.algorithm.set_datapoint(data_point)
        self.algorithm.create_model()
        self.algorithm.run()
        return self.algorithm.getEnergy(), self.algorithm.getRDMs()

    def resetParams(self, params):
        self.algorithm = self.algorithm.__class__(params)

    def __call__(self, batch):
        energies = []
        onerdms = []
        twordms = []

        for i in tqdm(range(batch.num_graphs), desc="Batch", leave=False):
            succeeded = False
            tries = 0
            while not succeeded: 
                start_ptr = batch.ptr[i]
                end_ptr = batch.ptr[i + 1]

                edge_mask = (batch.edge_index[0] >= start_ptr) & (batch.edge_index[0] < end_ptr)
                edges = batch.edge_index[:, edge_mask]

                # Adjust the edges to be relative to the start of the i-th graph
                edges[0] -= start_ptr
                edges[1] -= start_ptr

                data_point = Data(
                    x_nodes=batch.x_nodes[start_ptr:end_ptr],
                    edge_index=edges,
                    x_edges=batch.x_edges[edge_mask],
                    grid_extent=batch.grid_extent[i],
                    pbc=batch.pbc[i]
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        energy, rdms = self.run(data_point)
                    succeeded = True
                except ValueError:
                    print("Batch point error during processing at index", i)
                    tries += 1
                    if tries > 3:
                        raise DataPointError(i, message="Batch point error during processing")
                    
            energies.append(energy.real)
            onerdms.append(np.stack(rdms[0], axis=0))
            twordms.append(np.stack(rdms[1], axis=0))

        energies = torch.tensor(energies, dtype=torch.float32)

        return onerdms, twordms, energies

    def to(self, device):
        return self
    def eval(self):
        return self
    def train(self):
        return self
    def parameters(self):
        return self.algorithm.parameters()
    
