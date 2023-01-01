from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax
import math
import numpy as np

# Enable support double precision
# See https://github.com/google/jax#current-gotchas
jax.config.update("jax_enable_x64", True)

class FakeFinal:
    def __getitem__(self, x):
        return x

Final = FakeFinal()

class SpeciesAEV(NamedTuple):
    species: jnp.ndarray
    aevs: jnp.ndarray

@partial(jax.jit, static_argnums=(0,1,2,3,4,))
def compute_shifts(cell_x: int, cell_y: int, cell_z: int, pbc: bool, cutoff: float) -> jnp.ndarray:
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration

    Arguments:
        cell (:class:`jnp.ndarray`): ndarray of shape (3, 3) of the three
        vectors defining unit cell:
            ndarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`np.ndarray`): boolean storing if pbc is enabled for all 
            directions.

    Returns:
        :class:`jnp.ndarray`: long ndarray of shifts. the center cell and
            symmetric cells are not included.
    """
    cell = np.array([[cell_x, 0, 0],[0, cell_y, 0],[0, 0, cell_z]])
    reciprocal_cell = np.linalg.inv(cell).T
    inv_distances = np.linalg.norm(reciprocal_cell, axis=1)
    num_repeats = np.ceil(cutoff * inv_distances).astype(np.int64)
    num_repeats = np.where(pbc, num_repeats, np.zeros(num_repeats.shape))
    long_shifts = propagate_shifts_from_repeats(num_repeats[0], num_repeats[1], num_repeats[2])
    return long_shifts

@partial(jax.jit, static_argnums=(0,1,2))
def propagate_shifts_from_repeats(num_repeats_x: int, num_repeats_y: int, num_repeats_z: int):
    long_shifts = jnp.concatenate([
        jnp.mgrid[
            1:num_repeats_x + 1, 
            -num_repeats_y:num_repeats_y + 1, 
            -num_repeats_z:num_repeats_z + 1].reshape(3, -1).T,
        jnp.mgrid[
            0:1, 
            1:num_repeats_y + 1, 
            -num_repeats_z:num_repeats_z + 1].reshape(3, -1).T,
        jnp.mgrid[
            0:1, 
            0:1, 
            1:num_repeats_z + 1].reshape(3, -1).T,
    ])
    return long_shifts

def triu_index(num_species: int) -> jnp.ndarray:
    species1, species2 = jax_unbind(jax_triu_indices(num_species, num_species), 0)
    pair_index = jnp.arange(species1.shape[0], dtype=jnp.int64)
    ret = jnp.zeros((num_species, num_species), dtype=jnp.int64)
    ret = ret.at[species1, species2].set(pair_index)
    ret = ret.at[species2, species1].set(pair_index)
    return ret

def jax_triu_indices(row, col, offset=0):
  return jnp.stack(jnp.triu_indices(row, offset, col))

def jax_tril_indices(row, col, offset=0):
  return jnp.stack(jnp.tril_indices(row, offset, col))

def jax_unbind(input, dim=0):
    # Adapted from jakevdp: https://github.com/google/jax/discussions/11028
    return [jax.lax.index_in_dim(
        input, i, axis=dim, keepdims=False) for i in range(input.shape[dim])]

def compute_aev(species: jnp.ndarray, coordinates: jnp.ndarray, triu_index: jnp.ndarray,
                constants: Tuple[float, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                sizes: Tuple[int, int, int, int, int], cell_shifts: Optional[Tuple[jnp.ndarray, jnp.ndarray]]) -> jnp.ndarray:
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
    num_species, radial_sublength, radial_length, angular_sublength, angular_length = sizes
    num_molecules = species.shape[0]
    num_atoms = species.shape[1]
    num_species_pairs = angular_length // angular_sublength
    coordinates_ = coordinates
    coordinates = coordinates_.reshape(-1, 3)

    # PBC calculation is bypassed if there are no shifts
    if cell_shifts is None:
        atom_index12, vec = adjust_pair_nopbc(species, coordinates, Rcr, coordinates_)
    else:
        atom_index12, vec = adjust_pair_pbc(species, coordinates, cell_shifts, Rcr, coordinates_)

    species = species.flatten()
    species12 = species[atom_index12]

    distances = jnp.linalg.norm(vec, axis=1)

    # compute radial aev
    radial_terms_ = radial_terms(Rcr, EtaR, ShfR, distances)
    radial_aev = jnp.zeros((num_molecules * num_atoms * num_species, radial_sublength), dtype=radial_terms_.dtype)
    index12 = atom_index12 * num_species + jnp.flip(species12, axis=0)
    radial_aev = radial_aev.at[index12[0]].add(radial_terms_)
    radial_aev = radial_aev.at[index12[1]].add(radial_terms_)
    radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)

    # Rca is usually much smaller than Rcr, using neighbor list with cutoff=Rcr is a waste of resources
    # Now we will get a smaller neighbor list that only cares about atoms with distances <= Rca
    even_closer_indices = jnp.ravel(jnp.stack((distances <= Rca).nonzero()).T)
    atom_index12 = jnp.take(atom_index12, even_closer_indices, 1)
    species12= jnp.take(species12, even_closer_indices, 1)

    vec = jnp.take(vec, even_closer_indices, 0)

    # compute angular aev
    central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
    species12_small = species12[:, pair_index12]
    vec12 = jnp.take(vec, pair_index12.reshape(-1), 0).reshape(2, -1, 3) * jnp.expand_dims(sign12, axis=-1)
    species12_ = jnp.where(sign12 == 1, species12_small[1], species12_small[0])
    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
    angular_aev = jnp.zeros(
        (num_molecules * num_atoms * num_species_pairs, angular_sublength),
        dtype=angular_terms_.dtype)
    index = central_atom_index * num_species_pairs + triu_index[species12_[0], species12_[1]]
    angular_aev = angular_aev.at[index].add(angular_terms_)
    angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
    return jnp.concatenate([radial_aev, angular_aev], axis=-1)

def adjust_pair_nopbc(species, coordinates, Rcr, coordinates_):
    atom_index12 = neighbor_pairs_nopbc(species == -1, coordinates_, Rcr)
    selected_coordinates = jnp.take(coordinates, atom_index12.reshape(-1), axis=0).reshape(2, -1, 3)
    vec = selected_coordinates[0] - selected_coordinates[1]
    return atom_index12, vec

def adjust_pair_pbc(species, coordinates, cell_shifts, Rcr, coordinates_):
    # print('coordinates shape: ', coordinates.shape)
    cell, shifts = cell_shifts
    atom_index12, shifts = neighbor_pairs(species == -1, coordinates_, cell, shifts, Rcr)
    # print('atom_index12: ', atom_index12)
    # print('shifts shape: ', shifts.shape)
    shift_values = shifts.astype(cell.dtype) @ cell
    selected_coordinates = jnp.take(coordinates, atom_index12.reshape(-1), 0).reshape(2, -1, 3)
    # print('selected_coordinates shape: ', selected_coordinates.shape)
    vec = selected_coordinates[0] - selected_coordinates[1] + shift_values
    # print('vec: ', vec)
    return atom_index12, vec

def neighbor_pairs_nopbc(padding_mask: jnp.ndarray, coordinates: jnp.ndarray, cutoff: float) -> jnp.ndarray:
    """Compute pairs of atoms that are neighbors (doesn't use PBC)

    This function bypasses the calculation of shifts and duplication
    of atoms in order to make calculations faster

    Arguments:
        padding_mask (:class:`jnp.ndarray`): boolean ndarray of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`jnp.ndarray`): ndarray of shape
            (molecules, atoms, 3) for atom coordinates.
        cutoff (float): the cutoff inside which atoms are considered pairs
    """
    coordinates = jnp.where(jnp.expand_dims(padding_mask, axis=-1), math.nan, jax.lax.stop_gradient(coordinates))
    num_atoms = padding_mask.shape[1]
    num_mols = padding_mask.shape[0]
    p12_all = jnp.stack(jnp.triu_indices(num_atoms, 1, num_atoms))
    p12_all_flattened = p12_all.reshape(-1)

    pair_coordinates = jnp.take(coordinates, p12_all_flattened, axis=1).reshape(num_mols, 2, -1, 3)
    distances = jnp.linalg.norm(pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...], axis=2)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index] + molecule_index
    return atom_index12

def neighbor_pairs(padding_mask: jnp.ndarray, coordinates: jnp.ndarray, cell: jnp.ndarray,
                   shifts: jnp.ndarray, cutoff: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute pairs of atoms that are neighbors

    Arguments:
        padding_mask (:class:`jnp.ndarray`): boolean ndarray of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`jnp.ndarray`): ndarray of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`jnp.ndarray`): ndarray of shape (3, 3) of the three vectors
            defining unit cell: ndarray([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`jnp.ndarray`): ndarray of shape (?, 3) storing shifts
    """
    coordinates = jnp.where(jnp.expand_dims(padding_mask, axis=-1), math.nan, jax.lax.stop_gradient(coordinates))
    cell = jax.lax.stop_gradient(cell)
    num_atoms = padding_mask.shape[1]
    num_mols = padding_mask.shape[0]

    # Step 2: center cell
    # jax_tril_indices is faster than combinations
    p12_center = jax_triu_indices(num_atoms, num_atoms, 1)
    shifts_center = jnp.zeros((p12_center.shape[1], 3), dtype=shifts.dtype)

    # Step 3: cells with shifts
    # shape convention (shift index, molecule index, atom index, 3)
    num_shifts = shifts.shape[0]
    prod = jnp.mgrid[0:num_shifts, 0:num_atoms, 0:num_atoms].reshape(3, -1)
    shift_index = prod[0]
    p12 = prod[1:]
    shifts_outside = jnp.take(shifts, shift_index, 0)

    # Step 4: combine results for all cells
    shifts_all = jnp.concatenate([shifts_center, shifts_outside])
    p12_all = jnp.concatenate([p12_center, p12], axis=1)
    shift_values = shifts_all.astype(cell.dtype) @ cell

    # step 5, compute distances, and find all pairs within cutoff
    selected_coordinates = jnp.take(coordinates, p12_all.reshape(-1), 1).reshape(num_mols, 2, -1, 3)
    distances = jax_norm(selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shift_values, 2, -1)
    in_cutoff = jax_nonzero(distances <= cutoff)
    molecule_index, pair_index = jax_unbind(in_cutoff, 1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index]
    shifts = jnp.take(shifts_all, pair_index, 0)
    return molecule_index + atom_index12, shifts

def jax_norm(input, ord, dim):
    return jnp.linalg.norm(input, ord=ord, axis=dim)

def jax_nonzero(input):
    return jnp.stack(input.nonzero()).T

def radial_terms(Rcr: float, EtaR: jnp.ndarray, ShfR: jnp.ndarray, distances: jnp.ndarray) -> jnp.ndarray:
    """Compute the radial subAEV terms of the center atom given neighbors

    This correspond to equation (3) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where ``N``
    is the number of neighbor atoms within the cutoff radius and output
    tensor should have shape
    (conformations, atoms, ``self.radial_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    distances = distances.reshape(-1, 1, 1)
    fc = cutoff_cosine(distances, Rcr)
    # Note that in the equation in the paper there is no 0.25
    # coefficient, but in NeuroChem there is such a coefficient.
    # We choose to be consistent with NeuroChem instead of the paper here.
    ret = 0.25 * jnp.exp(-EtaR * (distances - ShfR)**2) * fc
    # At this point, ret now has shape
    # (conformations x atoms, ?, ?) where ? depend on constants.
    # We then should flat the last 2 dimensions to view the subAEV as a two
    # dimensional tensor (onnx doesn't support negative indices in flatten)
    return ret.reshape(ret.shape[0], ret.shape[1])

def cutoff_cosine(distances: jnp.ndarray, cutoff: float) -> jnp.ndarray:
    # assuming all elements in distances are smaller than cutoff
    return 0.5 * jnp.cos(distances * (math.pi / cutoff)) + 0.5

def triple_by_molecule(atom_index12: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Input: indices for pairs of atoms that are close to each other.
    each pair only appear once, i.e. only one of the pairs (1, 2) and
    (2, 1) exists.

    Output: indices for all central atoms and it pairs of neighbors. For
    example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
    central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
    are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
    """
    # convert representation from pair to central-others
    ai1 = atom_index12.reshape(-1)
    sorted_ai1 = jnp.sort(ai1)
    rev_indices = jnp.argsort(ai1)

    # sort and compute unique key
    uniqued_central_atom_index, counts = jnp.unique(sorted_ai1, return_inverse=False, return_counts=True)

    # compute central_atom_index
    pair_sizes = jnp.asarray(
        jnp.fix(jnp.divide(counts * (counts - 1), 2)), dtype=jnp.int64)
    pair_indices = jnp.repeat(jnp.arange(pair_sizes.size), pair_sizes)
    central_atom_index = jnp.take(uniqued_central_atom_index, pair_indices, 0)

    # do local combinations within unique key, assuming sorted
    m = counts.max().item() if counts.size > 0 else 0
    n = pair_sizes.shape[0]
    intra_pair_indices = jnp.repeat(jnp.expand_dims(
        jnp.stack(jnp.tril_indices(m, -1, m)), axis=1), repeats=n, axis=1)
    mask = (jnp.arange(intra_pair_indices.shape[2]) < jnp.expand_dims(pair_sizes, axis=1)).flatten()
    assert(len(intra_pair_indices.shape) == 3)
    # Flatten starting from axis 1 to axis 2
    indices_shape = intra_pair_indices.shape
    sorted_local_index12 = intra_pair_indices.reshape(
        (indices_shape[0], indices_shape[1]*indices_shape[2]))[:, mask]
    sorted_local_index12 += jnp.take(jax_cumsum_from_zero(counts).astype(jnp.int64), pair_indices, 0)

    # unsort result from last part
    local_index12 = rev_indices[sorted_local_index12]

    # compute mapping between representation of central-other to pair
    n = atom_index12.shape[1]
    sign12 = (jnp.asarray(local_index12 < n, dtype=jnp.int8) * 2) - 1
    return central_atom_index, local_index12 % n, sign12

def jax_cumsum_from_zero(input_: jnp.ndarray, dim=0) -> jnp.ndarray:
    cumsum = jnp.zeros_like(input_)
    return jnp.concatenate(
        (jnp.zeros((1,) + input_.shape[1:]), 
        jnp.cumsum(input_[:-1], axis=dim)))

def angular_terms(Rca: float, ShfZ: jnp.ndarray, EtaA: jnp.ndarray, Zeta: jnp.ndarray,
                  ShfA: jnp.ndarray, vectors12: jnp.ndarray) -> jnp.ndarray:
    """Compute the angular subAEV terms of the center atom given neighbor pairs.

    This correspond to equation (4) in the `ANI paper`_. This function just
    compute the terms. The sum in the equation is not computed.
    The input tensor have shape (conformations, atoms, N), where N
    is the number of neighbor atom pairs within the cutoff radius and
    output tensor should have shape
    (conformations, atoms, ``self.angular_sublength()``)

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    vectors12 = vectors12.reshape(2, -1, 3, 1, 1, 1, 1)
    distances12 = jnp.linalg.norm(vectors12, ord=2, axis=-5)
    cos_angles = vectors12.prod(0).sum(1) / jnp.clip(distances12.prod(0), a_min=1e-10)
    # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
    angles = jnp.arccos(0.95 * cos_angles)

    fcj12 = cutoff_cosine(distances12, Rca)
    factor1 = ((1 + jnp.cos(angles - ShfZ)) / 2) ** Zeta
    factor2 = jnp.exp(-EtaA * (distances12.sum(0) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj12.prod(0)
    # At this point, ret now has shape
    # (conformations x atoms, ?, ?, ?, ?) where ? depend on constants.
    # We then should flat the last 4 dimensions to view the subAEV as a two
    # dimensional tensor (onnx doesn't support negative indices in flatten)
    assert(len(ret.shape) == 5)
    ret_shape = ret.shape
    return ret.reshape((
        ret_shape[0], 
        ret_shape[1]*ret_shape[2]*ret_shape[3]*ret_shape[4]))

class AEVComputer():
    r"""The AEV computer that takes coordinates as input and outputs aevs.

    Arguments:
        Rcr (float): :math:`R_C` in equation (2) when used at equation (3)
            in the `ANI paper`_.
        Rca (float): :math:`R_C` in equation (2) when used at equation (4)
            in the `ANI paper`_.
        EtaR (:class:`jnp.ndarray`): The 1D tensor of :math:`\eta` in
            equation (3) in the `ANI paper`_.
        ShfR (:class:`jnp.ndarray`): The 1D tensor of :math:`R_s` in
            equation (3) in the `ANI paper`_.
        EtaA (:class:`jnp.ndarray`): The 1D tensor of :math:`\eta` in
            equation (4) in the `ANI paper`_.
        Zeta (:class:`jnp.ndarray`): The 1D tensor of :math:`\zeta` in
            equation (4) in the `ANI paper`_.
        ShfA (:class:`jnp.ndarray`): The 1D tensor of :math:`R_s` in
            equation (4) in the `ANI paper`_.
        ShfZ (:class:`jnp.ndarray`): The 1D tensor of :math:`\theta_s` in
            equation (4) in the `ANI paper`_.
        num_Torchspecies (int): Number of supported atom types.

    .. _ANI paper:
        http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
    """
    Rcr: Final[float]
    Rca: Final[float]
    num_species: Final[int]

    radial_sublength: Final[int]
    radial_length: Final[int]
    angular_sublength: Final[int]
    angular_length: Final[int]
    aev_length: Final[int]
    sizes: Final[Tuple[int, int, int, int, int]]
    triu_index: jnp.ndarray

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species):
        super().__init__()
        self.Rcr = Rcr
        self.Rca = Rca
        assert Rca <= Rcr, "Current implementation of AEVComputer assumes Rca <= Rcr"
        self.num_species = num_species

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (..., EtaR, ShfR)
        # self.register_buffer('EtaR', EtaR.view(-1, 1))
        # self.register_buffer('ShfR', ShfR.view(1, -1))
        self.EtaR = EtaR.reshape(-1, 1)
        self.ShfR = ShfR.reshape(-1, 1)
        
        # shape convension (..., EtaA, Zeta, ShfA, ShfZ)
        # self.register_buffer('EtaA', EtaA.view(-1, 1, 1, 1))
        # self.register_buffer('Zeta', Zeta.view(1, -1, 1, 1))
        # self.register_buffer('ShfA', ShfA.view(1, 1, -1, 1))
        # self.register_buffer('ShfZ', ShfZ.view(1, 1, 1, -1))
        self.EtaA = EtaA.reshape(-1, 1, 1, 1)
        self.Zeta = Zeta.reshape(1, -1, 1, 1)
        self.ShfA = ShfA.reshape(1, -1, 1, 1)
        self.ShfZ = ShfZ.reshape(1, 1, 1, -1)

        # The length of radial subaev of a single species
        self.radial_sublength = self.EtaR.size * self.ShfR.size
        # The length of full radial aev
        self.radial_length = self.num_species * self.radial_sublength
        # The length of angular subaev of a single species
        self.angular_sublength = self.EtaA.size * self.Zeta.size * self.ShfA.size * self.ShfZ.size
        # The length of full angular aev
        self.angular_length = (self.num_species * (self.num_species + 1)) // 2 * self.angular_sublength
        # The length of full aev
        self.aev_length = self.radial_length + self.angular_length
        self.sizes = self.num_species, self.radial_sublength, self.radial_length, self.angular_sublength, self.angular_length

        # self.register_buffer('triu_index', triu_index(num_species).to(device=self.EtaR.device))
        self.triu_index = triu_index(num_species)

        # Set up default cell and compute default shifts.
        # These values are used when cell and pbc switch are not given.
        cutoff = max(self.Rcr, self.Rca)
        default_cell = np.ones(3)
        default_pbc = False
        default_shifts = compute_shifts(*default_cell, default_pbc, cutoff)
        # self.register_buffer('default_cell', default_cell)
        # self.register_buffer('default_shifts', default_shifts)
        self.default_cell = default_cell
        self.default_shifts = default_shifts

    def init_cuaev_computer(self):
        pass

    def compute_cuaev(self, species, coordinates):
        pass

    @classmethod
    def cover_linearly(cls, radial_cutoff: float, angular_cutoff: float,
                       radial_eta: float, angular_eta: float,
                       radial_dist_divisions: int, angular_dist_divisions: int,
                       zeta: float, angle_sections: int, num_species: int,
                       angular_start: float = 0.9, radial_start: float = 0.9):
        r""" Provides a convenient way to linearly fill cutoffs

        This is a user friendly constructor that builds an
        :class:`AEVComputer` where the subdivisions along the the
        distance dimension for the angular and radial sub-AEVs, and the angle
        sections for the angular sub-AEV, are linearly covered with shifts. By
        default the distance shifts start at 0.9 Angstroms.

        To reproduce the ANI-1x AEV's the signature ``(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)``
        can be used.
        """
        # This is intended to be self documenting code that explains the way
        # the AEV parameters for ANI1x were chosen. This is not necessarily the
        # best or most optimal way but it is a relatively sensible default.
        Rcr = radial_cutoff
        Rca = angular_cutoff
        EtaR = jnp.array([float(radial_eta)])
        EtaA = jnp.array([float(angular_eta)])
        Zeta = jnp.array([float(zeta)])

        ShfR = jnp.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]
        ShfA = jnp.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[:-1]
        angle_start = math.pi / (2 * angle_sections)

        ShfZ = (jnp.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]

        return cls(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

    def constants(self):
        return self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

    def forward(self, input_: Tuple[jnp.ndarray, jnp.ndarray],
                cell: Optional[np.ndarray] = None,
                pbc: Optional[bool] = False) -> SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`jaxani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, ) of the
                three sizes of the unit cell:

                .. code-block:: python

                    tensor([cell_x, cell_y, cell_z])

                and pbc is boolean storing if pbc is enabled
                for all directions.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length())``
        """
        species, coordinates = input_
        assert species.ndim == 2
        assert species.shape == coordinates.shape[:-1]
        assert coordinates.shape[-1] == 3

        if cell is None and not pbc:
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, None)
        else:
            assert (cell is not None and pbc is not None)
            cutoff = max(self.Rcr, self.Rca)
            cell_x, cell_y, cell_z = cell
            shifts = compute_shifts(cell_x, cell_y, cell_z, pbc, cutoff)
            cell = jnp.array([[cell_x, 0, 0],[0, cell_y, 0],[0, 0, cell_z]])
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, (cell, shifts))

        return SpeciesAEV(species, aev)