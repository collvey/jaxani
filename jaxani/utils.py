from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp

class SpeciesEnergies(NamedTuple):
    species: jnp.ndarray
    energies: jnp.ndarray

def load_sae(filename, return_dict=False):
    """Returns an object of :class:`EnergyShifter` with self energies from
    NeuroChem sae file"""
    self_energies = []
    d = {}
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            species = line[0].split(',')[0].strip()
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            d[species] = value
            self_energies.append((index, value))
    self_energies = [i for _, i in sorted(self_energies)]
    if return_dict:
        return EnergyShifter(self_energies), d
    return EnergyShifter(self_energies)

class EnergyShifter():
    """Helper class for adding and subtracting self atomic energies

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
        fit_intercept (bool): Whether to calculate the intercept during the LSTSQ
            fit. The intercept will also be taken into account to shift energies.
    """

    def __init__(self, self_energies, fit_intercept=False):
        super().__init__()

        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = jnp.array(self_energies, dtype=jnp.double)

        self.self_energies = self_energies

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`jnp.ndarray`): Long array in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`jnp.ndarray`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]

        self_energies = self.self_energies[species]
        self_energies.at[species == jnp.array(-1)].set(jnp.array(0, dtype=jnp.double))
        return self_energies.sum(axis=1) + intercept

    def forward(self, species_energies: Tuple[jnp.ndarray, jnp.ndarray],
                cell: Optional[jnp.ndarray] = None,
                pbc: Optional[jnp.ndarray] = None) -> SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species)
        return SpeciesEnergies(species, energies + sae)

class ChemicalSymbolsToInts():
    r"""Helper that can be called to convert chemical symbol string to integers
    On initialization the class should be supplied with a :class:`list` (or in
    general :class:`collections.abc.Sequence`) of :class:`str`. The returned
    instance is a callable object, which can be called with an arbitrary list
    of the supported species that is converted into a tensor of dtype
    :class:`jnp.long`. Usage example:
    .. code-block:: python
       # We initialize ChemicalSymbolsToInts with the supported species
       species_to_tensor = ChemicalSymbolsToInts(['H', 'C', 'Fe', 'Cl'])
       # We have a species list which we want to convert to an index tensor
       index_tensor = species_to_tensor(['H', 'C', 'H', 'H', 'C', 'Cl', 'Fe'])
       # index_tensor is now [0 1 0 0 1 3 2]
    .. warning::
        If the input is a string python will iterate over
        characters, this means that a string such as 'CHClFe' will be
        intepreted as 'C' 'H' 'C' 'l' 'F' 'e'. It is recommended that you
        input either a :class:`list` or a :class:`numpy.ndarray` ['C', 'H', 'Cl', 'Fe'],
        and not a string. The output of a call does NOT correspond to a
        tensor of atomic numbers.
    Arguments:
        all_species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    rev_species: Dict[str, int]

    def __init__(self, all_species: Sequence[str]):
        super().__init__()
        self.rev_species = {s: i for i, s in enumerate(all_species)}

    def forward(self, species: List[str]) -> jnp.ndarray:
        r"""Convert species from sequence of strings to 1D tensor"""
        rev = [self.rev_species[s] for s in species]
        return jnp.array(rev, dtype=jnp.int64)

    def __len__(self):
        return len(self.rev_species)