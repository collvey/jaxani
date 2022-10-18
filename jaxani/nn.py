from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp

class SpeciesCoordinates(NamedTuple):
  species: jnp.ndarray
  coordinates: jnp.ndarray

# This constant, when indexed with the corresponding atomic number, gives the
# element associated with it. Note that there is no element with atomic number
# 0, so 'Dummy' returned in this case.
PERIODIC_TABLE = ['Dummy'] + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()

class SpeciesConverter():
    """Converts arrays with species labeled as atomic numbers into arrays
    labeled with internal indices according to a custom ordering scheme. It 
    takes a custom species ordering as initialization parameter. If the class is
    initialized with ['H', 'C', 'N', 'O'] for example, it will convert a arrays 
    [1, 1, 6, 7, 1, 8] into a arrays [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_array: jnp.ndarray

    def __init__(self, species):
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.conv_array = jnp.zeros(maxidx+2, dtype=jnp.int32) - 1
        for i, s in enumerate(species):
          self.conv_array = self.conv_array.at[rev_idx[s]].set(i)

    def __call__(self, input_: Tuple[jnp.ndarray, jnp.ndarray],
                cell: Optional[jnp.ndarray] = None,
                pbc: Optional[jnp.ndarray] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_array[species]

        # check if unknown species are included
        if (converted_species[species != -1] < 0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species, coordinates)