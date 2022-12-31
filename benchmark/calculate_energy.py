import jax
import jax.numpy as jnp
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from flax.training import checkpoints
from functools import partial
from jaxani.constants import Constants
from jaxani.aev import AEVComputer
from jaxani.nn import SpeciesConverter
from jaxani.utils import load_sae
from jaxani.model import rebuild_model_ensemble
from jaxutil.timer import measure_time, Timer
from test_util.generate_test_checkpoint import generate_test_checkpoint
from neurochem.parse_resources import parse_neurochem_resources

CKPT_DIR = os.path.join(os.path.dirname(__file__), '../test/test_ckpts')
CKPT_PREFIX = 'test_ensemble_'

def jax_energy_from_restored_state(test_coordinates):
    test_species = [[6, 1, 7, 8, 1]] # static

    jax_species_raw, jax_coordinates_raw, info_file = constant_initialization(test_species, test_coordinates)

    # Loads info file
    jax_aev_computer, jax_species_converter, jax_energy_shifter = load_info(info_file)

    # Converts species from periodic table index to internal ordering scheme
    jax_species, jax_coordinates = species_conversion(jax_species_converter, jax_species_raw, jax_coordinates_raw)

    # Computes AEVs
    jax_aevs = compute_aevs(jax_aev_computer, jax_species, jax_coordinates)

    # Load ensemble model and params from restored state
    restored_state, rebuilt_model_ensemble = load_ensemble()

    # Calculate potential energy and add atomic energies
    total_energy = calculate_total_energy(rebuilt_model_ensemble, restored_state, jax_species, jax_aevs, jax_energy_shifter)

    return total_energy

@measure_time
def constant_initialization(test_species, test_coordinates):
    jax_species = jnp.array(test_species)
    jax_coordinates = jnp.array(test_coordinates)
    info_file = 'ani-2x_8x.info'
    return jax_species, jax_coordinates, info_file

@measure_time
def load_info(info_file):
    const_file, sae_file, _ensemble_prefix, _ensemble_size = parse_neurochem_resources(info_file)

    consts = Constants(const_file)
    jax_aev_computer = AEVComputer(**consts)
    jax_species_converter = SpeciesConverter(consts.species)
    jax_energy_shifter, _sae_dict = load_sae(sae_file, return_dict=True)
    return jax_aev_computer, jax_species_converter, jax_energy_shifter

@measure_time
def species_conversion(jax_species_converter, jax_species_raw, jax_coordinates_raw):
    # Converts raw jax species and coordinates into converted species and coordinates
    jax_species, jax_coordinates = jax_species_converter((
            jax_species_raw, jax_coordinates_raw))
    return jax_species,jax_coordinates

@measure_time
def compute_aevs(jax_aev_computer, jax_species, jax_coordinates):
    _, jax_aevs = jax_aev_computer.forward((jax_species, jax_coordinates))
    return jax_aevs

@measure_time
def load_ensemble():
    if not os.path.exists(os.path.join(CKPT_DIR, f'{CKPT_PREFIX}0')):
        generate_test_checkpoint()
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None, prefix=CKPT_PREFIX)
    rebuilt_model_ensemble = rebuild_model_ensemble(restored_state['params'])
    return restored_state,rebuilt_model_ensemble

@measure_time
def calculate_total_energy(rebuilt_model_ensemble, restored_state, jax_species, jax_aevs, jax_energy_shifter):
    # Calculates potential energy
    _, total_energy = rebuilt_model_ensemble.apply(restored_state['params'], (jax_species, jax_aevs))

    # Adds atomic energies
    total_energy = total_energy + jax_energy_shifter.sae(jax_species)
    return total_energy[0]

if __name__ == '__main__':
    test_coordinates = [[
        [0.03192167, 0.00638559, 0.01301679],
        [-0.83140486, 0.39370209, -0.26395324],
        [-0.66518241, -0.84461308, 0.20759389],
        [0.45554739, 0.54289633, 0.81170881],
        [0.66091919, -0.16799635, -0.91037834]]]
    energy = jax_energy_from_restored_state(test_coordinates)
    print(energy)