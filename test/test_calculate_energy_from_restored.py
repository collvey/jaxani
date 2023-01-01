from flax.training import train_state, checkpoints
import jax.numpy as jnp
import torch
import torchani
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.constants import Constants
from jaxani.aev import AEVComputer
from jaxani.nn import SpeciesConverter
from jaxani.utils import load_sae
from jaxani.model import rebuild_model_ensemble
from test_util.generate_test_checkpoint import generate_test_checkpoint
from neurochem.parse_resources import parse_neurochem_resources

CKPT_DIR = os.path.join(os.path.dirname(__file__), 'test_ckpts')
CKPT_PREFIX = 'test_ensemble_'
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def jax_energy_from_restored_state(test_species, test_coordinates):
  jax_species = jnp.array(test_species)
  jax_coordinates = jnp.array(test_coordinates)

  info_file = 'ani-2x_8x.info'
  # Loads info file
  const_file, sae_file, _ensemble_prefix, _ensemble_size = parse_neurochem_resources(info_file)

  consts = Constants(const_file)
  jax_aev_computer = AEVComputer(**consts)
  jax_species_converter = SpeciesConverter(consts.species)
  jax_energy_shifter, _sae_dict = load_sae(sae_file, return_dict=True)

  # Converts species from periodic table index to internal ordering scheme
  jax_species, jax_coordinates = jax_species_converter((
      jax_species, jax_coordinates))
  # Computes AEVs
  jax_species, jax_aevs = jax_aev_computer.forward((jax_species, jax_coordinates))
  # Load ensemble model and params from restored state
  if not os.path.exists(os.path.join(CKPT_DIR, f'{CKPT_PREFIX}0')):
    generate_test_checkpoint()
  restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None, prefix=CKPT_PREFIX)
  rebuilt_model_ensemble = rebuild_model_ensemble(restored_state['params'])
  # Calculates potential energy
  _, total_energy = rebuilt_model_ensemble.apply(restored_state['params'], (jax_species, jax_aevs))
  # Adds atomic energies
  total_energy = total_energy + jax_energy_shifter.sae(jax_species)
  return total_energy

def torch_energy(test_species, test_coordinates):
  torch_model = torchani.models.ANI2x(periodic_table_index=True).to(TORCH_DEVICE)

  torch_coordinates = torch.tensor(test_coordinates, requires_grad=True, device=TORCH_DEVICE)
  torch_species = torch.tensor(test_species, device=TORCH_DEVICE)

  energy = torch_model((torch_species, torch_coordinates)).energies
  return energy

class ModelSaveLoadTest(unittest.TestCase):
  def setUp(self):
    self.test_species = [[6, 1, 7, 8, 1]]
    self.test_coordinates = [[
        [0.03192167, 0.00638559, 0.01301679],
        [-0.83140486, 0.39370209, -0.26395324],
        [-0.66518241, -0.84461308, 0.20759389],
        [0.45554739, 0.54289633, 0.81170881],
        [0.66091919, -0.16799635, -0.91037834]]]

  def testModelSaveLoad(self):
    actual = jax_energy_from_restored_state(self.test_species, self.test_coordinates)
    expected = torch_energy(self.test_species, self.test_coordinates)
    assert(jnp.allclose(
        actual, jnp.array(expected.tolist()), 
        atol=1e-4, rtol=1e-4))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelSaveLoadTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)