import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.nn import SpeciesConverter, SpeciesCoordinates

class SpeciesConverterTest(unittest.TestCase):
  SPECIES = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']

  def setUp(self):
    self.species_converter = SpeciesConverter(self.SPECIES)

  def testSpeciesConverter_convArray(self):
    jnp.allclose(
        self.species_converter.conv_array[jnp.array([1, 1, 6, 7, 1, 8])],
        jnp.array([0, 0, 1, 2, 0, 3]))
  
  def testSpeciesConverter_directCall(self):
    actual_species, actual_coord = self.species_converter(
        SpeciesCoordinates(jnp.array([1, 1, 6, 7, 1, 8]), jnp.array([42, 0, 42])))
    expected_species, expected_coord = SpeciesCoordinates(
        species=jnp.array([0, 0, 1, 2, 0, 3]), coordinates=jnp.array([42, 0, 42]))

    assert(jnp.allclose(actual_species, expected_species, atol=1e-4, rtol=1e-4))
    assert(jnp.allclose(actual_coord, expected_coord, atol=1e-4, rtol=1e-4))
        
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpeciesConverterTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)