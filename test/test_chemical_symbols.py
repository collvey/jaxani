import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.utils import ChemicalSymbolsToInts

class ChemicalSymbolsTest(unittest.TestCase):
  SPECIES = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']

  def setUp(self):
    self.chemical_symbols = ChemicalSymbolsToInts(self.SPECIES)

  def testChemicalSymbols(self):
    assert(self.chemical_symbols.rev_species == {
        'C': 1, 'Cl': 6, 'F': 5, 'H': 0, 'N': 2, 'O': 3, 'S': 4})
    jnp.allclose(
        self.chemical_symbols.forward(['H', 'C', 'F', 'H', 'C', 'N', 'O', 'Cl']), 
        jnp.array([0, 1, 5, 0, 1, 2, 3, 6]))
        
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChemicalSymbolsTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)