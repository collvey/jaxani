import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.aev import AEVComputer
from jaxani.nn import SpeciesCoordinates
from jaxutil.timer import Timer

class AniAevBenchmark(unittest.TestCase):
    def setUp(self):
        test_species = [[0, 0, 1, 2, 0, 3]]
        test_coordinates = [[[1, 0, 6], 
                            [2, 0, 5], 
                            [3, 1, 4], 
                            [4, 0, 3], 
                            [5, 6, 2], 
                            [6, 0, 1]]]
        test_cell = [[5.2, 0, 0],
                    [0, 3.5, 0],
                    [0, 0, 5]]
        test_pbc = [True, True, True]

        jax_consts = {
            'Rcr': 5.1, 
            'Rca': 3.5, 
            'EtaR': jnp.array([19.7000]), 
            'ShfR': jnp.array([
                0.8000, 1.0688, 1.3375, 1.6063, 1.8750, 2.1437, 2.4125, 2.6813, 
                2.9500, 3.2188, 3.4875, 3.7562, 4.0250, 4.2937, 4.5625, 4.8313]), 
            'EtaA': jnp.array([12.5000]), 
            'Zeta': jnp.array([14.1000]), 
            'ShfA': jnp.array([
                0.8000, 1.1375, 1.4750, 1.8125, 2.1500, 2.4875, 2.8250, 3.1625]), 
            'ShfZ': jnp.array([0.3927, 1.1781, 1.9635, 2.7489]), 
            'num_species': 7,}
        self.jax_aev_computer = AEVComputer(**jax_consts)

        self.jax_species = jnp.array(test_species)
        self.jax_coordinates = jnp.array(test_coordinates)
        self.jax_species_coordinates = SpeciesCoordinates(
            species=self.jax_species, coordinates=self.jax_coordinates)
        self.jax_cell = jnp.array(test_cell)
        self.jax_pbc = jnp.array(test_pbc)

    def testAevRadialLength(self):
        with Timer('Aev Radial Length'):
            jax_radial_length = self.jax_aev_computer.radial_length
      
    def testAevCalculation_withPbc(self):
        with Timer('Aev Calculation with PBC'):
            jax_species, jax_aevs = self.jax_aev_computer.forward(
                self.jax_species_coordinates, cell=self.jax_cell, pbc=self.jax_pbc)
      
    def testAevCalculation_noPbc(self):
        with Timer('Aev Calculation with no PBC'):
            jax_species, jax_aevs = self.jax_aev_computer.forward(
                self.jax_species_coordinates)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AniAevBenchmark)
    runner = unittest.TextTestRunner()
    runner.run(suite)