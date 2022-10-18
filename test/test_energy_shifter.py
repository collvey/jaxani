import jax
import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.utils import EnergyShifter

# Enable support double precision
# See https://github.com/google/jax#current-gotchas
jax.config.update("jax_enable_x64", True)

class EnergyShifterTest(unittest.TestCase):
    def setUp(self):
        self.self_energies = [
            -0.5978583943827134,
            -38.08933878049795,
            -54.711968298621066,
            -75.19106774742086,
            -398.1577125334925,
            -99.80348506781634,
            -460.1681939421027]
        self.jax_species = jnp.array([[0, 0, 1, 2, 0, 3]])
        self.expected_energy = -169.785950009688

    def testJaxEnergyShifter(self):
        jax_energy_shifter = EnergyShifter(self.self_energies)
        assert(jnp.isclose(
            jax_energy_shifter.sae(self.jax_species), self.expected_energy))
        
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EnergyShifterTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)