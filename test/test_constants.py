import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jaxani.constants import Constants

class ConstantsTest(unittest.TestCase):
  CONST_FILE = os.path.join(
    os.path.dirname(__file__), 'test_resources/rHCNOSFCl-5.1R_16-3.5A_a8-4.params')

  def setUp(self):
    self.consts = Constants(self.CONST_FILE)

  def testConstants(self):

    expected_attr_name_value = (
        ('Rcr', 5.1), 
        ('Rca', 3.5), 
        ('EtaR', jnp.array([19.7000])), 
        ('ShfR', jnp.array([0.8000, 1.0688, 1.3375, 1.6063, 1.8750, 2.1437, 2.4125, 2.6813, 2.9500,
            3.2188, 3.4875, 3.7562, 4.0250, 4.2937, 4.5625, 4.8313])), 
        ('EtaA', jnp.array([12.5000])), 
        ('Zeta', jnp.array([14.1000])), 
        ('ShfA', jnp.array([0.8000, 1.1375, 1.4750, 1.8125, 2.1500, 2.4875, 2.8250, 3.1625])), 
        ('ShfZ', jnp.array([0.3927, 1.1781, 1.9635, 2.7489])), 
        ('num_species', 7)
    )

    for i, e in zip(self.consts, expected_attr_name_value):
      assert(i == e[0])
      if isinstance(self.consts.get(i), jnp.ndarray):
        jnp.allclose(self.consts.get(i), e[1], rtol=0, atol=1e-4)
      else:
        assert(self.consts.get(i) == e[1])
        
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConstantsTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)