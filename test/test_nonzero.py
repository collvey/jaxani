import jax.numpy as jnp
import unittest

import sys
import os 
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(SCRIPT_DIR)
from jaxutil.jaxutil import jax_nonzero

from jax import random
from torch import tensor


def torch_nonzero(input):
    return tensor(input.tolist()).nonzero()
  
class TestNonzero(unittest.TestCase):
  def setUp(self):
    key = random.PRNGKey(0)
    ks, kt = random.split(key)
    self.source1d = random.normal(ks, (42, ))
    self.target1d = random.normal(kt, (42, ))
    self.source = random.normal(ks, (42, 16))
    self.target = random.normal(kt, (42, 16))

  def testNonzero1d(self):
    actual = jax_nonzero(self.source1d <= self.target1d)
    expected = torch_nonzero(self.source1d <= self.target1d)
    assert(jnp.allclose(
        actual, jnp.array(expected.tolist()), 
        atol=1e-4, rtol=1e-4))

  def testNonzero2d(self):
    actual = jax_nonzero(self.source <= self.target)
    expected = torch_nonzero(self.source <= self.target)
    assert(jnp.allclose(
        actual, jnp.array(expected.tolist()), 
        atol=1e-4, rtol=1e-4))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNonzero)
    runner = unittest.TextTestRunner()
    runner.run(suite)