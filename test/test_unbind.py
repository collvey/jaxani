import jax
import jax.numpy as jnp
import unittest

import sys
import os 
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(SCRIPT_DIR)
from jaxutil.jaxutil import jax_unbind

from jax import random
from torch import tensor

def torch_unbind(input, dim=0):
  return tensor(input.tolist()).unbind(dim)
  
class TestUnbind(unittest.TestCase):
  def setUp(self):
    key = random.PRNGKey(0)
    self.source_2d = random.normal(key, ((3, 8)))
    self.source_3d = random.normal(key, ((5, 6, 7)))

  def testUnbind_2d(self):
    actuals = jax_unbind(self.source_2d, dim=0)
    expects = torch_unbind(self.source_2d, dim=0)
    for (actual, expect) in zip(actuals, expects):
      assert(actual.shape == jnp.array(expect.tolist()).shape)
      assert(jnp.allclose(
          actual, jnp.array(expect.tolist()), 
          atol=1e-4, rtol=1e-4))

  def testUnbind_3d_dim1(self):
    actuals = jax_unbind(self.source_3d, dim=1)
    expects = torch_unbind(self.source_3d, dim=1)
    for (actual, expect) in zip(actuals, expects):
      assert(actual.shape == jnp.array(expect.tolist()).shape)
      assert(jnp.allclose(
          actual, jnp.array(expect.tolist()), 
          atol=1e-4, rtol=1e-4))
    
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestUnbind)
    runner = unittest.TextTestRunner()
    runner.run(suite)