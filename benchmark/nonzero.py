import jax.numpy as jnp
import unittest

import sys
import os 
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(SCRIPT_DIR)
from jaxutil.jaxutil import jax_nonzero
from jaxutil.timer import measure_time

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

    @measure_time
    def testNonzero1d_jax(self):
        actual = jax_nonzero(self.source1d <= self.target1d)

    @measure_time
    def testNonzero2d_jax(self):
        actual = jax_nonzero(self.source <= self.target)

    @measure_time
    def testNonzero1d_torch(self):
        expected = torch_nonzero(self.source1d <= self.target1d)

    @measure_time
    def testNonzero2d_torch(self):
        expected = torch_nonzero(self.source <= self.target)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNonzero)
    runner = unittest.TextTestRunner()
    runner.run(suite)