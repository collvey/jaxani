import jax
import jax.numpy as jnp
import unittest

import sys
import os 
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(SCRIPT_DIR)
from jaxutil.cacheutil import clear_caches
from jaxutil.jaxutil import jax_nonzero, jax_nonzero_torch
from jaxutil.timer import measure_time

from functools import partial
from jax import random
from torch import tensor

@partial(jax.jit, static_argnums=(1,))
def jit_nonzero(input, size):
    """Replace torch.Tensor.nonzero method with jax equivalent.
    """
    return jnp.stack(input.nonzero(size=size, fill_value=-1)).T

def torch_nonzero(input):
    return tensor(input.tolist()).nonzero()

class TestNonzero(unittest.TestCase):
    def setUp(self):
        clear_caches()
        DIM1 = 42
        DIM2 = 16
        key = random.PRNGKey(0)
        ks, kt = random.split(key)
        self.source1d = random.normal(ks, (DIM1, ))
        self.target1d = random.normal(kt, (DIM1, ))
        self.source = random.normal(ks, (DIM1, DIM2))
        self.target = random.normal(kt, (DIM1, DIM2))
        self.dim_1 = DIM1
        self.dim_2 = DIM2
        self.input = self.source1d <= self.target1d

    @measure_time
    def testNonzero1d_jax(self):
        actual = jax_nonzero(self.input)

    @measure_time
    def testNonzero2d_jax(self):
        actual = jax_nonzero(self.input)

    @measure_time
    def testNonzero1d_jit(self):
        actual = jit_nonzero(self.input, self.dim_1)

    @measure_time
    def testNonzero2d_jit(self):
        actual = jit_nonzero(self.input, self.dim_1*self.dim_2)

    @measure_time
    def testNonzero1d_torch(self):
        expected = torch_nonzero(self.input)

    @measure_time
    def testNonzero2d_torch(self):
        expected = torch_nonzero(self.input)

    @measure_time
    def testNonzero1d_jax_torch(self):
        expected = jax_nonzero_torch(self.input)

    @measure_time
    def testNonzero2d_jax_torch(self):
        expected = jax_nonzero_torch(self.input)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNonzero)
    runner = unittest.TextTestRunner()
    runner.run(suite)