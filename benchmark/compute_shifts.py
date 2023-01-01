import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from jax import random
from jaxutil.timer import measure_time
from jaxani.aev import compute_shifts

class AniComputeShiftTest(unittest.TestCase):
    def setUp(self):
        key = random.PRNGKey(0)
        coord_random = jnp.abs(random.normal(key, (10, 3, 3))).tolist()
        cutoff_random = jnp.abs(random.normal(key, (10,))).tolist()

        self.params = [
            ([[10, 1, 2], [3, 10, 4], [5, 6, 10]], 10.0),
            ([[6, 1, 2], [3, 7.5, 4], [5, 6, 8.2]], 10.0),
            ([[6.4, 2.1, 2], [3.7, 7.5, 4.5], [5.5, 1.6, 8.2]], 8.0),
            (coord_random[0], cutoff_random[0]),
        ]
    
    @measure_time
    def testComputeShiftsMatch_withPbc(self):
        for test_cell, cut_off in self.params:
            jax_shifts = compute_shifts(
                jnp.array(test_cell),
                jnp.array([True, True, True]),
                cut_off).astype(jnp.int64).tolist()
    
    @measure_time
    def testComputeShiftsMatch_noPbc(self):
        for test_cell, cut_off in self.params:
            jax_shifts = compute_shifts(
                jnp.array(test_cell),
                jnp.array([False, False, False]),
                cut_off).astype(jnp.int64).tolist()

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)