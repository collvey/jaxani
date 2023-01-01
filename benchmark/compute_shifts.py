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

        self.params = [
            ([[10, 0, 0], [0, 10, 0], [0, 0, 10]], (10, 10, 10), 10.0),
            ([[6, 0, 0], [0, 7.5, 0], [0, 0, 8.2]], (6, 7.5, 8.2), 10.0),
            ([[6.4, 0, 0], [0, 7.5, 0], [0, 0, 8.2]], (6, 7.5, 8.2), 8.0),
        ]
    
    @measure_time
    def testComputeShiftsMatch_withPbc(self):
        for test_cell, jax_cell, cut_off in self.params:
            jax_shifts = compute_shifts(
                jax_cell,
                (True, True, True),
                cut_off).astype(jnp.int64).tolist()
    
    @measure_time
    def testComputeShiftsMatch_noPbc(self):
        for test_cell, jax_cell, cut_off in self.params:
            jax_shifts = compute_shifts(
                jax_cell,
                (False, False, False),
                cut_off).astype(jnp.int64).tolist()

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)