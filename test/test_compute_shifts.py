import torch
import jax.numpy as jnp
import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch import Tensor
from torchani.aev import compute_shifts as compute_shifts_torch
from jaxani.aev import compute_shifts


class AniComputeShiftTest(unittest.TestCase):
    def setUp(self):
        self.params = [
            (((10, 12.0, 12.0), (13.0, 10, 14.0), (16.0, 18.0, 10)), 10.0),
            (((6, 12.0, 15.0), (16.0, 7.5, 19.0), (13.0, 7.2, 8.2)), 10.0),
            (((6, 12.0, 15.0), (16.0, 7.5, 19.0), (13.0, 7.2, 8.2)), 8.0),
        ]
    
    def testComputeShiftsMatch_withPbc(self):
      for test_cell, cut_off in self.params:
        torch_shifts = compute_shifts_torch(
            Tensor(test_cell),
            torch.ones(3, dtype=torch.bool),
            cut_off,
        ).tolist()

        jax_shifts = compute_shifts(
            test_cell,
            (True, True, True),
            cut_off).astype(jnp.int64).tolist()

        assert(len(jax_shifts) == len(torch_shifts))
        assert(sorted(jax_shifts) == sorted(torch_shifts))
      
    def testComputeShiftsMatch_noPbc(self):
      for test_cell, cut_off in self.params:
        torch_shifts = compute_shifts_torch(
            Tensor(test_cell),
            torch.zeros(3, dtype=torch.bool),
            cut_off,
        ).tolist()

        jax_shifts = compute_shifts(
            test_cell,
            (False, False, False),
            cut_off).astype(jnp.int64).tolist()

        assert(len(jax_shifts) == len(torch_shifts))
        assert(sorted(jax_shifts) == sorted(torch_shifts))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AniComputeShiftTest)
    runner = unittest.TextTestRunner()
    runner.run(suite)