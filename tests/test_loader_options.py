"""Regression checks for persistent DataLoader worker configuration."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train import data_loader_kwargs


class LoaderOptionsTest(unittest.TestCase):
    def test_workers_are_persistent_and_prefetched(self):
        options = data_loader_kwargs(4, torch.device("cuda"))
        self.assertEqual(options["num_workers"], 4)
        self.assertTrue(options["pin_memory"])
        self.assertTrue(options["persistent_workers"])
        self.assertEqual(options["prefetch_factor"], 2)

    def test_zero_workers_omits_worker_only_options(self):
        options = data_loader_kwargs(0, torch.device("cpu"))
        self.assertEqual(options, {"num_workers": 0, "pin_memory": False})


if __name__ == "__main__":
    unittest.main()
