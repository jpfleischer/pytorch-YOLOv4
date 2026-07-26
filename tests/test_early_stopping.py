"""Unit tests for validation-driven PyTorch YOLOv4 early stopping."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train import EarlyStopping


class EarlyStoppingTest(unittest.TestCase):
    def test_stops_after_configured_non_improving_evaluations(self):
        stopping = EarlyStopping(patience=2, min_delta=0.001)
        self.assertEqual(stopping.update(0.500, step=100), (True, False))
        self.assertEqual(stopping.update(0.5005, step=200), (False, False))
        self.assertEqual(stopping.update(0.5009, step=300), (False, True))
        self.assertEqual(stopping.best_step, 100)

    def test_min_delta_and_disabled_stopping(self):
        stopping = EarlyStopping(patience=0, min_delta=0.001)
        self.assertEqual(stopping.update(0.500, step=100), (True, False))
        self.assertEqual(stopping.update(0.501, step=200), (False, False))
        self.assertEqual(stopping.update(0.5011, step=300), (True, False))
        self.assertEqual(stopping.best_step, 300)

    def test_state_round_trip(self):
        stopping = EarlyStopping(patience=3, min_delta=0.001)
        stopping.update(0.5, step=100)
        stopping.update(0.5, step=200)
        restored = EarlyStopping(patience=3, min_delta=0.001)
        restored.load_state_dict(stopping.state_dict())
        self.assertEqual(restored.best_metric, 0.5)
        self.assertEqual(restored.best_step, 100)
        self.assertEqual(restored.bad_evaluations, 1)


if __name__ == "__main__":
    unittest.main()
