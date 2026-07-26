"""Regression oracle for Codeberg Darknet's YOLO CIoU update.

The expected deltas were exported from the exact Darknet revision used by
YoloBattle (358b0da) with one 14x10 fine head, the LEGO anchors, one truth,
and a deterministic raw-output tensor.  Darknet stores delta as the negative
gradient applied to the preceding convolution.
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train import Yolo_loss


class DarknetBoxDeltaTest(unittest.TestCase):
    def test_ciou_delta_matches_codeberg_oracle(self):
        spec = {
            "stride": 16,
            "anchors": np.asarray([[8, 8], [10, 10], [15, 13], [45, 44], [68, 65], [77, 74]], dtype=np.float32),
            "mask": [0, 1, 2],
            "scale_x_y": 1.05,
            "ignore_thresh": 0.7,
            "iou_loss": "ciou",
            "iou_normalizer": 0.07,
            "obj_normalizer": 1.0,
            "cls_normalizer": 1.0,
        }
        # Darknet layout is anchor, entry, y, x; reshape produces the NCHW
        # tensor consumed by the fork without changing that order.
        raw = torch.tensor(
            [0.17 * math.sin(0.13 * index) for index in range(3 * 10 * 10 * 14)],
            dtype=torch.float32,
        ).reshape(1, 30, 10, 14).requires_grad_()
        labels = torch.zeros((1, 60, 5), dtype=torch.float32)
        labels[0, 0] = torch.tensor([67.0, 83.0, 77.0, 93.0, 0.0])

        loss = Yolo_loss([spec], n_classes=5, device="cpu", loss_mode="darknet")([raw], labels)[0]
        loss.backward()

        # Best anchor is output anchor 1, at (x=4, y=5).  This is the exact
        # Codeberg layer.delta vector, i.e. negative d(loss)/d(raw logits).
        expected_delta = torch.tensor(
            [-0.00094388997, -0.047561515, -2.2729468, -1.6228657], dtype=torch.float32,
        )
        actual_delta = -raw.grad[0, 10:14, 5, 4]
        torch.testing.assert_close(actual_delta, expected_delta, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
