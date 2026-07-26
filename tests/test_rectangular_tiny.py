"""Regression checks for rectangular YOLOv4-tiny training support."""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tool.darknet2pytorch import Darknet
from train import Yolo_loss


class RectangularTinyTest(unittest.TestCase):
    def test_forward_loss_and_backward(self):
        model = Darknet(os.path.join(ROOT, "cfg", "yolov4-tiny.cfg"), width=224, height=160)
        outputs = model(torch.zeros(1, 3, 160, 224))
        self.assertEqual([tuple(output.shape) for output in outputs], [(1, 255, 5, 7), (1, 255, 10, 14)])

        criterion = Yolo_loss.from_darknet(model, n_classes=80, device=torch.device("cpu"))
        labels = torch.zeros(1, 60, 5)
        labels[0, 0] = torch.tensor([20, 30, 80, 90, 1])
        loss = criterion(outputs, labels)[0]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
