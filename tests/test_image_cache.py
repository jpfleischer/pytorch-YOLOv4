"""Regression check for decoded source-image caching."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import Yolo_dataset


class ImageCacheTest(unittest.TestCase):
    def test_validation_item_uses_predecoded_rgb_image(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "image.jpg"
            bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), bgr))
            labels = root / "labels.txt"
            labels.write_text("image.jpg 0,0,1,1,0\n", encoding="utf-8")
            config = SimpleNamespace(mixup=0, letter_box=0, cache_images=1, dataset_dir=str(root))

            dataset = Yolo_dataset(str(labels), config, train=False)
            with patch("dataset.cv2.imread", side_effect=AssertionError("cache miss")):
                image, _ = dataset[0]

            self.assertTrue(np.array_equal(image, np.array([[[30, 20, 10]]], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
