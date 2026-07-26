"""Report target assignment and gradient health for one training batch.

This is a diagnostic for repaired/custom configurations.  It deliberately
does not update model weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfg import Cfg
from dataset import Yolo_dataset
from train import Yolo_loss, collate
from tool.darknet2pytorch import Darknet


def decoded_predictions(raw, criterion, output_id):
    batch, _, grid_h, grid_w = raw.shape
    channels = 5 + criterion.n_classes
    spec = criterion.output_specs[output_id]
    anchors_count = len(spec["mask"])
    output = raw.view(batch, anchors_count, channels, grid_h, grid_w).permute(0, 1, 3, 4, 2)
    output = output.clone()
    output[..., np.r_[:2, 4:channels]] = torch.sigmoid(output[..., np.r_[:2, 4:channels]])
    pred = output[..., :4].clone()
    y, x = torch.meshgrid(
        torch.arange(grid_h, device=raw.device),
        torch.arange(grid_w, device=raw.device),
        indexing="ij",
    )
    anchors = torch.as_tensor(spec["anchors"], device=raw.device).view(-1, 2)[spec["mask"]] / spec["stride"]
    scale_x_y = spec["scale_x_y"]
    pred[..., 0] = pred[..., 0] * scale_x_y - 0.5 * (scale_x_y - 1) + x
    pred[..., 1] = pred[..., 1] * scale_x_y - 0.5 * (scale_x_y - 1) + y
    pred[..., 2] = torch.exp(pred[..., 2]) * anchors[:, 0].view(1, -1, 1, 1)
    pred[..., 3] = torch.exp(pred[..., 3]) * anchors[:, 1].view(1, -1, 1, 1)
    return pred, grid_h, grid_w, channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--weights", help="optional .pth checkpoint to inspect")
    args = parser.parse_args()

    cfg = EasyDict(dict(Cfg))
    cfg.update(
        cfgfile=args.cfg, width=args.width, height=args.height, w=args.width, h=args.height,
        classes=args.classes, dataset_dir=args.data_dir, train_label=args.labels,
        mosaic=0, mixup=0, jitter=0, hue=0, saturation=1, exposure=1, flip=0,
    )
    dataset = Yolo_dataset(args.labels, cfg, train=True)
    if len(dataset) < args.batch:
        parser.error("--batch cannot exceed the number of labelled images")
    images, labels = collate([dataset[index] for index in range(args.batch)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.cfg, width=args.width, height=args.height).to(device).train()
    if args.weights:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    criterion = Yolo_loss.from_darknet(model, n_classes=args.classes, device=device)
    labels = labels.to(device)
    outputs = model(images.to(device))
    print(f"ground-truth boxes: {int((labels.sum(dim=2) > 0).sum())}")
    for output_id, raw in enumerate(outputs):
        pred, grid_h, grid_w, channels = decoded_predictions(raw, criterion, output_id)
        obj_mask, _, _, target = criterion.build_target(
            pred, labels, raw.shape[0], grid_h, grid_w, channels, output_id,
        )
        spec = criterion.output_specs[output_id]
        objectness = torch.sigmoid(raw.view(raw.shape[0], len(spec["mask"]), channels, grid_h, grid_w).permute(0, 1, 3, 4, 2)[..., 4])
        positive = target[..., 4].bool()
        print(
            f"head={output_id} positive-objectness={float(objectness[positive].mean()) if positive.any() else 0:.6f} "
            f"negative-objectness={float(objectness[~positive].mean()):.6f}"
        )
        print(
            f"head={output_id} stride={spec['stride']} mask={spec['mask']} "
            f"positives={int(target[..., 4].sum())} ignored={int((obj_mask == 0).sum())}"
        )
    loss, loss_xy, loss_wh, loss_obj, loss_cls, _ = criterion(outputs, labels)
    loss.backward()
    first_gradient = next(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    print(
        "loss=" + str(float(loss)) +
        f" xy={float(loss_xy)} wh={float(loss_wh)} obj={float(loss_obj)} cls={float(loss_cls)} "
        f"first-gradient-norm={float(first_gradient.norm())}"
    )


if __name__ == "__main__":
    main()
