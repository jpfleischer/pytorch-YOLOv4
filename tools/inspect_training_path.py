"""Report target assignment and gradient health for one training batch.

This is a diagnostic for repaired/custom configurations.  It deliberately
does not update model weights.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfg import Cfg
from dataset import Yolo_dataset
from train import Yolo_loss, bboxes_iou, collate
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


def shape_iou(anchor, box):
    """Darknet's box_iou() for two origin-centred width/height boxes."""
    intersection = min(anchor[0], box[0]) * min(anchor[1], box[1])
    return intersection / (anchor[0] * anchor[1] + box[0] * box[1] - intersection)


def darknet_assignment_oracle(labels, criterion, outputs, width, height):
    """Reproduce Codeberg Darknet's primary target-assignment loop.

    This follows ``forward_yolo_layer()``: choose the best anchor globally,
    then assign it independently in every YOLO layer whose mask contains it.
    The supplied tiny cfg has no ``iou_thresh`` setting, so Darknet's default
    1.0 disables its optional additional-anchor assignment path.
    """
    anchors = criterion.output_specs[0]["anchors"]
    result = [set() for _ in outputs]
    attempted = [0 for _ in outputs]
    assigned_classes = [Counter() for _ in outputs]
    truth_classes = Counter()
    for batch_index, image_labels in enumerate(labels.cpu().numpy()):
        for x1, y1, x2, y2, _class_id in image_labels:
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(_class_id)
            truth_classes[class_id] += 1
            box = (float(x2 - x1), float(y2 - y1))
            best_anchor = max(range(len(anchors)), key=lambda n: shape_iou(anchors[n], box))
            center_x = (float(x1) + float(x2)) / 2
            center_y = (float(y1) + float(y2)) / 2
            for output_id, raw in enumerate(outputs):
                spec = criterion.output_specs[output_id]
                if best_anchor not in spec["mask"]:
                    continue
                grid_h, grid_w = raw.shape[-2:]
                cell_x = min(grid_w - 1, int(center_x * grid_w / width))
                cell_y = min(grid_h - 1, int(center_y * grid_h / height))
                masked_anchor = spec["mask"].index(best_anchor)
                attempted[output_id] += 1
                assigned_classes[output_id][class_id] += 1
                result[output_id].add((batch_index, masked_anchor, cell_y, cell_x))
    return result, attempted, truth_classes, assigned_classes


def ignored_predictions_oracle(pred, labels, stride, threshold):
    """Return Darknet's pre-positive-assignment objectness ignore candidates."""
    ignored = torch.zeros(pred.shape[:4], dtype=torch.bool, device=pred.device)
    for batch_index, image_labels in enumerate(labels):
        valid = (image_labels[:, 2] > image_labels[:, 0]) & (image_labels[:, 3] > image_labels[:, 1])
        image_labels = image_labels[valid]
        if not len(image_labels):
            continue
        truth = torch.empty((len(image_labels), 4), device=pred.device)
        truth[:, 0] = (image_labels[:, 0] + image_labels[:, 2]) / (2 * stride)
        truth[:, 1] = (image_labels[:, 1] + image_labels[:, 3]) / (2 * stride)
        truth[:, 2] = (image_labels[:, 2] - image_labels[:, 0]) / stride
        truth[:, 3] = (image_labels[:, 3] - image_labels[:, 1]) / stride
        best_iou = bboxes_iou(pred[batch_index].reshape(-1, 4), truth, xyxy=False).max(dim=1).values
        ignored[batch_index] = (best_iou > threshold).view(pred.shape[1:4])
    return ignored


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
    parser.add_argument("--anomaly", action="store_true", help="raise at the operation that creates a non-finite gradient")
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
    oracle, attempted, truth_classes, assigned_classes = darknet_assignment_oracle(
        labels, criterion, outputs, args.width, args.height,
    )
    print("ground-truth by class: " + ", ".join(f"{key}={value}" for key, value in sorted(truth_classes.items())))
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
        pytorch_slots = {tuple(index.tolist()) for index in target[..., 4].nonzero()}
        verdict = "MATCH" if pytorch_slots == oracle[output_id] else "MISMATCH"
        print(
            f"head={output_id} Codeberg-Darknet oracle attempted={attempted[output_id]} "
            f"unique-slots={len(oracle[output_id])} PyTorch={len(pytorch_slots)} {verdict}; "
            "assigned classes=" + ", ".join(
                f"{key}={value}" for key, value in sorted(assigned_classes[output_id].items())
            )
        )
        # Darknet applies this test to every head/image before overwriting
        # assigned positive slots.
        darknet_ignore = ignored_predictions_oracle(pred, labels, spec["stride"], threshold=0.7) & ~positive
        pytorch_ignore = obj_mask == 0
        threshold_only = (
            ignored_predictions_oracle(pred, labels, spec["stride"], threshold=0.5)
            & ~ignored_predictions_oracle(pred, labels, spec["stride"], threshold=0.7)
            & ~positive
        )
        print(
            f"head={output_id} ignore-mask Codeberg(.7)={int(darknet_ignore.sum())} "
            f"PyTorch(cfg {spec['ignore_thresh']:g})={int(pytorch_ignore.sum())} "
            f"legacy-threshold-only(.5-.7]={int(threshold_only.sum())} "
            f"current-vs-Codeberg-difference={int((pytorch_ignore ^ darknet_ignore).sum())}"
        )
    loss, loss_xy, loss_wh, loss_obj, loss_cls, _ = criterion(outputs, labels)
    if args.anomaly:
        with torch.autograd.detect_anomaly():
            loss.backward()
    else:
        loss.backward()
    first_gradient = next(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    print(
        "loss=" + str(float(loss)) +
        f" xy={float(loss_xy)} wh={float(loss_wh)} obj={float(loss_obj)} cls={float(loss_cls)} "
        f"first-gradient-norm={float(first_gradient.norm())}"
    )


if __name__ == "__main__":
    main()
