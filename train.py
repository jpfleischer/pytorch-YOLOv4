# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import json
import logging
import os, sys, math
import argparse
from collections import deque, defaultdict
from contextlib import nullcontext
import datetime
import random

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet
from tool.utils import nms_cpu

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


def darknet_dx_box_iou(pred, truth, raw_box, iou_normalizer, iou_loss):
    """Return Codeberg Darknet's direct YOLO box delta.

    ``pred`` and ``truth`` are centre-x/centre-y/width/height boxes in
    Darknet's normalized network coordinates.  The result is the *negative*
    raw-logit gradient written by Darknet's ``delta_yolo_box()``.  This is
    deliberately not an autograd derivative of a conventional CIoU scalar:
    Darknet's hand-derived ``dx_box_iou()`` has distinct coordinate scaling
    and width/height terms, so exact training compatibility requires it.
    """
    if iou_loss not in {'iou', 'giou', 'diou', 'ciou'}:
        raise ValueError(f"Darknet loss mode does not support iou_loss={iou_loss!r}; use --loss-mode legacy.")

    px, py, pw, ph = pred.unbind(dim=1)
    tx, ty, tw, th = truth.unbind(dim=1)
    pred_t, pred_b = py - ph / 2, py + ph / 2
    pred_l, pred_r = px - pw / 2, px + pw / 2
    truth_t, truth_b = ty - th / 2, ty + th / 2
    truth_l, truth_r = tx - tw / 2, tx + tw / 2

    # Keep the same min/max branches and intermediate expressions as
    # Darknet src-lib/box.cpp:dx_box_iou().  Positive predicted sizes make
    # the min/max orientation normal in YOLO, but preserving it here avoids
    # changing behaviour for unusual raw outputs.
    top, bottom = torch.minimum(pred_t, pred_b), torch.maximum(pred_t, pred_b)
    left, right = torch.minimum(pred_l, pred_r), torch.maximum(pred_l, pred_r)
    area = (bottom - top) * (right - left)
    truth_area = (truth_b - truth_t) * (truth_r - truth_l)
    intersection_h = torch.minimum(bottom, truth_b) - torch.maximum(top, truth_t)
    intersection_w = torch.minimum(right, truth_r) - torch.maximum(left, truth_l)
    intersection = intersection_w * intersection_h
    union = area + truth_area - intersection
    distance = (px - tx).square() + (py - ty).square()
    giou_width = torch.maximum(right, truth_r) - torch.minimum(left, truth_l)
    giou_height = torch.maximum(bottom, truth_b) - torch.minimum(top, truth_t)
    giou_area = giou_width * giou_height

    darea_t = -(right - left)
    darea_b = right - left
    darea_l = -(bottom - top)
    darea_r = bottom - top
    dinter_t = torch.where(top > truth_t, -intersection_w, torch.zeros_like(top))
    dinter_b = torch.where(bottom < truth_b, intersection_w, torch.zeros_like(top))
    dinter_l = torch.where(left > truth_l, -intersection_h, torch.zeros_like(top))
    dinter_r = torch.where(right < truth_r, intersection_h, torch.zeros_like(top))
    dunion_t, dunion_b = darea_t - dinter_t, darea_b - dinter_b
    dunion_l, dunion_r = darea_l - dinter_l, darea_r - dinter_r
    dgiou_t = torch.where(top < truth_t, -giou_width, torch.zeros_like(top))
    dgiou_b = torch.where(bottom > truth_b, giou_width, torch.zeros_like(top))
    dgiou_l = torch.where(left < truth_l, -giou_height, torch.zeros_like(top))
    dgiou_r = torch.where(right > truth_r, giou_height, torch.zeros_like(top))

    safe_union_sq = union.square().clamp_min(1e-30)
    valid_union = union > 0
    def iou_part(dinter, dunion):
        value = (union * dinter - intersection * dunion) / safe_union_sq
        return torch.where(valid_union, value, torch.zeros_like(value))

    p_t, p_b = iou_part(dinter_t, dunion_t), iou_part(dinter_b, dunion_b)
    p_l, p_r = iou_part(dinter_l, dunion_l), iou_part(dinter_r, dunion_r)
    normal_height = pred_t < pred_b
    normal_width = pred_l < pred_r
    p_t = torch.where(normal_height, p_t, p_b)
    p_b = torch.where(normal_height, p_b, p_t)
    p_l = torch.where(normal_width, p_l, p_r)
    p_r = torch.where(normal_width, p_r, p_l)

    enclosure_t = torch.minimum(py - ph / 2, ty - th / 2)
    enclosure_b = torch.maximum(py + ph / 2, ty + th / 2)
    enclosure_l = torch.minimum(px - pw / 2, tx - tw / 2)
    enclosure_r = torch.maximum(px + pw / 2, tx + tw / 2)
    enclosure_w, enclosure_h = enclosure_r - enclosure_l, enclosure_b - enclosure_t
    enclosure_diag = enclosure_w.square() + enclosure_h.square()
    dct_dy = torch.where(top < truth_t, torch.ones_like(top), torch.zeros_like(top))
    dct_dh = torch.where(top < truth_t, torch.full_like(top, -0.5), torch.zeros_like(top))
    dcb_dy = torch.where(bottom > truth_b, torch.ones_like(top), torch.zeros_like(top))
    dcb_dh = torch.where(bottom > truth_b, torch.full_like(top, 0.5), torch.zeros_like(top))
    dcl_dx = torch.where(left < truth_l, torch.ones_like(top), torch.zeros_like(top))
    dcl_dw = torch.where(left < truth_l, torch.full_like(top, -0.5), torch.zeros_like(top))
    dcr_dx = torch.where(right > truth_r, torch.ones_like(top), torch.zeros_like(top))
    dcr_dw = torch.where(right > truth_r, torch.full_like(top, 0.5), torch.zeros_like(top))
    dcw_dx, dcw_dy, dcw_dw, dcw_dh = dcr_dx - dcl_dx, torch.zeros_like(top), dcr_dw - dcl_dw, torch.zeros_like(top)
    dch_dx, dch_dy, dch_dw, dch_dh = torch.zeros_like(top), dcb_dy - dct_dy, torch.zeros_like(top), dcb_dh - dct_dh

    p_dx, p_dy = p_l + p_r, p_t + p_b
    # This intentionally follows Darknet's (non-half) width/height terms.
    p_dw, p_dh = p_r - p_l, p_b - p_t
    safe_enclosure_sq = enclosure_diag.square().clamp_min(1e-30)
    valid_enclosure = enclosure_diag > 0
    def distance_term(dx, dy, dw, dh):
        dc = 2 * enclosure_w * dx + 2 * enclosure_h * dy
        return (2 * (tx - px) * enclosure_diag - dc * distance) / safe_enclosure_sq, \
            (2 * (ty - py) * enclosure_diag - dc * distance) / safe_enclosure_sq, \
            (2 * enclosure_w * dw + 2 * enclosure_h * dh) * distance / safe_enclosure_sq

    diou_dx, _, diou_dw = distance_term(dcw_dx, dch_dx, dcw_dw, dch_dw)
    _, diou_dy, diou_dh = distance_term(dcw_dy, dch_dy, dcw_dh, dch_dh)
    no_overlap = (intersection_w <= 0) | (intersection_h <= 0)
    if iou_loss == 'giou':
        safe_giou_sq = giou_area.square().clamp_min(1e-30)
        def giou_part(dunion, dgiou):
            return (giou_area * dunion - union * dgiou) / safe_giou_sq
        g_t, g_b = giou_part(dunion_t, dgiou_t), giou_part(dunion_b, dgiou_b)
        g_l, g_r = giou_part(dunion_l, dgiou_l), giou_part(dunion_r, dgiou_r)
        p_t, p_b = p_t + g_t, p_b + g_b
        p_l, p_r = p_l + g_l, p_r + g_r
        p_t, p_b = torch.where(no_overlap, g_t, p_t), torch.where(no_overlap, g_b, p_b)
        p_l, p_r = torch.where(no_overlap, g_l, p_l), torch.where(no_overlap, g_r, p_r)
        p_dx, p_dy, p_dw, p_dh = p_l + p_r, p_t + p_b, p_r - p_l, p_b - p_t
    elif iou_loss in {'diou', 'ciou'}:
        diou_dx, diou_dy, diou_dw, diou_dh = (
            torch.where(valid_enclosure, diou_dx, torch.zeros_like(diou_dx)),
            torch.where(valid_enclosure, diou_dy, torch.zeros_like(diou_dy)),
            torch.where(valid_enclosure, diou_dw, torch.zeros_like(diou_dw)),
            torch.where(valid_enclosure, diou_dh, torch.zeros_like(diou_dh)),
        )
        if iou_loss == 'ciou':
            aspect = 4 / (math.pi * math.pi) * (torch.atan(tw / th) - torch.atan(pw / ph)).square()
            alpha = aspect / (1 - intersection / union + aspect + 1e-6)
            aspect_dw = 8 / (math.pi * math.pi) * (torch.atan(tw / th) - torch.atan(pw / ph)) * ph
            aspect_dh = -8 / (math.pi * math.pi) * (torch.atan(tw / th) - torch.atan(pw / ph)) * pw
            diou_dw = diou_dw + alpha * aspect_dw
            diou_dh = diou_dh + alpha * aspect_dh
        p_dx, p_dy, p_dw, p_dh = p_dx + diou_dx, p_dy + diou_dy, p_dw + diou_dw, p_dh + diou_dh
        p_dx, p_dy = torch.where(no_overlap, diou_dx, p_dx), torch.where(no_overlap, diou_dy, p_dy)
        p_dw, p_dh = torch.where(no_overlap, diou_dw, p_dw), torch.where(no_overlap, diou_dh, p_dh)

    # Darknet applies exp() only to its width/height raw-logit deltas; x/y
    # intentionally bypass sigmoid/scale_x_y derivatives.
    delta = torch.stack((p_dx, p_dy, p_dw * torch.exp(raw_box[:, 2]), p_dh * torch.exp(raw_box[:, 3])), dim=1)
    return delta * iou_normalizer


class Yolo_loss(nn.Module):
    """YOLO loss derived from the selected Darknet cfg.

    The original implementation baked in three 608x608 YOLOv4 heads.  Building
    the tensors from each output fixes rectangular inputs and YOLOv4-tiny's two
    detection heads.
    """
    def __init__(self, output_specs, n_classes, device=None, loss_mode='darknet'):
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.output_specs = output_specs
        if loss_mode not in {'darknet', 'legacy'}:
            raise ValueError(f'Unsupported loss mode: {loss_mode}')
        self.loss_mode = loss_mode
        self.profile_enabled = False
        self.profile_phases = defaultdict(float)

    def _profile_start(self):
        if not self.profile_enabled:
            return None
        _synchronize(self.device)
        return time.perf_counter()

    def _profile_end(self, phase, start):
        if start is None:
            return
        _synchronize(self.device)
        self.profile_phases[phase] += time.perf_counter() - start

    @classmethod
    def from_darknet(cls, model, n_classes, device=None, loss_mode='darknet'):
        specs = []
        for block, layer in zip(model.blocks[1:], model.models):
            if block['type'] == 'yolo':
                anchors = np.asarray(layer.anchors, dtype=np.float32).reshape(-1, 2)
                specs.append({
                    'stride': int(layer.stride),
                    'anchors': anchors,
                    'mask': list(layer.anchor_mask),
                    'scale_x_y': float(layer.scale_x_y),
                    # Match Darknet's per-[yolo] setting rather than using a
                    # global hard-coded threshold.  Darknet defaults to .5
                    # when the cfg omits it.
                    'ignore_thresh': float(block.get('ignore_thresh', 0.5)),
                    'iou_loss': block.get('iou_loss', 'mse').lower(),
                    'iou_normalizer': float(block.get('iou_normalizer', 0.75)),
                    'obj_normalizer': float(block.get('obj_normalizer', 1.0)),
                    'cls_normalizer': float(block.get('cls_normalizer', 1.0)),
                })
        if not specs:
            raise ValueError('The selected cfg contains no [yolo] detection layers.')
        return cls(specs, n_classes=n_classes, device=device, loss_mode=loss_mode)

    def build_target_reference(self, pred, labels, batchsize, grid_h, grid_w, n_ch, output_id):
        # target assignment
        spec = self.output_specs[output_id]
        stride = spec['stride']
        all_anchors = torch.as_tensor(spec['anchors'] / stride, device=self.device)
        masked_anchors = all_anchors[spec['mask']]
        n_anchors = len(spec['mask'])
        tgt_mask = torch.zeros(batchsize, n_anchors, grid_h, grid_w, 4 + self.n_classes, device=self.device)
        obj_mask = torch.ones(batchsize, n_anchors, grid_h, grid_w, device=self.device)
        tgt_scale = torch.zeros(batchsize, n_anchors, grid_h, grid_w, 2, device=self.device)
        target = torch.zeros(batchsize, n_anchors, grid_h, grid_w, n_ch, device=self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (stride * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (stride * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / stride
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / stride
        truth_i_all = truth_x_all.to(torch.long).clamp_(0, grid_w - 1).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.long).clamp_(0, grid_h - 1).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # Match on width/height IoU.  Detection-anchor assignment has no
            # meaningful centre coordinates, so CIoU here is incorrect.
            truth_wh = truth_box[:, 2:]
            inter = torch.minimum(truth_wh[:, None], all_anchors[None]).prod(dim=2)
            union = truth_wh.prod(dim=1, keepdim=True) + all_anchors.prod(dim=1) - inter
            best_n_all = (inter / (union + 1e-16)).argmax(dim=1)
            mask_to_output = {anchor_id: index for index, anchor_id in enumerate(spec['mask'])}
            best_n_mask = torch.tensor(
                [int(anchor) in mask_to_output for anchor in best_n_all.tolist()], device=self.device, dtype=torch.bool
            )

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > spec['ignore_thresh'])
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            # Darknet applies the ignore rule above to every head, including
            # heads that have no positive assignment for this image.
            if not best_n_mask.any():
                continue

            for ti in range(best_n_all.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = mask_to_output[int(best_n_all[ti])]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    scale_x_y = spec['scale_x_y']
                    target[b, a, j, i, 0] = (
                        truth_x_all[b, ti] - torch.floor(truth_x_all[b, ti]) + 0.5 * (scale_x_y - 1)
                    ) / scale_x_y
                    target[b, a, j, i, 1] = (
                        truth_y_all[b, ti] - torch.floor(truth_y_all[b, ti]) + 0.5 * (scale_x_y - 1)
                    ) / scale_x_y
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / masked_anchors[a, 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / masked_anchors[a, 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / (grid_w * grid_h))
        return obj_mask, tgt_mask, tgt_scale, target

    def build_target(self, pred, labels, batchsize, grid_h, grid_w, n_ch, output_id):
        """GPU-batched equivalent of ``build_target_reference``.

        This eliminates the per-image Python loops and CPU round trips in the
        original target assignment while retaining Darknet's anchor matching
        and ignore-mask rules.
        """
        spec = self.output_specs[output_id]
        stride = spec['stride']
        all_anchors = torch.as_tensor(spec['anchors'] / stride, device=self.device)
        n_anchors = len(spec['mask'])
        tgt_mask = torch.zeros(batchsize, n_anchors, grid_h, grid_w, 4 + self.n_classes, device=self.device)
        obj_mask = torch.ones(batchsize, n_anchors, grid_h, grid_w, device=self.device)
        tgt_scale = torch.zeros(batchsize, n_anchors, grid_h, grid_w, 2, device=self.device)
        target = torch.zeros(batchsize, n_anchors, grid_h, grid_w, n_ch, device=self.device)

        valid = labels.sum(dim=2) > 0
        truth_x = ((labels[:, :, 2] + labels[:, :, 0]) / (stride * 2)).to(pred.dtype)
        truth_y = ((labels[:, :, 3] + labels[:, :, 1]) / (stride * 2)).to(pred.dtype)
        truth_w = ((labels[:, :, 2] - labels[:, :, 0]) / stride).to(pred.dtype)
        truth_h = ((labels[:, :, 3] - labels[:, :, 1]) / stride).to(pred.dtype)
        truth_boxes = torch.stack((truth_x, truth_y, truth_w, truth_h), dim=-1)

        # Every prediction whose IoU with any ground truth exceeds
        # ignore_thresh is excluded from objectness loss, as in Darknet.
        pred_boxes = pred.reshape(batchsize, -1, 4)
        pred_tl = pred_boxes[..., :2].unsqueeze(2) - pred_boxes[..., 2:].unsqueeze(2) / 2
        pred_br = pred_boxes[..., :2].unsqueeze(2) + pred_boxes[..., 2:].unsqueeze(2) / 2
        truth_tl = truth_boxes[..., :2].unsqueeze(1) - truth_boxes[..., 2:].unsqueeze(1) / 2
        truth_br = truth_boxes[..., :2].unsqueeze(1) + truth_boxes[..., 2:].unsqueeze(1) / 2
        intersection_wh = (torch.minimum(pred_br, truth_br) - torch.maximum(pred_tl, truth_tl)).clamp_min(0)
        intersection = intersection_wh.prod(dim=-1)
        pred_area = pred_boxes[..., 2:].prod(dim=-1).unsqueeze(2)
        truth_area = truth_boxes[..., 2:].prod(dim=-1).unsqueeze(1)
        ious = intersection / (pred_area + truth_area - intersection).clamp_min(1e-16)
        ious = torch.where(valid.unsqueeze(1), ious, torch.zeros_like(ious))
        obj_mask = ~(ious.amax(dim=2) > spec['ignore_thresh'])
        obj_mask = obj_mask.view(batchsize, n_anchors, grid_h, grid_w)

        # Assign each ground truth to its best global anchor, then select the
        # assignments whose anchor belongs to this output head.
        truth_wh = truth_boxes[..., 2:]
        anchor_intersection = torch.minimum(truth_wh.unsqueeze(2), all_anchors.view(1, 1, -1, 2)).prod(dim=-1)
        anchor_union = truth_wh.prod(dim=-1, keepdim=True) + all_anchors.prod(dim=-1).view(1, 1, -1) - anchor_intersection
        best_anchor = (anchor_intersection / (anchor_union + 1e-16)).argmax(dim=2)
        anchor_to_head = torch.full((len(all_anchors),), -1, device=self.device, dtype=torch.long)
        head_mask = torch.as_tensor(spec['mask'], device=self.device, dtype=torch.long)
        anchor_to_head[head_mask] = torch.arange(n_anchors, device=self.device)
        head_anchor = anchor_to_head[best_anchor]
        positive = valid & (head_anchor >= 0)
        batch_index, truth_index = positive.nonzero(as_tuple=True)
        if batch_index.numel() == 0:
            return obj_mask, tgt_mask, tgt_scale, target

        anchor_index = head_anchor[batch_index, truth_index]
        grid_x = truth_x[batch_index, truth_index].long().clamp_(0, grid_w - 1)
        grid_y = truth_y[batch_index, truth_index].long().clamp_(0, grid_h - 1)
        selected_w = truth_w[batch_index, truth_index]
        selected_h = truth_h[batch_index, truth_index]
        scale_x_y = spec['scale_x_y']
        masked_anchors = all_anchors[head_mask]

        obj_mask[batch_index, anchor_index, grid_y, grid_x] = 1
        tgt_mask[batch_index, anchor_index, grid_y, grid_x, :] = 1
        target[batch_index, anchor_index, grid_y, grid_x, 0] = (
            truth_x[batch_index, truth_index] - torch.floor(truth_x[batch_index, truth_index])
            + 0.5 * (scale_x_y - 1)
        ) / scale_x_y
        target[batch_index, anchor_index, grid_y, grid_x, 1] = (
            truth_y[batch_index, truth_index] - torch.floor(truth_y[batch_index, truth_index])
            + 0.5 * (scale_x_y - 1)
        ) / scale_x_y
        target[batch_index, anchor_index, grid_y, grid_x, 2] = torch.log(
            selected_w / masked_anchors[anchor_index, 0] + 1e-16
        )
        target[batch_index, anchor_index, grid_y, grid_x, 3] = torch.log(
            selected_h / masked_anchors[anchor_index, 1] + 1e-16
        )
        target[batch_index, anchor_index, grid_y, grid_x, 4] = 1
        class_index = labels[batch_index, truth_index, 4].long()
        target[batch_index, anchor_index, grid_y, grid_x, 5 + class_index] = 1
        scale = torch.sqrt(2 - selected_w * selected_h / (grid_w * grid_h))
        tgt_scale[batch_index, anchor_index, grid_y, grid_x, :] = scale.unsqueeze(1)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        # Keep every reported component as a tensor.  Darknet CIoU mode does
        # not use the historical ``loss_wh``/``loss_l2`` terms, but callers
        # still log them every training step.
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = (
            xin[0].new_zeros(()) for _ in range(6)
        )
        self.profile_phases.clear()
        for output_id, output in enumerate(xin):
            spec = self.output_specs[output_id]
            batchsize = output.shape[0]
            grid_h, grid_w = output.shape[2:]
            n_ch = 5 + self.n_classes
            n_anchors = len(spec['mask'])
            expected_channels = n_anchors * n_ch
            if output.shape[1] != expected_channels:
                raise ValueError(
                    f'YOLO head {output_id} has {output.shape[1]} channels, expected {expected_channels}. '
                    'Update each detection convolution filters value to anchors * (classes + 5).'
                )

            phase_start = self._profile_start()
            output = output.view(batchsize, n_anchors, n_ch, grid_h, grid_w)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()
            box_logits = output[..., :4]

            # Keep the logits for objectness/classification loss.  Applying
            # sigmoid first and feeding the result to BCELoss can underflow to
            # exact zero after a few negative-heavy updates, at which point
            # the positive-object gradient is lost permanently.
            obj_logits = output[..., 4].clone()
            cls_logits = output[..., 5:].clone()

            # logistic activation for xy, obj, cls
            output = output.clone()
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            y, x = torch.meshgrid(torch.arange(grid_h, device=output.device), torch.arange(grid_w, device=output.device), indexing='ij')
            anchors = torch.as_tensor(spec['anchors'], device=output.device).view(-1, 2)[spec['mask']] / spec['stride']
            scale_x_y = spec['scale_x_y']
            # Assignment and ignore-mask comparisons are non-differentiable.
            # Detach them so extreme logits in an unselected cell cannot make
            # an ExpBackward(0 * inf) NaN during CIoU training.
            with torch.no_grad():
                assignment_pred = output[..., :4].clone()
                assignment_pred[..., 0] = assignment_pred[..., 0] * scale_x_y - 0.5 * (scale_x_y - 1) + x
                assignment_pred[..., 1] = assignment_pred[..., 1] * scale_x_y - 0.5 * (scale_x_y - 1) + y
                assignment_pred[..., 2] = torch.exp(assignment_pred[..., 2]) * anchors[:, 0].view(1, -1, 1, 1)
                assignment_pred[..., 3] = torch.exp(assignment_pred[..., 3]) * anchors[:, 1].view(1, -1, 1, 1)
            self._profile_end('loss_prepare_and_assignment_pred_s', phase_start)

            phase_start = self._profile_start()
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                assignment_pred, labels, batchsize, grid_h, grid_w, n_ch, output_id,
            )
            self._profile_end('loss_target_assignment_s', phase_start)

            phase_start = self._profile_start()
            active_obj = obj_mask.bool()
            positive_cls = target[..., 4].bool()

            if self.loss_mode == 'darknet':
                # Darknet's objectness delta is target - sigmoid(logit) for
                # every non-ignored prediction: ordinary, unbalanced BCE.
                loss_obj += spec['obj_normalizer'] * F.binary_cross_entropy_with_logits(
                    obj_logits[active_obj], target[..., 4][active_obj], reduction='sum',
                )
                if positive_cls.any():
                    loss_cls += spec['cls_normalizer'] * F.binary_cross_entropy_with_logits(
                        cls_logits[positive_cls], target[..., 5:][positive_cls], reduction='sum',
                    )

                    # Decode a conventional quality value for logging, then
                    # inject Darknet's exact hand-written box gradient below.
                    grid = torch.stack((x, y), dim=-1).view(1, 1, grid_h, grid_w, 2)
                    grid = grid.expand(batchsize, n_anchors, -1, -1, -1)[positive_cls]
                    positive_anchors = anchors.view(1, n_anchors, 1, 1, 2).expand(
                        batchsize, -1, grid_h, grid_w, -1,
                    )[positive_cls]
                    positive_output = output[positive_cls]
                    positive_target = target[positive_cls]
                    pred_box = torch.empty_like(positive_output[:, :4])
                    pred_box[:, :2] = positive_output[:, :2] * scale_x_y - 0.5 * (scale_x_y - 1) + grid
                    pred_box[:, 2:4] = torch.exp(positive_output[:, 2:4]) * positive_anchors
                    target_box = torch.empty_like(pred_box)
                    target_box[:, :2] = positive_target[:, :2] * scale_x_y - 0.5 * (scale_x_y - 1) + grid
                    target_box[:, 2:4] = torch.exp(positive_target[:, 2:4]) * positive_anchors
                    kind = spec['iou_loss']
                    if kind == 'ciou':
                        quality = bboxes_iou(pred_box, target_box, xyxy=False, CIoU=True).diagonal()
                    elif kind == 'diou':
                        quality = bboxes_iou(pred_box, target_box, xyxy=False, DIoU=True).diagonal()
                    elif kind == 'giou':
                        quality = bboxes_iou(pred_box, target_box, xyxy=False, GIoU=True).diagonal()
                    elif kind == 'iou':
                        quality = bboxes_iou(pred_box, target_box, xyxy=False).diagonal()
                    else:
                        raise ValueError(f"Darknet loss mode does not support iou_loss={kind!r}; use --loss-mode legacy.")
                    reported_box_loss = spec['iou_normalizer'] * (1 - quality).sum()
                    normalized_scale = pred_box.new_tensor((grid_w, grid_h, grid_w, grid_h))
                    with torch.no_grad():
                        darknet_delta = darknet_dx_box_iou(
                            pred_box / normalized_scale,
                            target_box / normalized_scale,
                            box_logits[positive_cls],
                            spec['iou_normalizer'],
                            kind,
                        )
                    # Darknet's layer.delta is the negative gradient passed
                    # directly into the preceding convolution.  Preserve the
                    # useful scalar loss for logs while replacing only its
                    # derivative with that exact delta.
                    direct_gradient = (-darknet_delta.detach() * box_logits[positive_cls]).sum()
                    loss_xy += reported_box_loss.detach() + direct_gradient - direct_gradient.detach()
            else:
                # Kept for reproducing prior fork experiments.  This is not
                # equivalent to a cfg with ``iou_loss=ciou``.
                legacy_output = output.clone()
                legacy_target = target.clone()
                legacy_output[..., 4] *= obj_mask
                legacy_output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
                legacy_output[..., 2:4] *= tgt_scale
                legacy_target[..., 4] *= obj_mask
                legacy_target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
                legacy_target[..., 2:4] *= tgt_scale
                loss_xy += F.binary_cross_entropy(
                    input=legacy_output[..., :2], target=legacy_target[..., :2],
                    weight=tgt_scale * tgt_scale, reduction='sum',
                )
                loss_wh += F.mse_loss(input=legacy_output[..., 2:4], target=legacy_target[..., 2:4], reduction='sum') / 2
                positive_obj = active_obj & target[..., 4].bool()
                negative_obj = active_obj & ~target[..., 4].bool()
                if positive_obj.any():
                    loss_obj += F.binary_cross_entropy_with_logits(
                        obj_logits[positive_obj], target[..., 4][positive_obj], reduction='sum',
                    )
                if negative_obj.any():
                    negative_loss = F.binary_cross_entropy_with_logits(
                        obj_logits[negative_obj], target[..., 4][negative_obj], reduction='sum',
                    )
                    loss_obj += negative_loss * (positive_obj.sum() / negative_obj.sum()).to(negative_loss.dtype)
                if positive_cls.any():
                    loss_cls += F.binary_cross_entropy_with_logits(
                        cls_logits[positive_cls], target[..., 5:][positive_cls], reduction='sum',
                    )
                loss_l2 += F.mse_loss(input=legacy_output, target=legacy_target, reduction='sum')
            self._profile_end('loss_objective_and_gradient_s', phase_start)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    collate_start = time.perf_counter()
    images = []
    bboxes = []
    timings = defaultdict(float)
    has_timings = len(batch[0]) == 3
    for sample in batch:
        img, box = sample[:2]
        images.append([img])
        bboxes.append([box])
        if has_timings:
            for phase, seconds in sample[2].items():
                timings[phase] += seconds
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    if has_timings:
        timings['collate_s'] = time.perf_counter() - collate_start
        return images, bboxes, dict(timings)
    return images, bboxes


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _limit_dataset(dataset, max_samples):
    """Keep a deterministic prefix of a Yolo_dataset for diagnostics."""
    if not max_samples:
        return dataset
    dataset.imgs = dataset.imgs[:max_samples]
    dataset.truth = {image: dataset.truth[image] for image in dataset.imgs}
    return dataset


def data_loader_kwargs(workers, device):
    """Use persistent loader processes when a run spans many short epochs."""
    kwargs = {
        'num_workers': workers,
        'pin_memory': device.type == 'cuda',
    }
    if workers > 0:
        # LEGO's batch-64 setup has one batch per epoch.  Without this,
        # PyTorch tears down and respawns all workers thousands of times.
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return kwargs


class EarlyStopping:
    """Track validation AP50-95 and optionally stop after a plateau."""

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float('-inf')
        self.best_step = 0
        self.bad_evaluations = 0

    def update(self, metric, step):
        improved = math.isfinite(metric) and metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = metric
            self.best_step = step
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        should_stop = self.patience > 0 and self.bad_evaluations >= self.patience
        return improved, should_stop

    def state_dict(self):
        return {
            'best_metric': self.best_metric,
            'best_step': self.best_step,
            'bad_evaluations': self.bad_evaluations,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.best_metric = float(state.get('best_metric', self.best_metric))
        self.best_step = int(state.get('best_step', self.best_step))
        self.bad_evaluations = int(state.get('bad_evaluations', self.bad_evaluations))


class PhaseProfiler:
    """Opt-in synchronized wall-clock timings for a short training diagnostic."""

    def __init__(self, output_path, warmup_updates, updates):
        self.output_path = output_path
        self.warmup_updates = warmup_updates
        self.updates = updates
        self.samples = defaultdict(list)

    @property
    def enabled(self):
        return self.output_path is not None

    def capture_training(self, optimizer_step):
        return self.enabled and self.warmup_updates <= optimizer_step < self.warmup_updates + self.updates

    def complete(self, optimizer_step):
        return self.enabled and optimizer_step >= self.warmup_updates + self.updates

    def record(self, phase, seconds):
        self.samples[phase].append(seconds)

    def write(self, config):
        if not self.enabled:
            return
        summary = {}
        for phase, values in self.samples.items():
            array = np.asarray(values, dtype=np.float64)
            summary[phase] = {
                'count': int(array.size),
                'total_s': float(array.sum()),
                'mean_s': float(array.mean()),
                'median_s': float(np.median(array)),
                'p95_s': float(np.percentile(array, 95)),
            }
        payload = {
            'schema_version': 1,
            'warmup_updates': self.warmup_updates,
            'profiled_updates': self.updates,
            'batch': config.batch,
            'subdivisions': config.subdivisions,
            'workers': config.workers,
            'input_size': [config.width, config.height],
            'phases': summary,
        }
        output_path = os.path.abspath(self.output_path)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        logging.info('Wrote phase profile to %s', output_path)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5,
          resume_checkpoint=None):
    train_dataset = Yolo_dataset(
        config.train_label, config, train=True, profile_data_pipeline=config.profile_data_pipeline,
    )
    if config.overfit_samples:
        train_dataset = _limit_dataset(train_dataset, config.overfit_samples)
        # Evaluate the exact same images without training augmentation.  This
        # is a graph/target/coordinate diagnostic, not a benchmark mode.
        val_dataset = _limit_dataset(Yolo_dataset(config.train_label, config, train=False), config.overfit_samples)
    else:
        val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    micro_batch = config.batch // config.subdivisions
    if micro_batch < 1:
        raise ValueError('batch must be greater than or equal to subdivisions')
    generator = torch.Generator().manual_seed(config.seed)
    worker_init_fn = lambda worker_id: _seed_everything(config.seed + worker_id)
    train_loader = DataLoader(train_dataset, batch_size=micro_batch, shuffle=True, drop_last=True,
                              collate_fn=collate, generator=generator, worker_init_fn=worker_init_fn,
                              **data_loader_kwargs(config.workers, device))

    val_loader = DataLoader(val_dataset, batch_size=micro_batch, shuffle=False, drop_last=False,
                            collate_fn=val_collate, worker_init_fn=worker_init_fn,
                            **data_loader_kwargs(config.workers, device))

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    optimizer_step = 0
    start_epoch = 0
    profiler = PhaseProfiler(config.profile_output, config.profile_warmup_updates, config.profile_updates)
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = config.scales[0]
        else:
            factor = config.scales[0] * config.scales[1]
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    if resume_checkpoint and 'optimizer' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        scheduler.load_state_dict(resume_checkpoint['scheduler'])
        start_epoch = int(resume_checkpoint['epoch'])
        global_step = int(resume_checkpoint.get('global_step', 0))
        optimizer_step = int(resume_checkpoint.get('optimizer_step', global_step // config.subdivisions))
        early_stopping.load_state_dict(resume_checkpoint.get('early_stopping'))
        logging.info(f'Resuming from epoch {start_epoch + 1}')

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    criterion = Yolo_loss.from_darknet(
        base_model, n_classes=config.classes, device=device, loss_mode=config.loss_mode,
    )
    criterion.profile_enabled = bool(config.profile_output)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()

    def checkpoint_payload(epoch, validation_ap=None):
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        return {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'optimizer_step': optimizer_step,
            'early_stopping': early_stopping.state_dict(),
            'validation_ap50_95': validation_ap,
            'config': dict(config),
        }

    model.train()
    updates_per_epoch = math.ceil(len(train_loader) / config.subdivisions)
    planned_updates = epochs * updates_per_epoch
    pbar = tqdm(
        total=planned_updates, initial=optimizer_step, desc='Training', unit='update', ncols=80, mininterval=5.0,
    )
    for epoch in range(start_epoch, epochs):
        # model.train()
        epoch_loss = 0
        epoch_step = 0

        # Keep the per-epoch body scoped without closing the run-wide bar.
        with nullcontext(pbar) as pbar:
            train_iterator = iter(train_loader)
            while True:
                capture_profile = profiler.capture_training(optimizer_step)
                data_start = time.perf_counter()
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    break
                if capture_profile:
                    profiler.record('train_dataloader_s', time.perf_counter() - data_start)
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]
                if capture_profile and len(batch) == 3:
                    for phase, seconds in batch[2].items():
                        profiler.record(f'train_dataset_{phase}', seconds)

                if capture_profile:
                    _synchronize(device)
                    phase_start = time.perf_counter()
                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)
                if capture_profile:
                    _synchronize(device)
                    profiler.record('train_host_to_device_s', time.perf_counter() - phase_start)

                if capture_profile:
                    _synchronize(device)
                    phase_start = time.perf_counter()
                bboxes_pred = model(images)
                if capture_profile:
                    _synchronize(device)
                    profiler.record('train_forward_s', time.perf_counter() - phase_start)

                if capture_profile:
                    _synchronize(device)
                    phase_start = time.perf_counter()
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                if capture_profile:
                    _synchronize(device)
                    profiler.record('train_loss_and_targets_s', time.perf_counter() - phase_start)
                    for phase, seconds in criterion.profile_phases.items():
                        profiler.record(f'train_{phase}', seconds)

                if capture_profile:
                    _synchronize(device)
                    phase_start = time.perf_counter()
                # loss = loss / config.subdivisions
                loss.backward()
                if capture_profile:
                    _synchronize(device)
                    profiler.record('train_backward_s', time.perf_counter() - phase_start)

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    if capture_profile:
                        _synchronize(device)
                        phase_start = time.perf_counter()
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    optimizer_step += 1
                    pbar.update(1)
                    if capture_profile:
                        _synchronize(device)
                        profiler.record('train_optimizer_s', time.perf_counter() - phase_start)

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{'epoch': f'{epoch + 1}/{epochs}', 'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_last_lr()[0] * config.batch
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_last_lr()[0] * config.batch))

            if epoch_step % config.subdivisions:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer_step += 1
                pbar.update(1)

            is_final_epoch = epoch + 1 == epochs
            profile_complete = profiler.complete(optimizer_step)
            should_evaluate = is_final_epoch or profile_complete or optimizer_step % config.eval_interval == 0
            stop_early = False
            validation_ap = None
            if should_evaluate:
                if config.use_darknet_cfg:
                    eval_model = Darknet(config.cfgfile, inference=True, width=config.width, height=config.height)
                else:
                    eval_model = Yolov4(config.pretrained, n_classes=config.classes, inference=True)
                # eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
                if torch.cuda.device_count() > 1:
                    eval_model.load_state_dict(model.module.state_dict())
                else:
                    eval_model.load_state_dict(model.state_dict())
                eval_model.to(device)
                eval_model.eval()
                validation_start = time.perf_counter()
                evaluator = evaluate(eval_model, val_loader, config, device, profiler=profiler if profiler.enabled else None)
                _synchronize(device)
                if profiler.enabled:
                    profiler.record('validation_total_s', time.perf_counter() - validation_start)
                del eval_model

                stats = evaluator.coco_eval['bbox'].stats
                validation_ap = float(stats[0])
                writer.add_scalar('train/AP', stats[0], optimizer_step)
                writer.add_scalar('train/AP50', stats[1], optimizer_step)
                writer.add_scalar('train/AP75', stats[2], optimizer_step)
                writer.add_scalar('train/AP_small', stats[3], optimizer_step)
                writer.add_scalar('train/AP_medium', stats[4], optimizer_step)
                writer.add_scalar('train/AP_large', stats[5], optimizer_step)
                writer.add_scalar('train/AR1', stats[6], optimizer_step)
                writer.add_scalar('train/AR10', stats[7], optimizer_step)
                writer.add_scalar('train/AR100', stats[8], optimizer_step)
                writer.add_scalar('train/AR_small', stats[9], optimizer_step)
                writer.add_scalar('train/AR_medium', stats[10], optimizer_step)
                writer.add_scalar('train/AR_large', stats[11], optimizer_step)
                improved, stop_early = early_stopping.update(validation_ap, optimizer_step)
                writer.add_scalar('train/early_stopping_bad_evaluations', early_stopping.bad_evaluations, optimizer_step)
                if improved:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    best_path = os.path.join(config.checkpoints, 'Yolov4_best.pth')
                    torch.save(checkpoint_payload(epoch, validation_ap), best_path)
                    logging.info(
                        'Best checkpoint saved at update %d: AP50-95=%.6f', optimizer_step, validation_ap,
                    )
                elif stop_early:
                    logging.info(
                        'Early stopping at update %d after %d non-improving evaluations; '
                        'best AP50-95=%.6f at update %d',
                        optimizer_step, early_stopping.bad_evaluations,
                        early_stopping.best_metric, early_stopping.best_step,
                    )

            should_save = is_final_epoch or stop_early or optimizer_step % config.checkpoint_interval == 0
            if save_cp and should_save:
                try:
                    # os.mkdir(config.checkpoints)
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{optimizer_step}.pth')
                torch.save(checkpoint_payload(epoch, validation_ap), save_path)
                logging.info(f'Checkpoint {optimizer_step} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')

            if stop_early or profile_complete:
                break

    pbar.close()
    writer.close()
    profiler.write(config)


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, profiler=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    evaluation_start = time.perf_counter()
    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    data_iterator = iter(data_loader)
    while True:
        data_start = time.perf_counter()
        try:
            images, targets = next(data_iterator)
        except StopIteration:
            break
        if profiler:
            profiler.record('validation_dataloader_s', time.perf_counter() - data_start)

        if profiler:
            _synchronize(device)
            phase_start = time.perf_counter()
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if profiler:
            _synchronize(device)
            profiler.record('validation_preprocess_and_host_to_device_s', time.perf_counter() - phase_start)

        if profiler:
            _synchronize(device)
            model_time = time.perf_counter()
            outputs = model(model_input)
            _synchronize(device)
            model_time = time.perf_counter() - model_time
            profiler.record('validation_forward_s', model_time)
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(model_input)
            model_time = time.time() - model_time

        if profiler:
            phase_start = time.perf_counter()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.reshape(-1, 4).cpu().detach().numpy()
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            scores = np.max(confs, axis=1).flatten()
            labels = np.argmax(confs, axis=1).flatten()
            keep = []
            for label in np.unique(labels):
                class_indices = np.where((labels == label) & (scores >= cfg.eval_conf_threshold))[0]
                if class_indices.size:
                    kept = nms_cpu(boxes[class_indices], scores[class_indices], cfg.nms_threshold)
                    keep.extend(class_indices[kept].tolist())
            keep = np.asarray(keep, dtype=np.int64)
            boxes = np.clip(boxes[keep], 0.0, 1.0)
            scores = scores[keep]
            labels = labels[keep]
            # COCO expects x, y, width, height in original-image pixels.
            boxes[:, [0, 2]] *= img_width
            boxes[:, [1, 3]] *= img_height
            boxes[:, 2:] -= boxes[:, :2]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        if profiler:
            profiler.record('validation_postprocess_s', time.perf_counter() - phase_start)

        evaluator_time = time.perf_counter()
        coco_evaluator.update(res)
        evaluator_time = time.perf_counter() - evaluator_time
        if profiler:
            profiler.record('validation_coco_update_s', evaluator_time)

    # gather the stats from all processes
    finalize_start = time.perf_counter()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    if profiler:
        profiler.record('validation_coco_finalize_s', time.perf_counter() - finalize_start)
        profiler.record('validation_evaluate_s', time.perf_counter() - evaluation_start)

    return coco_evaluator


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('--cfg', dest='cfgfile', type=str, default=Cfg.cfgfile, help='Darknet model cfg')
    parser.add_argument('--width', type=int, default=Cfg.width, help='network input width (must be divisible by 32)')
    parser.add_argument('--height', type=int, default=Cfg.height, help='network input height (must be divisible by 32)')
    parser.add_argument('--epochs', dest='TRAIN_EPOCHS', type=int, default=Cfg.TRAIN_EPOCHS)
    parser.add_argument('--batch', type=int, default=Cfg.batch, help='effective batch size')
    parser.add_argument('--subdivisions', type=int, default=Cfg.subdivisions, help='gradient-accumulation subdivisions')
    parser.add_argument('--burn-in', dest='burn_in', type=int, default=Cfg.burn_in,
                        help='optimizer updates used for fourth-power warmup')
    parser.add_argument('--steps', type=int, nargs=2, default=Cfg.steps, metavar=('STEP1', 'STEP2'),
                        help='optimizer-update milestones for learning-rate decay')
    parser.add_argument('--scales', type=float, nargs=2, default=Cfg.scales, metavar=('SCALE1', 'SCALE2'),
                        help='learning-rate multipliers at the two milestones')
    parser.add_argument('--workers', type=int, default=4, help='data-loader worker processes')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='evaluate every N optimizer updates (and at training end)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='save a checkpoint every N optimizer updates (and at training end)')
    parser.add_argument('--profile-output', type=str, default=None,
                        help='write synchronized phase timings to this JSON file and stop after the requested updates')
    parser.add_argument('--profile-warmup-updates', type=int, default=10,
                        help='optimizer updates to exclude before recording a phase profile')
    parser.add_argument('--profile-updates', type=int, default=0,
                        help='optimizer updates to record when --profile-output is set')
    parser.add_argument('--profile-data-pipeline', action='store_true',
                        help='include DataLoader-worker stage timings in an opt-in phase profile')
    parser.add_argument('--cache-images', type=int, choices=[0, 1], default=Cfg.cache_images,
                        help='cache decoded RGB source images before DataLoader workers start')
    parser.add_argument('--early-stopping-patience', type=int, default=Cfg.early_stopping_patience,
                        help='stop after N non-improving AP50-95 evaluations; 0 disables stopping')
    parser.add_argument('--early-stopping-min-delta', type=float, default=Cfg.early_stopping_min_delta,
                        help='minimum AP50-95 improvement that resets early-stopping patience')
    parser.add_argument('--loss-mode', choices=('darknet', 'legacy'), default=Cfg.loss_mode,
                        help='darknet: cfg-driven objectness/IoU loss; legacy: prior balanced BCE/MSE loss')
    parser.add_argument('--overfit-samples', type=int, default=0,
                        help='diagnostic mode: train and evaluate a fixed prefix of N training images')
    parser.add_argument('--seed', type=int, default=0, help='random seed for split-independent reproducibility')
    parser.add_argument('--mosaic', type=int, choices=[0, 1], default=Cfg.mosaic)
    parser.add_argument('--letter-box', dest='letter_box', type=int, choices=[0, 1], default=Cfg.letter_box)
    parser.add_argument('--jitter', type=float, default=Cfg.jitter)
    parser.add_argument('--hue', type=float, default=Cfg.hue)
    parser.add_argument('--saturation', type=float, default=Cfg.saturation)
    parser.add_argument('--exposure', type=float, default=Cfg.exposure)
    parser.add_argument('--flip', type=int, choices=[0, 1], default=Cfg.flip)
    parser.add_argument('--checkpoints', type=str, default=Cfg.checkpoints)
    parser.add_argument('--log-dir', dest='TRAIN_TENSORBOARD_DIR', type=str, default=Cfg.TRAIN_TENSORBOARD_DIR)
    parser.add_argument('--eval-conf-threshold', dest='eval_conf_threshold', type=float, default=0.001)
    parser.add_argument('--nms-threshold', dest='nms_threshold', type=float, default=0.5)
    parser.add_argument('-train_label_path', dest='train_label', type=str, default=Cfg.train_label, help="train label path")
    parser.add_argument('--val-label-path', dest='val_label', type=str, default=Cfg.val_label, help='validation label path')
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)
    cfg['w'] = cfg['width']
    cfg['h'] = cfg['height']
    if cfg['width'] % 32 or cfg['height'] % 32:
        parser.error('--width and --height must each be divisible by 32')
    if cfg['burn_in'] < 0:
        parser.error('--burn-in must be non-negative')
    if cfg['steps'][0] <= cfg['burn_in'] or cfg['steps'][1] <= cfg['steps'][0]:
        parser.error('--steps must satisfy burn-in < STEP1 < STEP2')
    if any(scale <= 0 for scale in cfg['scales']):
        parser.error('--scales must be positive')
    if cfg['eval_interval'] <= 0 or cfg['checkpoint_interval'] <= 0:
        parser.error('--eval-interval and --checkpoint-interval must be positive')
    if cfg['profile_warmup_updates'] < 0 or cfg['profile_updates'] < 0:
        parser.error('--profile warmup and update counts must be non-negative')
    if cfg['profile_output'] and cfg['profile_updates'] == 0:
        parser.error('--profile-updates must be positive when --profile-output is set')
    if cfg['early_stopping_patience'] < 0:
        parser.error('--early-stopping-patience must be non-negative')
    if cfg['early_stopping_min_delta'] < 0:
        parser.error('--early-stopping-min-delta must be non-negative')
    if cfg['overfit_samples'] < 0:
        parser.error('--overfit-samples must be non-negative')
    if not cfg['use_darknet_cfg']:
        parser.error('Only --use-darknet-cfg training is supported by the cfg-derived loss.')
    if cfg['letter_box'] and cfg['mosaic']:
        parser.error('--letter-box and --mosaic cannot both be enabled')
    cfg['mixup'] = 3 if cfg['mosaic'] else 0

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    logging = init_logger(log_dir=cfg.TRAIN_TENSORBOARD_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    _seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile, width=cfg.width, height=cfg.height)
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    resume_checkpoint = None
    weights_path = cfg.load or cfg.pretrained
    if cfg.use_darknet_cfg and weights_path:
        if weights_path.endswith('.weights'):
            model.load_weights(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint.get('model', checkpoint))
            if cfg.load and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
                resume_checkpoint = checkpoint
        logging.info(f'Loaded weights from {weights_path}')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device,
              resume_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
