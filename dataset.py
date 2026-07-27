# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys
import zlib
import time
from collections import defaultdict

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth, timings=None):
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        resize_start = time.perf_counter() if timings is not None else None
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if src_rect == [0, 0, ow, oh]:
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            # The colour transform below converts to float32.  Creating this
            # transient canvas as NumPy's default float64 makes every random
            # crop substantially more expensive without retaining precision.
            cropped = np.empty((sheight, swidth, 3), dtype=np.float32)
            cropped[...] = cv2.mean(img)[:3]

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
        if timings is not None:
            timings['resize_and_flip_s'] += time.perf_counter() - resize_start

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            color_start = time.perf_counter() if timings is not None else None
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = list(cv2.split(hsv_src))
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp
            if timings is not None:
                timings['color_s'] += time.perf_counter() - color_start

        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except Exception as exc:
        raise RuntimeError(f"OpenCV augmentation failed for {w}x{h}") from exc

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, label_path, cfg, train=True, profile_data_pipeline=False):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise ValueError("cutmix is not supported")
        elif cfg.mixup == 3 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise ValueError("mosaic and letter_box cannot be enabled together")

        self.cfg = cfg
        self.train = train
        self.profile_data_pipeline = profile_data_pipeline

        truth = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            data = line.split()
            if not data:
                continue
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())
        self.cache_images = bool(getattr(cfg, 'cache_images', False))
        self.image_cache = {}
        if self.cache_images:
            # Populate in the parent process.  Linux DataLoader workers fork
            # this read-only cache, avoiding repeated JPEG decode without
            # multiplying memory for every worker.
            for img_path in self.imgs:
                self._load_image(img_path)

    def _resolve_image_path(self, img_path):
        return os.path.join(self.cfg.dataset_dir, img_path)

    def _load_image(self, img_path):
        cached = self.image_cache.get(img_path)
        if cached is not None:
            return cached
        full_path = self._resolve_image_path(img_path)
        bgr = cv2.imread(full_path)
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {full_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.cache_images:
            self.image_cache[img_path] = image
        return image

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        sample_start = time.perf_counter() if self.profile_data_pipeline else None
        timings = {'image_load_s': 0.0, 'target_transform_s': 0.0, 'augmentation_s': 0.0}
        img_path = self.imgs[index]
        bboxes = np.asarray(self.truth.get(img_path), dtype=np.float32).reshape(-1, 5)
        use_mixup = self.cfg.mixup
        if random.randint(0, 1):
            use_mixup = 0

        if use_mixup == 3:
            min_offset = 0.2
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])
        out_bboxes = []

        for i in range(use_mixup + 1):
            if i != 0:
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.asarray(self.truth.get(img_path), dtype=np.float32).reshape(-1, 5)
            phase_start = time.perf_counter() if self.profile_data_pipeline else None
            img = self._load_image(img_path)
            if self.profile_data_pipeline:
                timings['image_load_s'] += time.perf_counter() - phase_start
            oh, ow, oc = img.shape
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int32)

            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
            dsat = rand_scale(self.cfg.saturation)
            dexp = rand_scale(self.cfg.exposure)

            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            flip = random.randint(0, 1) if self.cfg.flip else 0

            if (self.cfg.blur):
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:
                gaussian_noise = 0

            if self.cfg.letter_box:
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            phase_start = time.perf_counter() if self.profile_data_pipeline else None
            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if self.profile_data_pipeline:
                timings['target_transform_s'] += time.perf_counter() - phase_start
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            phase_start = time.perf_counter() if self.profile_data_pipeline else None
            augmentation_timings = defaultdict(float) if self.profile_data_pipeline else None
            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth, timings=augmentation_timings)
            if self.profile_data_pipeline:
                timings['augmentation_s'] += time.perf_counter() - phase_start
                timings['augmentation_resize_and_flip_s'] = (
                    timings.get('augmentation_resize_and_flip_s', 0.0) + augmentation_timings['resize_and_flip_s']
                )
                timings['augmentation_color_s'] = (
                    timings.get('augmentation_color_s', 0.0) + augmentation_timings['color_s']
                )

            if use_mixup == 0:
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

                right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
                bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        if self.profile_data_pipeline:
            timings['other_s'] = time.perf_counter() - sample_start - (
                timings['image_load_s'] + timings['target_transform_s'] + timings['augmentation_s']
            )
            return out_img, out_bboxes1, timings
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.asarray(self.truth.get(img_path), dtype=np.float32).reshape(-1, 5)
        img = self._load_image(img_path)
        # img_height, img_width = img.shape[:2]
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # lv = lv.replace("level", "")
    # no = f"{int(no):04d}"
    # return int(lv+no)

    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(stem)
    except ValueError:
        # COCO requires an integer ID; a deterministic hash supports ordinary
        # filenames without relying on a dataset-specific naming convention.
        return zlib.crc32(filename.encode('utf-8')) & 0x7fffffff


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    Cfg.dataset_dir = '/mnt/e/Dataset'
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(a.astype(np.int32))
        plt.show()
