#!/usr/bin/env python3
"""Convert a YOLO image list into Tianxiaomo's training-label format."""

import argparse
from pathlib import Path
import re

import cv2


def convert(image_list: Path, output: Path) -> None:
    lines = []
    for raw_path in image_list.read_text(encoding="utf-8").splitlines():
        image_path = Path(raw_path.strip())
        if not image_path:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read {image_path}")
        height, width = image.shape[:2]
        boxes = []
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, center_x, center_y, box_w, box_h = map(float, line.split())
                x1 = (center_x - box_w / 2) * width
                y1 = (center_y - box_h / 2) * height
                x2 = (center_x + box_w / 2) * width
                y2 = (center_y + box_h / 2) * height
                boxes.append(f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{int(class_id)}")
        lines.append(" ".join([str(image_path), *boxes]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_cfg(template: Path, output: Path, classes: int, width: int, height: int) -> None:
    """Create a rectangular, class-correct Darknet cfg from a template."""
    sections = re.split(r"(\n\s*\n)", template.read_text(encoding="utf-8"))
    yolo_indices = [i for i, part in enumerate(sections) if part.lstrip().startswith("[yolo]")]
    if not yolo_indices:
        raise ValueError(f"No [yolo] sections in {template}")
    sections[0], width_count = re.subn(r"(?m)^width\s*=\s*\d+\s*$", f"width={width}", sections[0])
    sections[0], height_count = re.subn(r"(?m)^height\s*=\s*\d+\s*$", f"height={height}", sections[0])
    if width_count != 1 or height_count != 1:
        raise ValueError("Expected one width and one height in the [net] section")
    for yolo_index in yolo_indices:
        mask_match = re.search(r"(?m)^mask\s*=\s*([^\n]+)$", sections[yolo_index])
        if not mask_match:
            raise ValueError(f"YOLO head {yolo_index} has no anchor mask")
        anchors = len(mask_match.group(1).split(','))
        sections[yolo_index], class_count = re.subn(
            r"(?m)^classes\s*=\s*\d+\s*$", f"classes={classes}", sections[yolo_index]
        )
        if class_count != 1:
            raise ValueError(f"YOLO head {yolo_index} has no classes entry")
        for conv_index in range(yolo_index - 1, -1, -1):
            if sections[conv_index].lstrip().startswith("[convolutional]"):
                sections[conv_index], filter_count = re.subn(
                    r"(?m)^filters\s*=\s*\d+\s*$", f"filters={anchors * (classes + 5)}", sections[conv_index]
                )
                if filter_count != 1:
                    raise ValueError(f"Detection convolution before head {yolo_index} has no filters entry")
                break
        else:
            raise ValueError(f"No detection convolution before head {yolo_index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_list", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cfg-template", type=Path)
    parser.add_argument("--cfg-output", type=Path)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    convert(args.image_list, args.output)
    if args.cfg_template or args.cfg_output:
        if not all((args.cfg_template, args.cfg_output, args.classes, args.width, args.height)):
            parser.error("cfg generation requires --cfg-template, --cfg-output, --classes, --width, and --height")
        write_dataset_cfg(args.cfg_template, args.cfg_output, args.classes, args.width, args.height)


if __name__ == "__main__":
    main()
