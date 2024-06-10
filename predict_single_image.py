'''
Filename: predict_single_image.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 06/06/2024
Description: Predict bounding boxes on a single image using a trained RetinaNet model.
'''

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch

CV2_COLORS = {  # cv2 saves colors in BGR
    "teal": [255, 255, 0],
    "green": [0, 255, 0],
    "yellow": [0, 255, 255],
    "red": [0, 0, 255],
    "blue": [255, 0, 0],
    "magenta": [255, 0, 255],
    "pastel-blue": [255, 128, 70],
    "orange": [0, 127, 255],
    "purple": [255, 0, 128],
    "dark-teal": [128, 128, 0],
    "pink": [180, 105, 255]
}


def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_path', help='Path to images to predict boxes on.')
    parser.add_argument('--model_path', help='Path to trained model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names')
    parser.add_argument('--save_dir', type=str, help='Directory to save image to.')

    return parser.parse_args()


def draw_box(image: np.ndarray, box: List[int], color: Union[str, list], thickness: int = 2) -> None:
    """
    Draw a bounding box on the image.

    Parameters
    ----------
    image     : the image to draw on
    box       : top left and bottom right coordinates of the bounding box; [x1, y1, x2, y2]
    color     : string or list representing the color to make the box
    thickness : thickness/width of the bounding box lines
    """
    if isinstance(color, str):
        color = CV2_COLORS[color]
    box = np.array(box).astype(int)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image: np.ndarray, box: List[int], caption: str, loc: str = '') -> None:
    """
    Write a caption above or below the bounding box.

    Parameters
    ----------
    image   : the image to write on
    box     : top left and bottom right coordinates of the bounding box; [x1, y1, x2, y2]
    caption : text string to write on image next to the box
    loc     : location of where to draw caption (either bottom or default to top)
    """
    box = np.array(box).astype(int)
    if loc == 'bottom':
        cv2.putText(image, caption, (box[0], box[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (box[0], box[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    else:
        cv2.putText(image, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError(f'line {line + 1}: format should be \'class_name,class_id\'')
        class_id = int(class_id)

        if class_name in result:
            raise ValueError(f'line {line + 1}: duplicate class name: \'{class_name}\'')
        result[class_name] = class_id
    return result


def detect_image(image_path, model_path, class_list, save_dir):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    filename = Path(image_path).stem
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image at {image_path}.\nExiting...")
        return

    image_orig = image.copy()

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.cuda().float())
        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            print(bbox, classification.shape)
            score = scores[j]
            caption = '{} {:.3f}'.format(label_name, score)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            draw_box(image_orig, (x1, y1, x2, y2), color=CV2_COLORS['red'])

        cv2.imwrite(os.path.join(save_dir, f'{filename}_detections.png'), image_orig)


def main():
    """Main Function"""
    parse_args = parse_cmd_args()

    detect_image(parse_args.image_path, parse_args.model_path, parse_args.class_list, parse_args.save_dir)


if __name__ == "__main__":
    print(f"\n{'Starting execution: ' + Path(__file__).name:-^80}\n")
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'Done!':-^80}")
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
