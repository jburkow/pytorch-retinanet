from __future__ import division, print_function

import csv
import os
import random
import sys
from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.color
import skimage.io
import skimage.transform
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, preprocessing='', seg_dir='', metadata_file='', transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.preprocessing = preprocessing
        self.seg_dir = seg_dir
        self.metadata_file = metadata_file
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError(f'invalid CSV class file: {self.class_list}: {e}')

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise ValueError(f'invalid CSV annotations file: {self.train_file}: {e}')
        self.image_names = list(self.image_data.keys())

        # Read and prepare metadata csv
        if self.metadata_file != '':
            self.orig_metadata_df = pd.read_csv(self.metadata_file)

            self.metadata_df = self._prepare_metadata(self.orig_metadata_df)

    def _prepare_metadata(self, metadata_df):
        # Deep copy metadata df
        processed_df = deepcopy(metadata_df)

        # Subset metadata df for images in current set (determined by self.image_names)
        patient_ids = [s.split('/')[-1].split('.')[0] for s in self.image_names]

        processed_df['tmp_id'] = [s.split('/')[-1].split('.')[0][:-2] for s in processed_df['patient_id']]
        processed_df = processed_df[processed_df['tmp_id'].isin(patient_ids)]

        # Sort metadata df
        processed_df = processed_df.sort_values(by=['patient_id'])
        processed_df['patient_id'] = sorted(self.image_names)

        # Clean vendor names (e.g., reduce "CANON", "CANON INC.", and "Canon" to simply "Canon")
        vendor_list = ['Canon', 'Philips', 'Fujifilm']
        # vendor_list = ['Canon', 'Philips', 'Fujifilm', 'Swissray', 'GE', 'Kodak']
        for vendor in vendor_list:
            processed_df['vendor'] = processed_df['vendor'].apply(lambda x: vendor if vendor.upper() in x or vendor in x else x)    

        # Create "Other" vendor group for simplicity
        processed_df['vendor'] = processed_df['vendor'].apply(lambda x: 'Other' if x not in vendor_list else x)

        # Impute age with median training set age (computed elsewhere)
        processed_df['age_days'] = processed_df['age_days'].fillna(112.0)

        # Standardize age with mean and std training set age (computed elsewhere)
        processed_df['age_days'] = (processed_df['age_days'] - 201.145) / 580.376

        # Extract image path, age (days), sex, and scanner dummy variables -- 5 non-image features total
        out_df = processed_df[['patient_id', 'age_days', 'male']]
        out_df = out_df.join(pd.get_dummies(processed_df['vendor']).astype(np.float32))

        # Reset index to 0-number of images
        out_df.index = list(range(out_df.shape[0]))

        return out_df

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError(f"line {line}: format should be \'class_name,class_id\'")
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError(f"line {line}: duplicate class name: \'{class_name}\'")
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        if self.metadata_file != '':
            metadata = self.load_metadata(idx)
            sample = {'img': img, 'annot': annot, 'metadata': metadata}
        else:
            sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_metadata(self, image_index):
        metadata = self.metadata_df.loc[self.metadata_df['patient_id'] == self.image_names[image_index], ['age_days', 'male', 'Canon', 'Philips', 'Fujifilm']]

        return metadata.values.squeeze()

    def load_image(self, image_index):
        img_path = self.image_names[image_index]

        # If no special preprocessing, simply load and return the image
        if self.preprocessing == '':
            img = skimage.io.imread(self.image_names[image_index])

            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)

            return img.astype(np.uint8)

        # Get patient ID and associated chest segmentation path
        patient_id = img_path.split('/')[-1].split('.')[0]
        seg_path = os.path.join(self.seg_dir, [f for f in os.listdir(self.seg_dir) if patient_id in f][0])

        # Load image
        img = skimage.io.imread(img_path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        # Load segmentation
        seg = np.load(seg_path)

        # Extract foreground and convert to 8 bit for OpenCV usage
        fg = ((seg.sum(axis=-1) - seg[:, :, 0])*255).astype(np.uint8)

        # Zero out (mask) background pixels in original image
        masked_img = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=fg)

        if self.preprocessing == 'rib-seg':
            # Extract rough rib segmentation by applying adaptive thresholding to the masked image
            rib_seg = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 0)

            # Create 3-channel image of rib segmentation
            out = np.stack([rib_seg, rib_seg, rib_seg], axis=-1)

            return out.astype(np.uint8)

        if self.preprocessing == 'three-filters':
            masked_histeq_img = cv2.equalizeHist(masked_img)  # apply histogram equalization
            masked_bilateral_img = cv2.bilateralFilter(masked_img, 9, 75, 75)  # apply a bilateral low-pass filter

            # Stack original image, histogram-equalized image, and bilateral filtered image into RGB channels
            out = np.stack([masked_img, masked_histeq_img, masked_bilateral_img], axis=-1)

            return out.astype(np.uint8)

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]

                # If using the "three-filters" preprocessing, use non-histogram-equalized images
                if self.preprocessing == 'three-filters':
                    img_file = img_file.replace('cropped_histeq_png', 'cropped_png')
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    metadata = False

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    if 'metadata' in data[0]:
        metadata = True
        metadatas = [s['metadata'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    if metadata:
        return {'img': padded_imgs, 'annot': annot_padded, 'metadata': torch.stack(metadatas), 'scale': scales}
    else:
        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, metadata=False):
        self.metadata = metadata

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']
        if self.metadata:
            metadata = sample['metadata']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        if self.metadata:
            return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'metadata': torch.from_numpy(metadata), 'scale': scale}
        else:
            return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, metadata=False, augment=True, transform=None):
        self.metadata = metadata
        self.augment = augment
        self.transform = transform

        if self.augment:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.HorizontalFlip(p=0.5),
                # A.RandomBrightness(p=0.5),  # DEPRECATED IN FUTURE
                # A.RandomContrast(p=0.5),  # DEPRECATED IN FUTURE
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.5)
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['category_id']))

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if self.metadata:
            metadata = sample['metadata']

        if not self.augment:
            return sample

        transformed = self.transform(image=image, bboxes=annots[:, :4], category_id=annots[:, 4])
        transformed['bboxes'] = np.array(transformed['bboxes'])  #.round(0).astype(int)
        transformed['category_id'] = np.array(transformed['category_id'])

        if transformed['bboxes'].shape[0] == 0:  # if bbox(es) removed by aug s.t. none remain, undo augmentation
            return sample

        if self.metadata:
            return {'img': transformed['image'], 'annot': np.hstack((transformed['bboxes'], transformed['category_id'][:, np.newaxis])), 'metadata': metadata}
        else:
            return {'img': transformed['image'], 'annot': np.hstack((transformed['bboxes'], transformed['category_id'][:, np.newaxis]))}


class Normalizer(object):

    def __init__(self, no_normalize=False, metadata=False):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

        self.no_normalize = no_normalize  # whether to do ImageNet normalization
        self.metadata = metadata

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if self.metadata:
            metadata = sample['metadata']

        image = ((image - image.min()) / (image.max() - image.min())).astype(np.float32)

        if not self.no_normalize:
            if self.metadata:
                return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'metadata': metadata}
            else:
                return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}
        else:
            if self.metadata:
                return {'img': image, 'annot': annots, 'metadata': metadata}
            else:
                return {'img': image, 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
