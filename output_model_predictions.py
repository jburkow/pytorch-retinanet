'''
Filename: output_model_predictions.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/18/2021
Description: Used a trained model to predict bounding boxes on a set of
    images and create a CSV output of each bounding box for each patient
    image with their corresponding scores.
'''

import argparse
import csv
import os
import random
import time
from math import ceil, floor

import cv2
import numpy as np
import pandas as pd
import torch
from retinanet.dataloader import (AspectRatioBasedSampler, Augmenter,
                                  CSVDataset, Normalizer, Resizer, collater)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def main(parse_args):
    """
    Main function.
    """
    # Create dataset and loader
    dataset = CSVDataset(train_file=parse_args.annotation_path, class_list=parse_args.classes_path, metadata_file=parse_args.metadata_path,
                         transform=transforms.Compose([Augmenter(augment=False, metadata=parse_args.metadata_path!=''),
                                                       Normalizer(no_normalize=parse_args.no_normalize, metadata=parse_args.metadata_path!=''),
                                                       Resizer(metadata=parse_args.metadata_path!='')]))

    dataloader = DataLoader(dataset, batch_size=1, num_workers=3, collate_fn=collater, shuffle=False, worker_init_fn=val_worker_init_fn, pin_memory=True)

    # Load the model
    model = torch.load(parse_args.model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # print(model)

    # Convert the model to evaluation mode
    model.eval()

    # Create prediction data frame (with image height and width if using "old" format)
    prediction_dict = pd.DataFrame(columns=['Patient', 'x1', 'y1', 'x2', 'y2', 'score'])
    # if parse_args.old:
    #     prediction_dict = pd.DataFrame(columns=['Patient', 'x1', 'y1', 'x2', 'y2', 'score'])
    # else:
    #     prediction_dict = pd.DataFrame(columns=['Patient', 'height', 'width', 'x1', 'y1', 'x2', 'y2', 'score'])

    print(f'{len(dataloader)} unique images in annotation file.')
    print(f'Beginning fracture detection on the dataset with model "{parse_args.model_path.split("/")[-1]}":')

    # PERFORM DETECTION ON TEST IMAGES!
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating dataset')
    for idx, batch in pbar:
        # Get image path of current image (this only works because shuffle=False in the data loader)
        file = dataset.image_names[idx]

        # Load image (and, optionally, metadata)
        img = batch['img'].float()
        if parse_args.metadata_path != '':
            metadata = batch['metadata'].float()

        # Transfer image (and, optionally, metadata) to GPU
        if torch.cuda.is_available():
            img = img.cuda()
            if parse_args.metadata_path != '':
                metadata = metadata.cuda()

        scale = batch['scale'][0]

        with torch.no_grad():
            # Process image
            if parse_args.metadata_path != '':
                scores, _, boxes = model([img, metadata])
            else:
                scores, _, boxes = model(img)

        # If model has no predictions, add a row with empty values
        if boxes.nelement() == 0:
            if parse_args.old:
                prediction_dict = prediction_dict.append({'Patient' : file,
                                                          'x1' : pd.NA,
                                                          'y1' : pd.NA,
                                                          'x2' : pd.NA,
                                                          'y2' : pd.NA,
                                                          'score' : pd.NA},
                                                          ignore_index=True)
            else:
                prediction_dict = prediction_dict.append({'Patient' : file,
                                                        #   'height' : img.shape[0],
                                                        #   'width' : img.shape[1],
                                                          'x1' : pd.NA,
                                                          'y1' : pd.NA,
                                                          'x2' : pd.NA,
                                                          'y2' : pd.NA,
                                                          'score' : pd.NA},
                                                          ignore_index=True)
        else:
            # Correct for image scale
            boxes /= scale

            for box, score in zip(boxes, scores):
                if parse_args.old:
                    prediction_dict = prediction_dict.append({'Patient' : file,
                                                              'x1' : int(floor(box[0].item())),
                                                              'y1' : int(floor(box[1].item())),
                                                              'x2' : int(ceil(box[2].item())),
                                                              'y2' : int(ceil(box[3].item())),
                                                              'score' : score.item()},
                                                              ignore_index=True)
                else:
                    prediction_dict = prediction_dict.append({'Patient' : file,
                                                            #   'height' : img.shape[0],
                                                            #   'width' : img.shape[1],
                                                              'x1' : int(floor(box[0].item())),
                                                              'y1' : int(floor(box[1].item())),
                                                              'x2' : int(ceil(box[2].item())),
                                                              'y2' : int(ceil(box[3].item())),
                                                              'score' : score.item()},
                                                              ignore_index=True)

    print(f'{len(prediction_dict.Patient.unique())} unique images in model predictions.')

    # Output prediction_dict to CSV
    if not parse_args.no_save:
        print('Writing to file...')
        prediction_dict.to_csv(os.path.join(parse_args.save_dir, parse_args.filename), header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output all detection boxes and corresponding scores from the trained model.')

    parser.add_argument('--annotation_path',
                        help='Path to the CSV containing dataset annotations.')

    parser.add_argument('--classes_path',
                        help='Path to the CSV containing classes.')

    parser.add_argument('--metadata_path', type=str, default='',
                        help='Path to the CSV containing metadata associated with images.')

    parser.add_argument('--model_path',
                        help='Path to the model state to load weights.')

    parser.add_argument('--save_dir', default='/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet',
                        help='Directory to save the model predictions CSV to.')

    parser.add_argument('--filename', default='retinanet_model_predictions.csv',
                        help='Filename to save the prediction output CSV as.')

    parser.add_argument('--no_save', action='store_true',
                        help='Use not to save CSV output; for debugging.')

    parser.add_argument('--no_normalize', action='store_true',
                        help='Omit ImageNet standardization of pixel values.')

    parser.add_argument('--old', action='store_true',
                        help='Omit image height and width to match "old" annotation csv format.')

    parser_args = parser.parse_args()

    if not parser_args.annotation_path:
        parser.error('Please provide a path to the annotations file.')

    if not parser_args.model_path:
        parser.error('Please provide a path to the trained model file.')

    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(parser_args)
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
