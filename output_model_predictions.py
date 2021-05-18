'''
Filename: output_model_predictions.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/18/2021
Description: Used a trained model to predict bounding boxes on a set of
    images and create a CSV output of each bounding box for each patient
    image with their corresponding scores.
'''

import os
import time
import argparse
import csv

from math import floor, ceil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def read_file(file, ind=0):
    """
    Read in the file and return a list of its contents.

    Parameters
    ----------
    file : str
        path to the file to be read in
    ind : int
        index to use to pull info from each line (only used for csv's)

    Returns
    -------
    file_list : list
        list of contents of each row from the data file
    """
    temp_list = []
    if file[-4:] == '.csv':
        with open(file, 'r') as data_file:
            csv_reader = csv.reader(data_file)
            for line in csv_reader:
                temp_list.append(line[ind])
    else:
        with open(file, 'r') as data_file:
            for line in data_file:
                temp_list.append(line.replace('\n', ''))

    return temp_list


def pytorch_resize(image, min_side=608, max_side=1024):
    """
    Resizes and outputs the image and scale.
    Adopted from https://github.com/yhenon/pytorch-retinanet.
    """
    # Pull out shape of the image.
    rows, cols, cns = image.shape
    # Find the smaller side.
    smallest_side = min(rows, cols)
    # Define scale based on smallest side.
    scale = min_side / smallest_side

    # Check if larger side is now greater than max_side.
    # Can happen when images have a large aspect ratio.
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # Resize the image with the computed scale
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

    return image, scale


def main(parse_args):
    """
    Main function.
    """
    # Create a list of the test images to evaluate
    dataset = read_file(parse_args.annotation_path)

    # Get rid of duplicates
    dataset = list(set(dataset))

    # Load the model
    model = torch.load(parse_args.model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # Convert the model to evaluation mode
    model.eval()

    ### EVALUATE/DETECT ON IMAGES
    # Loop through all images and save detection boxes along with scores
    prediction_dict = pd.DataFrame(columns=['Patient', 'height', 'width', 'x1', 'y1', 'x2', 'y2', 'score'])
                                            # dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
    print(f'{len(dataset)} unique images in annotation file.')
    print(f'Beginning fracture detection on the dataset with model "{parse_args.model_path.split("/")[-1]}":')

    pbar = tqdm(iterable=enumerate(dataset), total=len(dataset), desc='Evaluating dataset')
    for _, file in pbar:
        # Load in image
        image = cv2.imread(file)

        # Get shape of image
        im_shape = image.shape

        # Scale image for network
        image, scale = pytorch_resize(image)

        # Process image
        with torch.no_grad():
            # Convert image array to tensor
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            # Process image
            scores, _, boxes = model(image.float())

        # If model has no predictions, add a row with empty values
        if boxes.nelement() == 0:
            prediction_dict = prediction_dict.append({'Patient' : file,
                                                        'height' : im_shape[0],
                                                        'width' : im_shape[1],
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
                prediction_dict = prediction_dict.append({'Patient' : file,
                                                        'height' : im_shape[0],
                                                        'width' : im_shape[1],
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

    parser.add_argument('--model_path',
                        help='Path to the model state to load weights.')

    parser.add_argument('--save_dir', default='/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet',
                        help='Directory to save the model predictions CSV to.')

    parser.add_argument('--filename', default='retinanet_model_predictions.csv',
                        help='Filename to save the prediction output CSV as.')

    parser.add_argument('--no_save', action='store_true',
                        help='Use not to save CSV output; for debugging.')

    parser_args = parser.parse_args()

    if not parser_args.annotation_path:
        parser.error('Please provide a path to the annotations file.')

    if not parser_args.model_path:
        parser.error('Please provide a path to the trained model file.')

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
