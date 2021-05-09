import os
import argparse
import collections
import random
import time
import sys

import numpy as np
import pandas as pd
import shutil
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import model, coco_eval, csv_eval
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, \
                                 AspectRatioBasedSampler, Augmenter, Normalizer

assert torch.__version__.split('.')[0] == '1'

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def main(parser):
    """Main Function"""
    # Set seeds for reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(parser.seed)
    np.random.seed(parser.seed)
    random.seed(parser.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f'CUDA available: {torch.cuda.is_available()}')

    # Create folder to save model states to if it doesn't exist
    if os.path.exists(parser.save_dir):
        shutil.rmtree(parser.save_dir)
    os.mkdir(parser.save_dir)
    os.mkdir(os.path.join(parser.save_dir, 'model_states'))

    # Create csv files for logging training metrics
    train_history = pd.DataFrame({'epoch': [], 'loss': [], 'cls_loss': [], 'bbox_loss': []})
    val_history   = pd.DataFrame({'epoch': [], 'mAP': [], 'precision': [], 'recall': []})
    train_history.to_csv(os.path.join(parser.save_dir, 'train_history.csv'), index=False)
    val_history.to_csv(os.path.join(parser.save_dir, 'val_history.csv'), index=False)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on CSV.')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on CSV.')

        if parser.no_normalize:
            dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                       transform=transforms.Compose([Augmenter(augment=parser.augment), Resizer()]))
        else:
            dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                       transform=transforms.Compose([Augmenter(augment=parser.augment), Normalizer(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            if parser.no_normalize:
                dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                         transform=transforms.Compose([Augmenter(augment=parser.augment), Resizer()]))
            else:
                dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                         transform=transforms.Compose([Augmenter(augment=parser.augment), Normalizer(), Resizer()]))
    
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler, worker_init_fn=worker_init_fn, pin_memory=True)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val, worker_init_fn=val_worker_init_fn, pin_memory=True)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Move model to cuda if GPU is available
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    # Verify device the model is on
    print(f'Model Device: {next(retinanet.parameters()).device}')

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print()
    print(f'Num training images: {len(dataset_train)}')
    if parser.csv_val:
        print(f'Num validation images: {len(dataset_val)}')
    print(f'Epochs: {parser.epochs}')
    print(f'Batch size: {parser.batch}')
    print(f'Backbone: ResNet{parser.depth}')
    print()

    for epoch_num in range(1, parser.epochs+1):
        epoch_start = time.perf_counter()

        retinanet.train()
        retinanet.module.freeze_bn()

        running_loss = 0.
        running_cls_loss = 0.
        running_bbox_loss = 0.

        pbar = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f'Epoch {epoch_num}')
        for iter_num, data in pbar:
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                running_loss += loss.item()
                running_cls_loss += classification_loss.item()
                running_bbox_loss += regression_loss.item()

                pbar.set_postfix({'loss': running_loss/(iter_num+1), 'cls_loss': running_cls_loss/(iter_num+1), 'bbox_loss': running_bbox_loss/(iter_num+1)})

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        l, cls_l, box_l, t = running_loss/(iter_num+1), running_cls_loss/(iter_num+1), running_bbox_loss/(iter_num+1), time.perf_counter()-epoch_start
        pbar.set_postfix({'loss': l, 'cls_loss': cls_l, 'bbox_loss': box_l, 'time': t})

        # Save metrics to csv
        current_metrics = pd.DataFrame({'epoch': [epoch_num], 'loss': [l], 'cls_loss': [cls_l], 'bbox_loss': [box_l], 'time': [t]})
        current_metrics.to_csv(os.path.join(parser.save_dir, 'train_history.csv'), mode='a', header=False, index=False)

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating validation dataset')

            mAP, precision, recall = csv_eval.evaluate(dataset_val, retinanet, save_path=parser.save_dir)

        # Save val metrics to csv
        current_metrics = pd.DataFrame({'epoch': [epoch_num], 'mAP': [mAP], 'precision': [precision], 'recall': [recall]})
        current_metrics.to_csv(os.path.join(parser.save_dir, 'val_history.csv'), mode='a', header=False, index=False)

        scheduler.step(mAP)

        torch.save(retinanet.module, os.path.join(parser.save_dir, 'model_states', f'{parser.dataset}_retinanet_{epoch_num}.pt'))
        print(f'Time for epoch {epoch_num}: {round(time.perf_counter() - epoch_start, 2)} seconds.')

    retinanet.eval()

    torch.save(retinanet, os.path.join(parser.save_dir, 'model_final.pt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for training a RetinaNet network.')

    parser.add_argument('--dataset', required=True, help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', type=int, default=50,
                        help='Resnet depth, must be one of 18, 34, 50, 101, 152.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size for training dataset.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to log metrics and model states to.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Determines whether to start with randomized or pre-trained weights.')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Determine whether to apply ImageNet mean/std normalization (ONLY WITH PRETRAINING).')
    parser.add_argument('--augment', action='store_true',
                        help='Determines whether to augment training images.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')

    parser = parser.parse_args()

    if not parser.pretrained:
        parser.no_normalize = True  # no ImageNet normalization if randomly initializing weights

    main(parser)
