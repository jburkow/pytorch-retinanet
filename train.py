import argparse
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from retinanet import coco_eval, csv_eval, model
from retinanet.dataloader import (AspectRatioBasedSampler, Augmenter,
                                  CocoDataset, CSVDataset, Normalizer, Resizer,
                                  collater)
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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

    # Set current day string
    timestr = time.strftime("%Y%m%d")

    # Create folder to save model states to if it doesn't exist
    MODEL_NAME = f'{timestr}'
    MODEL_NAME += str(parser.preprocessing) if parser.preprocessing != '' else ''
    MODEL_NAME += f'_{parser.input_type}' if parser.input_type else ''
    MODEL_NAME += f'_FiLM-resnet{parser.depth}' if parser.metadata_path != '' else f'_resnet{parser.depth}'
    MODEL_NAME += '_pretrained' if parser.pretrained else ''
    MODEL_NAME += f'_{parser.epochs}epoch'
    MODEL_NAME += '_no-norm' if parser.no_normalize else ''
    MODEL_NAME += '_aug' if parser.augment else ''
    MODEL_NAME += f'_lr{parser.lr}'
    MODEL_NAME += f'_bs{parser.batch}'
    MODEL_NAME += f'_patience{parser.lr_patience}-{parser.patience}' if parser.patience != 0 else f'_patience{parser.lr_patience}'
    MODEL_NAME += f'_seed{parser.seed}'
    MODEL_NAME += f'_split{parser.train_val_split}'
    MODEL_NAME += f'_{parser.save_name}' if parser.save_name else ''

    save_dir = os.path.join(parser.save_dir, MODEL_NAME)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'model_states'))

    # Create csv files for logging training metrics
    train_history = pd.DataFrame({'epoch': [], 'loss': [], 'cls_loss': [], 'bbox_loss': [], 'time': []})
    val_history   = pd.DataFrame({'epoch': [], 'val_metric': [], 'mAP': [], 'max_F1': [], 'max_F1_pr': [], 'max_F1_re': [], 'max_F2': [], 'max_F_pr': [], 'max_F2_re': []})
    train_history.to_csv(os.path.join(save_dir, 'train_history.csv'), index=False)
    val_history.to_csv(os.path.join(save_dir, 'val_history.csv'), index=False)

    # Create dataloaders using COCO dataset
    if parser.dataset == 'coco':
        dataset_train = CocoDataset(parser.coco_path,
                                    set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path,
                                  set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create dataloaders using custom CSV dataset
    if parser.dataset == 'csv':
        dataset_train = CSVDataset(train_file=parser.csv_train,
                                   class_list=parser.csv_classes,
                                   preprocessing=parser.preprocessing,
                                   seg_dir=parser.seg_dir,
                                   metadata_file=parser.metadata_path,
                                   transform=transforms.Compose([
                                       Augmenter(augment=parser.augment, metadata=parser.metadata_path != ''),
                                       Normalizer(no_normalize=parser.no_normalize, metadata=parser.metadata_path != ''),
                                       Resizer(metadata=parser.metadata_path != '')
                                   ]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val,
                                     class_list=parser.csv_classes,
                                     preprocessing=parser.preprocessing,
                                     seg_dir=parser.seg_dir,
                                     metadata_file=parser.metadata_path,
                                     transform=transforms.Compose([
                                         Augmenter(augment=False, metadata=parser.metadata_path != ''),
                                         Normalizer(no_normalize=parser.no_normalize, metadata=parser.metadata_path != ''),
                                         Resizer(metadata=parser.metadata_path != '')
                                     ]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler, worker_init_fn=worker_init_fn, pin_memory=True)

    ## THIS CODE IS NEVER USED
    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val, worker_init_fn=val_worker_init_fn, pin_memory=True)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FiLMed=parser.metadata_path != '')
    if parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FiLMed=parser.metadata_path != '')
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FiLMed=parser.metadata_path != '')
    if parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FiLMed=parser.metadata_path != '')
    if parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=parser.pretrained, FiLMed=parser.metadata_path != '')

    # Move model to cuda if GPU is available
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=parser.lr_patience, mode='max', verbose=True)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('\n' + f'{"TRAINING VARIABLES":=^80}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Model Device: {next(retinanet.parameters()).device}')  # Verify model is on GPU
    print(f'Num training images: {len(dataset_train)}')
    if parser.csv_val:
        print(f'Num validation images: {len(dataset_val)}')
    print(f'Epochs: {parser.epochs}')
    print(f'Batch size: {parser.batch}')
    print(f'Backbone: ResNet{parser.depth}')
    if parser.metadata_path != '':
        print('Using FiLMed RetinaNet')
    if parser.preprocessing != '':
        print(f'Using "{parser.preprocessing}" preprocessing method')
    print("="*80 + '\n')

    print(retinanet)

    best_epoch = 0
    best_val_metric = 0.
    for epoch_num in range(1, parser.epochs+1):
        epoch_start = time.perf_counter()

        retinanet.train()
        retinanet.module.freeze_bn()

        running_loss = 0.
        running_cls_loss = 0.
        running_bbox_loss = 0.

        total_iters = len(dataloader_train)
        pbar = tqdm(enumerate(dataloader_train), total=total_iters, desc=f'Epoch {epoch_num}')
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

        l, cls_l, box_l, t = running_loss/(total_iters), running_cls_loss/(total_iters), running_bbox_loss/(total_iters), time.perf_counter()-epoch_start
        pbar.set_postfix({'loss': l, 'cls_loss': cls_l, 'bbox_loss': box_l, 'time': t})

        # Save training metrics to csv
        train_metrics = pd.DataFrame({'epoch': [epoch_num], 'loss': [l], 'cls_loss': [cls_l], 'bbox_loss': [box_l], 'time': [t]})
        train_metrics.to_csv(os.path.join(save_dir, 'train_history.csv'), mode='a', header=False, index=False)

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print(f'\n{"-"*20} EPOCH {epoch_num} VALIDATION {"-"*20}')
            mAP, max_F1, max_F1_pr, max_F1_re, max_F2, max_F2_pr, max_F2_re = csv_eval.evaluate(dataset_val, retinanet, save_path=save_dir)

        if dataset_train.num_classes() > 1:
            # val_metric to use if more than one class
            val_metric = 0.0
            for _, value in mAP.items():
                val_metric += value[0]
            val_metric /= len(mAP)
        else:
            val_metric = (mAP+max_F2)/2

        if parser.patience != 0:
            if val_metric > best_val_metric:
                print(f'\nEARLY STOPPING: Validation metric has improved from {best_val_metric} to {val_metric}.\n')
                best_val_metric = val_metric
                best_epoch = epoch_num

                torch.save(retinanet.module, os.path.join(save_dir, 'model_states', f'{parser.dataset}_retinanet_{epoch_num}.pt'))
            else:
                print(f'\nEARLY STOPPING: Validation metric has not improved from {best_val_metric} (for {epoch_num-best_epoch} epochs).\n')
        else:
            if val_metric > best_val_metric:
                print(f'\nValidation metric has improved from {best_val_metric} to {val_metric}.\n')
            else:
                print(f'\nValidation metric has not improved from {best_val_metric} (for {epoch_num-best_epoch} epochs).\n')

            torch.save(retinanet.module, os.path.join(save_dir, 'model_states', f'{parser.dataset}_retinanet_{epoch_num}.pt'))

        print(f'Time for epoch {epoch_num}: {round(time.perf_counter() - epoch_start, 2)} seconds.')
        print(f'{"-"*20} END EPOCH {epoch_num} VALIDATION {"-"*20}\n')

        # Save val metrics to csv
        val_metrics = pd.DataFrame({'epoch': [epoch_num], 'val_metric': [val_metric], 'mAP': [mAP], 'max_F1': [max_F1], 'max_F1_pr': [max_F1_pr],
                                        'max_F1_re': [max_F1_re], 'max_F2': [max_F2], 'max_F_pr': [max_F2_pr], 'max_F2_re': [max_F2_re]})
        val_metrics.to_csv(os.path.join(save_dir, 'val_history.csv'), mode='a', header=False, index=False)

        scheduler.step(val_metric)

        if epoch_num - best_epoch > parser.patience > 0:
            print(f'TERMINATING TRAINING AT EPOCH {epoch_num}. BEST VALIDATION METRIC WAS {best_val_metric}.')
            break

    retinanet.eval()

    torch.save(retinanet, os.path.join(save_dir, 'model_final.pt'))


def parse_args():
    """Dedicated function to parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple script for training a RetinaNet network.')

    parser.add_argument('--dataset', required=True, choices=['coco', 'csv'],
                        help='Dataset type; must be one of ["coco", "csv"].')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--metadata_path', type=str, default='', help='Path to metadata csv file')
    parser.add_argument('--seg_dir', type=str, default='', help='Path to directory containing segmentations for each image')
    parser.add_argument('--preprocessing', type=str, default='', choices=['', 'three-filters', 'rib-seg'],
                        help='Image preprocessing method; must be one of ["", "three-filters", "rib-seg"]')
    parser.add_argument('--input_type', type=str, help='The type of varied-input processing type used (bin-bin-bin or raw-hist-bi).')

    parser.add_argument('--depth', type=int, default=50, choices=[18, 34, 50, 101, 152],
                        help='ResNet backbone depth; must be one of [18, 34, 50, 101, 152].')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size for training dataset.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate for Adam optimizer.')
    parser.add_argument('--save_dir', type=str, default='/mnt/research/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/',
                        help='Path to log metrics and model states to.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Determines whether to start with randomized or pre-trained weights.')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Determine whether to apply ImageNet mean/std normalization.')
    parser.add_argument('--augment', action='store_true',
                        help='Determines whether to augment training images.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Number of epochs of no improvement in validation metric before stopping training early.')
    parser.add_argument('--lr_patience', type=int, default=4,
                        help='Number of epochs of no improvement in validation metric before decreasing learning rate tenfold.')
    parser.add_argument('--train_val_split', type=str, default=0,
                        help='The train/val split used for the current model training.')
    parser.add_argument('--save_name', type=str,
                        help='String to add to the model name to save file and model as.')
    parser.add_argument('--csv_negative', type=str, help='Path to file containing negative annotations')
    parser.add_argument('--weights_path', type=str, help='Path to model weights to load')

    parser_args = parser.parse_args()

    if parser_args.dataset == 'coco' and parser_args.coco_path is None:
        parser.error('Must provide --coco_path when training on COCO.')

    if parser_args.dataset == 'csv':
        if parser_args.csv_train is None:
            parser.error('Must provide --csv_train when training on CSV.')

        if parser_args.csv_classes is None:
            parser.error('Must provide --csv_classes when training on CSV.')

    if not parser_args.pretrained:
        parser_args.no_normalize = True  # no ImageNet normalization if randomly initializing weights

    return parser_args


if __name__ == '__main__':
    p_args = parse_args()
    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(p_args)
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
