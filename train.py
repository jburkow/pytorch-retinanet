import os
import argparse
import collections
import time

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import model, coco_eval, csv_eval
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, \
                                 AspectRatioBasedSampler, Augmenter, Normalizer

assert torch.__version__.split('.')[0] == '1'


def main(parser):
    """Main Function"""
    print(f'CUDA available: {torch.cuda.is_available()}')

    # Create folder to save model states to if it doesn't exist
    if not os.path.exists(parser.snapshot_path):
        os.mkdir(parser.snapshot_path)

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

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=parser.random)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=parser.random)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=parser.random)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=parser.random)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=parser.random)
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

    for epoch_num in range(parser.epochs):
        epoch_start = time.perf_counter()

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            iter_start = time.perf_counter()
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

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                iter_end = time.perf_counter()

                print(
                    'Epoch: {} | Iteration: {} | {:.3} sec | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, iter_end - iter_start, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating validation dataset')

            csv_eval.evaluate(dataset_val, retinanet, save_path=parser.snapshot_path)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, os.path.join(parser.snapshot_path, f'{parser.dataset}_retinanet_{epoch_num}.pt'))
        print(f'Time for epoch {epoch_num}: {round(time.perf_counter() - epoch_start, 2)} seconds.')

    retinanet.eval()

    torch.save(retinanet, os.path.join(parser.snapshot_path, 'model_final.pt'))


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
    parser.add_argument('--snapshot_path', type=str, required=True,
                        help='Path to save model states to.')
    parser.add_argument('--random', action='store_false',
                        help='Determines whether to start with randomized or pre-trained weights.')

    parser = parser.parse_args()

    main(parser)
