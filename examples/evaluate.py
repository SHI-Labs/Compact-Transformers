#!/usr/bin/env python

import argparse
from time import time
import math

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import src as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(models.__dict__[name]))

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick evaluation script')

    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        choices=model_names,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=2, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--gpu-id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')

    parser.add_argument('--download', action='store_true',
                        help='download dataset (or verify if already downloaded)')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']

    model = models.__dict__[args.model]('', False, False, img_size=img_size,
                                        num_classes=num_classes,
                                        positional_embedding=args.positional_embedding,
                                        n_conv_layers=args.conv_layers,
                                        kernel_size=args.conv_size,
                                        patch_size=args.patch_size)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    print("Loaded checkpoint.")

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)

    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.data, train=False, download=args.download, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    print("Beginning evaluation")
    time_begin = time()
    acc1 = cls_validate(val_loader, model, args, time_begin=time_begin)

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'final top-1: {acc1:.2f}')


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_validate(val_loader, model, args, time_begin=None):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu_id, non_blocking=True)
                target = target.cuda(args.gpu_id, non_blocking=True)

            output = model(images)

            acc1 = accuracy(output, target)
            n += images.size(0)
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_acc1 = (acc1_val / n)
                print(f'[Eval][{i}] \t Top-1 {avg_acc1:6.2f}')

    avg_acc1 = (acc1_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Final]\t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

    return avg_acc1


if __name__ == '__main__':
    main()
