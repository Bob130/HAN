#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model import HAN_2S as model_file
from data import hand_gesture_dataset_2stream as HandGestureDatasetFile
from data import FPHA_dataset_2stream as FPHADatasetFile
from data import DHG_dataset as DHGDatasetFile

import sys
import os
import contextlib
import argparse
import time
import datetime
import math
import random

import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from thop import profile
# from ptflops import get_model_complexity_info

from progress.bar import Bar
from utils.logger import Logger
from utils.evaluation import AverageMeter
from utils.misc import save_checkpoint, save_pred, adjust_learning_rate, warm_up_learning_rate, set_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join

import numpy as np
import matplotlib
# # matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')


# plt.ion()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # make training procedure a bit faster


def main_loop(args):
    if args.loop and not args.evaluate and not args.find_lr:
        args.checkpoint += 'test/'
        seed_tmp = args.seed
        args.seed = 0
        main(args)
        args.seed = seed_tmp
        while True:
            main(args)
    else:
        main(args)


def main(args):
    # +++++++++++++++++++++++++++seed+++++++++++++++++++++++++++++++++++++++
    if args.seed >= 0:
        seed = args.seed
    else:   # args.seed < 0
        seed = int(time.time() * 256 % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ---------------------------------------------------------------------

    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []
    reduce_lr_epochs = []
    best_acc = 0
    best_epoch = 1

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    checkpoint_dir = args.checkpoint
    if args.loop and not args.evaluate and not args.find_lr:
        checkpoint_dir = checkpoint_dir[0:-5]
    checkpoint_dir1 = '/'.join(checkpoint_dir.split('/')[:-2]) + '/'
    checkpoint_dir2 = '/'.join(checkpoint_dir.split('/')[-2:])

    # copy_source_code(args)

    # create model
    model = model_file.HAN(args.class_num, args.dpout_rate, args.input_frames, args.raw_input_dim, args.dataset, device=device)

    if torch.cuda.device_count() > 1:
        print("\n\nData parallel in ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print('\n    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # dataset
    train_loader = None
    test_loader = None
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.dataset == 'HandGesture':
        train_loader = torch.utils.data.DataLoader(
            HandGestureDatasetFile.HandGestureDataset('data/HandGestureDataset_SHREC2017_dir/Train.json',
                               args.class_num, args.input_frames, is_train=True),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        test_loader = torch.utils.data.DataLoader(
            HandGestureDatasetFile.HandGestureDataset('data/HandGestureDataset_SHREC2017_dir/Test.json',
                               args.class_num, args.input_frames, is_train=False),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'FPHA':
        train_loader = torch.utils.data.DataLoader(
            FPHADatasetFile.FPHADataset('data/FPHA_dir/Train.json',
                        args.class_num, args.input_frames, is_train=True),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        test_loader = torch.utils.data.DataLoader(
            FPHADatasetFile.FPHADataset('data/FPHA_dir/Test.json',
                        args.class_num, args.input_frames, is_train=False),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'DHG':
        train_loader = torch.utils.data.DataLoader(
            DHGDatasetFile.DHGDataset('data/DHG_dir/Data.json', args.class_num, args.input_frames,
                       is_train=True, test_subject_id=args.test_subject_id),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        test_loader = torch.utils.data.DataLoader(
            DHGDatasetFile.DHGDataset('data/DHG_dir/Data.json', args.class_num, args.input_frames,
                       is_train=False, test_subject_id=args.test_subject_id),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    # --------------------------------------------------------------------------------------------------------------


    # optionally resume from a checkpoint
    title = 'HAN'
    if args.resume:
        if isfile(args.resume):
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            train_acces = checkpoint['train_acces']
            test_acces = checkpoint['test_acces']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            reduce_lr_epochs = checkpoint['reduce_lr_epochs']

            set_learning_rate(optimizer, args.lr)

            print("\n=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("\n=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # evaluate
    if args.evaluate:
        print('\n\nEvaluation only\n')
        # loss, acc, predictions = validate(test_loader, model, criterion)
        loss, acc, num, predictions = test(test_loader, model, criterion)

        save_pred(predictions, checkpoint=args.checkpoint)
        print('Test acc:%f\nTest loss:%f\nLength of test data:%d\n\n' % (acc, loss, num))
        return

    # training
    print('\n\nTraining...\n')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    if args.reduce_lr_on_plateau:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=args.patience,
                                                   threshold=0.001, threshold_mode='abs', eps=1e-15)
    old_lr = float(optimizer.param_groups[0]['lr'])
    for epoch in range(args.start_epoch, args.epochs+1):  # loop over the dataset multiple times
        warm_up_learning_rate(optimizer, args.lr, epoch, warm_up_epochs=10)

        if not args.reduce_lr_on_plateau:
            adjust_learning_rate(optimizer, epoch, old_lr, args.schedule, args.gamma)

        lr = float(optimizer.param_groups[0]['lr'])
        if old_lr-lr > 1e-10 and epoch > 1:  # when lr has been reduced (lr may be reduced at epoch 1 due to warmup)
            reduce_lr_epochs.append(epoch)
        old_lr = lr

        # early stop
        if len(reduce_lr_epochs) == 3 + 1:
            reduce_lr_epochs.pop()
            epoch -= 1
            break

        print('\nEpoch: %d | LR: %.1e' % (epoch, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)

        train_losses.append(train_loss)
        train_acces.append(train_acc)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = evaluate(test_loader, model, criterion)

        test_losses.append(valid_loss)
        test_acces.append(valid_acc)

        # ReduceLROnPlateau
        if args.reduce_lr_on_plateau:
            # scheduler.step(train_acc)
            scheduler.step(valid_acc)

        # append logger file
        logger.append([epoch, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        if is_best:
            best_epoch = epoch
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_acces': train_acces,
            'test_acces': test_acces,
            'reduce_lr_epochs': reduce_lr_epochs,
            'optimizer': optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint)

        # plot
        for axis in fig.get_axes():
            axis.clear()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylabel('acc', color='tab:blue')  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        x = list(range(1, len(train_acces)+1))
        ax1.plot(x, train_losses, color='tab:red', linestyle='--', label='train loss')
        ax1.plot(x, test_losses, color='tab:red', label='test loss')
        ax2.plot(x, train_acces, color='tab:blue', linestyle='--', label='train acc')
        ax2.plot(x, test_acces, color='tab:blue', label='test acc')
        for item in reduce_lr_epochs:
            ax1.axvline(x=item, color='tab:green', linestyle='--')
        plt.title('(train---, test___)(BatchSize=%d)(GPUs=%d)\n'
                  '(lr=%.1e, %d, [%s], decay=%.1f)\n'
                  '(weight-decay=%.1e, momentum=%.3f, seed=%d)\n'
                  '(Best_train_acc=%.4f)(Best_test_acc=%.4f, epoch=%d)\n'
                  '%s\n%s'
                  % (args.train_batch, torch.cuda.device_count(), args.lr, epoch, ','.join(map(str, reduce_lr_epochs)),
                     args.gamma, args.weight_decay, args.momentum, seed, max(train_acces), best_acc, best_epoch,
                     checkpoint_dir1, checkpoint_dir2), fontsize=12)
        plt.grid(True)
        # fig.legend(loc='upper right')
        # fig.legend(loc='best')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(os.path.join(args.checkpoint, 'log.jpg'))
        # plt.pause(0.1)

    logger.file.write('(BatchSize=%d)(Gpus=%d)\n'
                      '(lr=%.1e, %d, [%s], decay=%.1f)\n'
                      '(weight-decay=%.1e, momentum=%.3f, seed=%d)\n'
                      '(Best_train_acc=%f)(Best_test_acc=%f, epoch=%d)\n'
                      % (args.train_batch, torch.cuda.device_count(), args.lr, epoch, ','.join(map(str, reduce_lr_epochs)),
                         args.gamma, args.weight_decay, args.momentum, seed, max(train_acces), best_acc, best_epoch))
    # logger.file.write('\n' + argv_str + '\n')
    logger.close()
    # logger.plot(['Train Acc', 'Val Acc'])
    # savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('\nFinished Training\n')

    # delete checkpoint file and modify dir name
    os.remove(args.checkpoint + '_checkpoint.pth.tar')
    os.remove(args.checkpoint + '_preds.mat')
    dir_modify = args.checkpoint[0:-1]
    if args.seed >= 0:
        seed_label = '_seed' + str(args.seed)
    else:
        seed_label = ''
    zeros_label = ''
    while True:
        try:
            os.rename(dir_modify, dir_modify + seed_label + '-' + "{:.4f}".format(best_acc) + zeros_label)
            break
        except OSError:
            zeros_label += '0'
            # print('\n!!! ' + dir_modify + seed_label + '-' + "{:.4f}".format(best_acc) + ' is not empty\n')

    # plt.ioff()
    # if datetime.datetime.now().hour >= 9:
    #     plt.show()


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, labels, batch_size = data2output(data, model)
        # outputs = outputs.to('cpu')

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()   

        # acc
        _, predicted = torch.max(outputs.data, 1)
        total = batch_size
        correct = (predicted == labels).sum().item()
        acc = correct / total

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)
        acces.update(acc, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg


def evaluate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.LongTensor(val_loader.dataset.__len__(), 2)

    # switch to evaluate mode
    model.eval()

    # for child in model.children():
    #     for name, modules in child.named_modules():
    #         if 'norm' in name:
    #             modules.track_running_stats = False

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():       
        for i, data in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            end = time.time()

            # forward
            outputs, labels, batch_size = data2output(data, model)

            loss = criterion(outputs, labels)

            # acc
            _, predicted = torch.max(outputs.data, 1)
            total = batch_size
            correct = (predicted == labels).sum().item()
            acc = correct / total

            predictions[data['index'], 0] = labels.to('cpu')
            predictions[data['index'], 1] = predicted.to('cpu')
            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)
            acces.update(acc, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acces.avg
                        )
            bar.next()

    bar.finish()
    return losses.avg, acces.avg, predictions


def test(test_loader, model, criterion):
    # predictions
    predictions = torch.LongTensor(test_loader.dataset.__len__(), 2)
    # predictions.to(device)

    model.eval()

    # for child in model.children():
    #     for name, modules in child.named_modules():
    #         if 'norm' in name:
    #             modules.track_running_stats = False

    end = time.time()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # forward
            outputs, labels, batch_size = data2output(data, model)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            predictions[data['index'], 0] = labels.to('cpu')
            predictions[data['index'], 1] = predicted.to('cpu')
            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]
    print('Time: ', time.time() - end, 's')
    return test_loss / (i+1), correct / total, total, predictions


def data2output(data, model):
    # get the inputs
    input, labels = data['input'], data['label']
    # labels = labels.view(-1)
    # if 7 in labels or 8 in labels:
    #     ttt = 0
    # input = input.to(device) 
    labels = labels.to(device)

    outputs = model(input)

    batch_size = labels.size(0)

    # # compute params and FLOPs
    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # flops, params = profile(model, inputs=(input,))
    # print('Number of params: %.2fM' % (params / 1e6))
    # print('Number of FLOPs: %.2fG' % (flops / 1e9 / batch_size))
    # # ---------------------------------------------------------------------------

    # compute params and FLOPs
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # macs, params = get_model_complexity_info(model, (8, 22, 3), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # ---------------------------------------------------------------------------

    return outputs, labels, batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('-d', '--dataset', default='HandGesture', type=str,
                        help='which dataset to use (HandGesture, DHG or FPHA)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--class_num', default=14, type=int, metavar='N', help='Number of class 14 28  45')
    parser.add_argument('--input_frames', default=8, type=int, metavar='N', help='Frame length for input')
    parser.add_argument('--raw_input_dim', default=3, type=int, metavar='N', help='The feature dimension of the raw input')
    parser.add_argument('--test_subject_id', default=1, type=int, metavar='N', help='test_subject_id')

    # Training strategy
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train_batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test_batch', default=32, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--reduce_lr_on_plateau', action='store_true', default=False,
                        help='Using ReduceLROnPlateau to adjust lr.')
    parser.add_argument('--patience', default=50, type=int, metavar='N',
                        help='patience of reduce lr on plateau')
    parser.add_argument('--dpout_rate', default=0.2, type=float, metavar='M',
                        help='Dropout rate')
    parser.add_argument('--seed', default=-1, type=int, metavar='N',
                        help='seed, -1:random, >=0:seed=seed')
    parser.add_argument('--loop', default=0, type=int, metavar='N',
                        help='loop, 0:train once, 1:train loop')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/test/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--find_lr', type=float, nargs='+', default=[],
                        help='[] not find lr; [1e-5, 1e-1] find lr, lr from 1e-5 to 1e-1')

    # GPU
    parser.add_argument('--num_gpu', default=1, type=int, metavar='N',
                        help='number of gpus to use (default: 1)')
    parser.add_argument('--use_free_gpu', action='store_true', default=False,
                        help='if use free gpus')

    args = parser.parse_args()
    main_loop(args)
