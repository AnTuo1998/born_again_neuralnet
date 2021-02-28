# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.tensorboard import SummaryWriter


from ban import config
from ban.updater import BANUpdater, AverageMeter
from common.logger import Logger


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_gen", type=int, default=3)
    parser.add_argument("--resume_gen", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--outdir", type=str, default="snapshots")
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument('--seed', type=int, default=0, help='seed')

    args = parser.parse_args()
    set_seed(args.seed)
    logger = Logger(args)
    logger.print_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    args.save_path = os.path.join(args.outdir, args.model,
                                  f"{args.n_epoch}_{args.n_gen}")
    args.tb_folder = os.path.join(args.save_path, "tb")
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(args.tb_folder)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root="./Data",
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root="./Data",
                          train=False,
                          download=True,
                          transform=transform)
    elif args.dataset == "cifar100":
        trainset = CIFAR10(root="./Data",
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root="./Data",
                          train=False,
                          download=True,
                          transform=transform)
    else:
        raise NotImplementedError(args.dataset)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False)

    # model = config.get_model().to(device)
    model = getattr(config, args.model)().to(device)
    if args.weight:
        model.load_state_dict(torch.load(args.weight))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "n_gen": args.n_gen,
    }

    writer = SummaryWriter(log_dir=args.tb_folder)
    updater = BANUpdater(**kwargs)
    criterion = nn.CrossEntropyLoss()

    i = 0
    best_loss = 1e+9
    best_loss_list = []

    print("train...")
    for gen in range(args.resume_gen, args.n_gen):
        for epoch in range(args.n_epoch):
            train_loss = 0
            top1 = AverageMeter()
            top5 = AverageMeter()
            losses = AverageMeter()
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                t_loss, outputs = updater.update(inputs, targets, criterion)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                train_loss += t_loss

                i += 1
                if i % args.print_interval == 0:
                    writer.add_scalar("train_loss", t_loss, i)

                    test_top1 = AverageMeter()
                    test_top5 = AverageMeter()

                    val_loss = 0
                    with torch.no_grad():
                        for idx, (inputs, targets) in enumerate(test_loader):
                            inputs, targets = inputs.to(
                                device), targets.to(device)
                            outputs = updater.model(inputs)
                            loss = criterion(outputs, targets).item()
                            val_loss += loss
                            acc1, acc5 = accuracy(
                                outputs, targets, topk=(1, 5))
                            test_top1.update(acc1[0], inputs.size(0))
                            test_top5.update(acc5[0], inputs.size(0))

                    val_loss /= len(test_loader)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        last_model_weight = os.path.join(args.save_path,
                                                         f"{args.model}_{gen}.pth.tar")
                        torch.save(updater.model.state_dict(),
                                   last_model_weight)

                    writer.add_scalar("val_loss", val_loss, i)

                    logger.print_log(epoch, i, train_loss /
                                     args.print_interval, val_loss,
                                     top1.avg, top5.avg,
                                     test_top1.avg, test_top5.avg)

        print("best loss: ", best_loss)
        print("Born Again...")
        updater.register_last_model(args.model, last_model_weight, device)
        updater.gen += 1
        best_loss_list.append(best_loss)
        best_loss = 1e+9
        # model = config.get_model().to(device)
        model = getattr(config, args.model)().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        updater.model = model
        updater.optimizer = optimizer

    for gen in range(args.n_gen):
        print("Gen: ", gen,
              ", best loss: ", best_loss_list[gen])


if __name__ == "__main__":
    main()
