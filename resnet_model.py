'''Train ResNet-18 on CIFAR10 from scratch with PyTorch.'''
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


EPOCHS = 200  # lower this (e.g. 30) for a quick first run; the LR schedule follows it


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==> Using device: {device}')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8,
        pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True)

    # Model
    print('==> Building ResNet-18 model..')
    net = resnet18(weights=None, num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net = net.to(device, memory_format=torch.channels_last)

    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    use_amp = device == 'cuda'

    # Training
    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return train_loss / len(trainloader), 100. * correct / total

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return test_loss / len(testloader), 100. * correct / total

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        train_loss, train_acc = train()
        test_loss, test_acc = test()
        scheduler.step()

        print(f'Epoch {epoch + 1:3d}/{EPOCHS} | '
              f'train loss {train_loss:.3f} acc {train_acc:.2f}% | '
              f'test loss {test_loss:.3f} acc {test_acc:.2f}% | '
              f'lr {scheduler.get_last_lr()[0]:.4f} | {time.time() - start:.1f}s')

        # Save checkpoint.
        if test_acc > best_acc:
            print('  Saving..')
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = test_acc

    print(f'==> Done. Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()