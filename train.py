from curses.panel import top_panel
from gettext import translation
from random import choices
from turtle import left, right
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse
from model import Net, Net1, Net2, Net3
from tools import print_msg, create_floder
import datetime

#generate a random translate on Mnist dataset
def randomTranslateMnist(i):
    if random.random() < 0.5:
        desireSize = 45
        left = random.randint(-1*(desireSize-28),0) #mnist size 28*28
        #right = random.randint(-28-left,0)
        top = random.randint(-1*(desireSize-28),0)
        #bot = random.randint(-28-top,0)
        #res = transforms.functional.crop(i, left = left, top = top, height = 28+(-1)*top+(-1)*bot, width = 28+(-1)*left+(-1)*right)
        res = transforms.functional.crop(i, left = left, top = top, height = desireSize, width = desireSize)
        return transforms.functional.resize(res, 28)
    return transforms.functional.resize(i, 28)
#generate a random translate on Cifar dataset
def randomTranslateCifar(i):
    if random.random() < 0.75:
        desireSize = 50
        left = random.randint(-1*(desireSize-32),0) #cifar size 32*32
        #right = random.randint(-32-left,0)
        top = random.randint(-1*(desireSize-32),0)
        #bot = random.randint(-32-top,0)
        res = transforms.functional.crop(i, left = left, top = top, height = desireSize, width = desireSize)
        return transforms.functional.resize(res, 32)
    return transforms.functional.resize(i, 32)


def load_data(args1, args2, dataset, status):
    if dataset == "mnist":
        if status == 0:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
        elif status == 5:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        elif status == 6:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=0, translate=(0.5,0.5), scale=(0.6,0.65)),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        elif status == 7:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(randomTranslateMnist),
                transforms.Normalize((0.1307,), (0.3081,))
                #transforms.Resize(28)
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        train_data_loader = torch.utils.data.DataLoader(dataset1,**args1)
        test_data_loader = torch.utils.data.DataLoader(dataset2,**args2)
    elif dataset == 'cifar':
        if status == 0:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])    
        elif status == 1:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])      
        elif status == 2:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])  
        elif status == 3:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        elif status == 4:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        elif status == 6:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=0, translate=(0.5,0.5), scale=(0.6,0.65)),
                transforms.Resize(32),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        elif status == 7:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(randomTranslateCifar),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.CIFAR10('../data', train=False,
                        transform=transform)
        train_data_loader = torch.utils.data.DataLoader(dataset1,**args1)
        test_data_loader = torch.utils.data.DataLoader(dataset2,**args2)        
    else:
        train_data_loader = test_data_loader = None
    return train_data_loader, test_data_loader

#mnist 28*28
#cifar 32*32


#reduceFactor is used to reduce blank area, avoid the suqare of feature is too tiny to recoginze
#set to 1: no reduce, full pad method
def load_data_after_pad(args1, args2, dataset, percent, location, reduceFactor):
    size = 28 if dataset == 'mnist' else 32
    block_size = size // reduceFactor
    #location 0 -- percent*percent-1
    target_block_row = location//percent
    target_block_col = location%percent
    left = target_block_col * block_size
    right = (percent-target_block_col-1) * block_size
    top = target_block_row * block_size
    bot = (percent-target_block_row-1) * block_size

    if dataset == "mnist":
        transform_pad=transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(padding=[left,top,right,bot], fill=0),#left,top,right,bot
            transforms.Resize(28),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        ds = datasets.MNIST('../data', train=False,
                        transform=transform_pad)
    elif dataset == "cifar":
        transform_pad=transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad([left,top,right,bot]),#left,top,right,bot
            transforms.Resize(32),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])  
        ds = datasets.CIFAR10('../data', train=False,
                        transform=transform_pad)
    else:
        ds = None
    return torch.utils.data.DataLoader(ds,**args2)
 

def train(args, model, device, train_loader, optimizer, epoch, path):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item())
        print_msg(path+"trainmsg.txt", msg)

#parameter test_pad decide whether this function is used for normal test or test_pad
def test(model, device, test_loader, path, epoch, test_pad = False, percent = 0, location = 0, reduceFactor = 1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if test_pad == False:
        msg = 'Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Total Correct: {}, Total Case: {}\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset))
    else:
        msg = 'Test_pad Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Total Correct: {}, Total Case: {}, split_percent: {}, cur_location: {}, reduce_factor: {}\n'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset), percent, location, reduceFactor)
    print_msg(path+"testmsg.txt", msg)

def test_mutation(model, device, test_loader, path, epoch, mutation, tp, percent=3, location=1):
    model.eval()
    test_loss = 0
    correct = 0
    model.setMutation(mutation)
    model.setMutationType(tp)
    model.setPercent(percent)
    model.setLocation(location)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()           

    test_loss /= len(test_loader.dataset)
    if tp == 's' or tp == 'c':
        msg = 'Mutation: {}, Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Total Correct: {}, Total Case: {}\n'.format(
            mutation, epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset))
        fileName = "mutationtestmsg" + str(mutation) + ".txt"
    elif tp == 'r':
        msg = 'Remove percent: {}, Current Block Location: {}, Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Total Correct: {}, Total Case: {}\n'.format(
            percent, location, epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct, len(test_loader.dataset))
        fileName = "removetestmsg" + str(percent) + " " + str(location) + ".txt"
    print_msg(path+fileName, msg)


def main():
    #training settings parsing from input
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu','cuda'])
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='index of gpu you want to use')
    parser.add_argument('--evaluate', type=str, default='train', choices=['train', 'eva'],
                        help='train=train, evaluate=eva') 
    parser.add_argument('--mutationType', type=str, default='s', choices=['s', 'c', 'r', 'n'],
                        help='s=single, one kind of mutation; c=combine, combine two kind of mutation; r=remove block; n=none mutation test') 
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'],
                        help='dataset, mnist or cifar10') 
    parser.add_argument('--rmp', type=int, default=3,
                        help='remove percent,2,3,4,5')
    parser.add_argument('--model', type=int, default=1,
                        help='1 or 2 for mnist, 3 for cifar')
                        
    parser.add_argument('--padtest', type=int, default=0,
                        help='0 not pad test, 1 for pad test')

    parser.add_argument('--aug', type=int, default=0,
                        help='0 donot use data augment to solve; 1 random horizatal flip for cifar; \
                        2 random vertical for cifar; 3 random rotation for cifar; 4 combine h-flip with v-flip for cifar;\
                        5 random perceptive for mnist')

    parser.add_argument('--wei', type=int, default=0,
                        help='0 donot use weight adjust to solve; 1 or other int use')
    
    args = parser.parse_args()

    dt = str(datetime.datetime.now().strftime('%Y%m%d %H%M%S'))
    folder_path = dt + "/"

    create_floder(folder_path)

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.model == 1 and args.dataset == 'mnist':
        model = Net1().to(device)
    elif args.model == 2 and args.dataset == 'mnist':
        model = Net2().to(device)
    elif args.model == 3 and args.dataset == 'cifar':
        model = Net3().to(device)
    else:
        print("Wrong parameter! Only support model = 1 or 2 for dataset mnist and model = 3 for cifar!")
        exit(0)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #load data
    train_loader, test_loader = load_data(train_kwargs, test_kwargs, args.dataset, args.aug)

    mutation_types = [i for i in range(1, 14)] if args.mutationType == 's' else [i for i in range(1,8)]

    print("begin to "+ str(args.evaluate))

    if args.evaluate == 'train':
        #train + test
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, folder_path)
            #evaluate performance on test data every two epoch
            if epoch % 2 == 0:
                test(model, device, test_loader, folder_path, epoch)
            scheduler.step()

        #original version of test_loader
        test(model, device, test_loader, folder_path, "End")
        
        
        if args.padtest == 1:
            for reduceFactor in range(1,6):
                #pad test dataset for different block
                if args.mutationType == 'r':
                    for location in range(0, int(args.rmp) * int(args.rmp)):
                        test_loader_pad = load_data_after_pad(train_kwargs, test_kwargs, args.dataset, args.rmp, location, reduceFactor)
                        test(model, device, test_loader_pad, folder_path, "End", True, args.rmp, location, reduceFactor)
        
        if args.mutationType == 'c' or args.mutationType == 's':
            for mt in mutation_types:
                test_mutation(model, device, test_loader, folder_path, 'End', mt, args.mutationType)
        elif args.mutationType == 'r':
            for location in range(0, int(args.rmp) * int(args.rmp)):
                #print(location)
                test_mutation(model, device, test_loader, folder_path, 'End', 0, args.mutationType, args.rmp, location)
        #save model
        sd_path = args.dataset + "_cnn.pt"
        torch.save(model.state_dict(), sd_path)
    else:
        path = args.dataset + "_cnn.pt"
        model.load_state_dict(torch.load(path))
        for mt in mutation_types:
            test_mutation(model, device, test_loader, folder_path, 'End', mt, args.mutationType)




if __name__ == '__main__':
    main()