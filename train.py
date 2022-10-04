from random import choices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse
from model import Net
from tools import print_msg, create_floder
import datetime

def load_data(args1, args2, dataset):
    if dataset == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        train_data_loader = torch.utils.data.DataLoader(dataset1,**args1)
        test_data_loader = torch.utils.data.DataLoader(dataset2,**args2)
    elif dataset == 'cifar':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
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

def test(model, device, test_loader, path, epoch):
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
    msg = 'Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    print_msg(path+"testmsg.txt", msg)

def test_mutation(model, device, test_loader, path, epoch, mutation, tp):
    model.eval()
    test_loss = 0
    correct = 0
    model.setMutation(mutation)
    model.setMutationType(tp)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()           

    test_loss /= len(test_loader.dataset)
    msg = 'Mutation: {}, Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        mutation, epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    fileName = "mutationtestmsg" + str(mutation) + ".txt"
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
    parser.add_argument('--mutationType', type=str, default='s', choices=['s', 'c'],
                        help='s=single, one kind of mutation; c=combine, combine two kind of mutation') 
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'],
                        help='dataset, mnist or cifar10') 
    
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

    model = Net().to(device) if args.dataset == "mnist" else Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #load data
    train_loader, test_loader = load_data(train_kwargs, test_kwargs, args.dataset)

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
        test(model, device, test_loader, folder_path, "End")
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