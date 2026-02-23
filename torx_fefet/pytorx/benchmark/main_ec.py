import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import sys
import copy
import math
import matplotlib.pyplot as plt
from torx.module.layer import *
from benchmark import replace_layers
import argparse
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
# from torchsummary import summary
from resnet import *
from mobilenet import *
from densenet import * 
from VGG import *
import json

def load_config(file_path):
    """Load the config file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, file_path):
    """Save the modified config file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
##arguments
# Arguments setup
parser = argparse.ArgumentParser(description='PyTorch CNN Inference')
parser.add_argument('--model', type=str, default='resnet18', help='model to use')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--trained_model_path', type=str, default='trained_models', help='path to trained model')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--cuda', type=str, default='cuda:0', help='dataset to use')
parser.add_argument('--fault_dist', type=str, default='uniform', help='fault dist to be used')
args = parser.parse_args()
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100. * correct / total
    avg_test_loss = running_loss / len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        avg_test_loss, correct, total, test_accuracy))
    return avg_test_loss, test_accuracy
def main():
    ##load dataset
    cuda = args.cuda
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    dataset= args.dataset
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # Data Preparation
    if dataset == 'cifar10':
        # Define the folder name
        folder_name = "cifar-10"

        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")
        save_path = os.path.join(os.getcwd(), f'{folder_name}') + '/'
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std= (0.2470, 0.2435, 0.2616)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, **kwargs)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, **kwargs)
    elif dataset == 'cifar100':
        folder_name = "cifar-100"

        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")

        save_path = os.path.join(os.getcwd(), f'{folder_name}') + '/'
        num_classes = 100
        mean =  (0.5071, 0.4867, 0.4408)
        std =  (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, **kwargs)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, **kwargs)

 ##define model
    model_name = args.model

    criterion = nn.CrossEntropyLoss()
    #pth file
    trained_model_path = args.trained_model_path
    model_pth_file = os.path.join(trained_model_path , f'{model_name}_{dataset}.pth')
    config_file_path = os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}')
    os.makedirs(os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}/{model_name}'), exist_ok=True)
    new_config_file_path = os.path.join(os.path.dirname(__file__), f'../config/{args.fault_dist}/{model_name}')
    # Here, you would add your code to run the experiment based on the config
    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif model_name == 'vgg16':
        # torch.cuda.memory_summary()
        model = VGG('VGG16', num_classes=num_classes)
    elif model_name == 'mobilenet':
        model = MobileNet(num_classes=num_classes)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes)
    model = model.to(device)
    # Load the original configuration
    config = load_config(config_file_path+'/config.json')
    def run_experiment(list_acc=[]):
        """Simulate running an experiment with the given config."""
        replaced_model = replace_layers(model, device,new_config_file_path)
        replaced_model.to(device)
        replaced_model.load_state_dict(torch.load(model_pth_file))
        replaced_model.to(device)
        test_loss, test_accuracy = validate(replaced_model, test_loader, criterion, device)
        list_acc.append(test_accuracy)

    experiment_settings = [
            # {'fault_rate': 0.00},
            {'fault_rate': 0.001},
            {'fault_rate': 0.005},
            {'fault_rate': 0.01},
            {'fault_rate': 0.05},
            {'fault_rate': 0.1},
            {'fault_rate': 0.15},
        ]
    experiment_settings_ec = [
            {'fault_rate': 0.001, 'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.005,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.01,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.05,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.1,   'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.15,   'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.2,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.30,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.4,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            {'fault_rate': 0.50,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True},
            # {'fault_rate': 0.01,'enable_ec_SAF': True, 'enable_ec_SAF_msb': True}
        ]
    dict_acc = {}
    num_runs = 100
    dict_acc_ec = {}
    num_runs = 10
    dict_acc_ec = {}
    for exp_config in experiment_settings_ec:
        # Update the original configuration with the experimental setup
        fault_rate = exp_config['fault_rate']
        print("running experiment with EC fault rate and model: ", fault_rate*100, model_name)
        new_config = {**config, **exp_config}
        # Optionally, save the modified configuration to a file
        save_config(new_config, new_config_file_path+'/config.json')
        dict_acc_ec[f'fault_{fault_rate*100}'] = []
        accuracy_list = dict_acc_ec[f'fault_{fault_rate*100}']

        for i in range(num_runs):
            set_seed(i)
            print(f"Run {i+1}/{num_runs}")
            # Run the experiment
            run_experiment(accuracy_list)
        torch.save(dict_acc_ec, save_path+f'{model_name}_{dataset}_{args.fault_dist}_error_ec.pth')
    
    for exp_config in experiment_settings:
        # Update the original configuration with the experimental setup
        fault_rate = exp_config['fault_rate']
        print("running experiment with fault rate and model: ", fault_rate*100, model_name)
        new_config = {**config, **exp_config}
        # Optionally, save the modified configuration to a file
        save_config(new_config, new_config_file_path+'/config.json')
        dict_acc[f'fault_{fault_rate*100}'] = []
        accuracy_list = dict_acc[f'fault_{fault_rate*100}']
  
        for i in range(num_runs):
            set_seed(i)
            print(f"Run {i+1}/{num_runs}")
            # Run the experiment
            run_experiment(accuracy_list)
        # torch.save(dict_acc_ec, save_path+f'{model_name}_{dataset}_error_ec.pth')
        torch.save(dict_acc,save_path+ f'{model_name}_{dataset}_{args.fault_dist}_error.pth')

if __name__ == '__main__':
    # args.model = 'resnet18'
    main()
