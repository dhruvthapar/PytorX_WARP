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
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from degradation_fit_functions import *
from torx.module.layer import *
from benchmark import replace_layers
from collections import defaultdict
from torch.utils.data import Subset


with open('/home/dthapar1/CHIMES/aspdac/saved_dicts/quadratic_fit_coeffs.pkl', 'rb') as f:
    fit_coeffs = pickle.load(f)

# def quadratic_I_fit(coeffs, n02, n03):
#     return (coeffs[0]
#             + coeffs[1]*n02
#             + coeffs[2]*n03
#             + coeffs[3]*n02**2
#             + coeffs[4]*n02*n03
#             + coeffs[5]*n03**2)

fits = defaultdict(dict)
def make_fit(coeffs_local):
    coeffs_local = torch.tensor(coeffs_local, dtype=torch.float32)  # CPU for now

    def I_fit(n02, n03):
        device = n02.device if torch.is_tensor(n02) else torch.device("cpu")
        coeffs = coeffs_local.to(device)   # move coeffs to same device

        n02, n03 = torch.broadcast_tensors(n02.float(), n03.float())
        Aloc = torch.stack([
            torch.ones_like(n02),
            n02, n03,
            n02**2, n02*n03, n03**2
        ], dim=-1)
        y_pred = Aloc @ coeffs
        return torch.exp(y_pred)
    return I_fit

for I in fit_coeffs:
    for C in fit_coeffs[I]:
        I_fit = make_fit(fit_coeffs[I][C])
        fits[I][C] = I_fit

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
parser.add_argument('--cuda', type=str, default='cuda:2', help='dataset to use')
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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        avg_test_loss, correct, total, test_accuracy))
    return avg_test_loss, test_accuracy

def make_balanced_subset(dataset, samples_per_class):
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    for label in class_indices:
        selected_indices.extend(class_indices[label][:samples_per_class])

    return Subset(dataset, selected_indices)

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
        # test_loader = torch.utils.data.DataLoader(
        #     testset, batch_size=128, shuffle=True, **kwargs)
        samples_per_class = 100   # change this freely
        test_subset = make_balanced_subset(testset, samples_per_class)

        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=64, shuffle=False, **kwargs
        )
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
    config_file_path = os.path.join(os.path.dirname(__file__), f'../config/')
    os.makedirs(os.path.join(os.path.dirname(__file__), f'../config/'), exist_ok=True)

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
    elif model_name == 'vgg8':
        model = VGG('VGG8', num_classes=num_classes)
    model = model.to(device)
    def run_experiment(num_runs=5, wl_lsb=None, num_cycles=None, warp=None, var_case=None):
        """Simulate running an experiment with the given config."""
        replaced_model = replace_layers(model, device, config_path=config_file_path, wl_lsb=wl_lsb, num_cycles=num_cycles, fits=fits, warp=warp, var_case=var_case)
        #replaced_model.to(device)
        #replaced_model.load_state_dict(torch.load(model_pth_file))
        #replaced_model.to(device)
        replaced_model.load_state_dict(torch.load(model_pth_file, map_location=device))
        #data parallel the model with device_ids
        # replaced_model = nn.DataParallel(replaced_model, device_ids=[0, 1])
        list_acc = []
        for i in range(num_runs):
            torch.manual_seed(i)
            print(f"Run {i+1}/{num_runs}")
            start_time = time.time()
            test_loss, test_accuracy = validate(replaced_model, test_loader, criterion, device)
            list_acc.append(test_accuracy)
            print(f"time: {time.time()-start_time}")
        print(f"mean_accuracy: {np.mean(list_acc)}")
        print(f"sigma_accuracy: {np.std(list_acc)}")
        

    # Run experiments with different configurations
    # Run the experiment
    num_runs = 10
    print(f"cuda: {args.cuda}")
    for var_case in ["default"]:#, "default", "var2"]:
        for wl_lsb in [4]:
            for num_cycles in [10**6]:
                for warp in [True]:
                    print(f'var_case: {var_case}, wl_lsb: {wl_lsb}, num_cycles: 10^{int(np.log10(num_cycles))}, warp: {warp}')
                    run_experiment(num_runs,wl_lsb=wl_lsb,num_cycles=num_cycles,warp=warp,var_case=var_case)
                    # torch.save(dict_acc,save_path+ f'{model_name}_{dataset}_error.pth')
                    print("\n")

if __name__ == '__main__':
    # args.model = 'resnet18'
    main()
