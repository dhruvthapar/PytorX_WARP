from __future__ import print_function

import argparse
import os
import shutil
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from python.torx.module.layer import crxb_Conv2d
from python.torx.module.layer import crxb_Linear
from python.torx.module.fault_assign import *

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Net(nn.Module):
    def __init__(self, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, fault_rate, fault_dist):
        super(Net, self).__init__()

        saf_rate_dict = {
            'conv1': {'sa00': 0.25, 'sa01': 0.25, 'sa10':0.25, 'sa11': 0.25},
            'conv2': {'sa00': 0.25, 'sa01': 0.25, 'sa10':0.25, 'sa11': 0.25},
            'fc1': {'sa00': 0.25, 'sa01': 0.25, 'sa10':0.25, 'sa11': 0.25},
            'fc2': {'sa00': 0.25, 'sa01': 0.25, 'sa10':0.25, 'sa11': 0.25},
        }

        # XB_fault_rate = []
        # self.no_of_forward_pass = 0
        # NO_OF_XBs = torch.tensor(0.0)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.conv1 = crxb_Conv2d(1, 10, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device, 
                                 sa00_rate=saf_rate_dict['conv1']['sa00'],
                                 sa01_rate=saf_rate_dict['conv1']['sa01'],
                                 sa10_rate=saf_rate_dict['conv1']['sa10'],
                                 sa11_rate=saf_rate_dict['conv1']['sa11'], 
                                 fault_rate = fault_rate,
                                 fault_dist = fault_dist)
        # self.conv1.register_forward_hook(self.forward_hook)
        # XB_fault_rate.extend(self.conv1.w2g.SAF_pos.total_faults_per_crossbar.flatten().tolist())
        # XB_fault_rate.extend(self.conv1.w2g.SAF_neg.total_faults_per_crossbar.flatten().tolist())
        # NO_OF_XBs.add_(self.conv1.crxb_col * self.conv1.crxb_row * 2)

        self.conv2 = crxb_Conv2d(10, 20, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device,
                                 sa00_rate=saf_rate_dict['conv2']['sa00'],
                                 sa01_rate=saf_rate_dict['conv2']['sa01'],
                                 sa10_rate=saf_rate_dict['conv2']['sa10'],
                                 sa11_rate=saf_rate_dict['conv2']['sa11'], 
                                 fault_rate = fault_rate,
                                 fault_dist = fault_dist)
        # XB_fault_rate.extend(self.conv2.w2g.SAF_pos.total_faults_per_crossbar.flatten().tolist())
        # XB_fault_rate.extend(self.conv2.w2g.SAF_neg.total_faults_per_crossbar.flatten().tolist())
        # NO_OF_XBs.add_(self.conv2.crxb_col * self.conv2.crxb_row * 2)
        self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = crxb_Linear(320, 50, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF,
                               sa00_rate=saf_rate_dict['fc1']['sa00'],
                               sa01_rate=saf_rate_dict['fc1']['sa01'],
                               sa10_rate=saf_rate_dict['fc1']['sa10'],
                               sa11_rate=saf_rate_dict['fc1']['sa11'], 
                               fault_rate = fault_rate,
                               fault_dist = fault_dist)
        # XB_fault_rate.extend(self.fc1.w2g.SAF_pos.total_faults_per_crossbar.flatten().tolist())
        # XB_fault_rate.extend(self.fc1.w2g.SAF_neg.total_faults_per_crossbar.flatten().tolist())
        # NO_OF_XBs.add_(self.fc1.crxb_col * self.fc1.crxb_row * 2)
        self.fc2 = crxb_Linear(50, 10, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF,
                               sa00_rate=saf_rate_dict['fc2']['sa00'],
                               sa01_rate=saf_rate_dict['fc2']['sa01'],
                               sa10_rate=saf_rate_dict['fc2']['sa10'],
                               sa11_rate=saf_rate_dict['fc2']['sa11'], 
                               fault_rate = fault_rate,
                               fault_dist = fault_dist)
        
        # XB_fault_rate.extend(self.fc2.w2g.SAF_pos.total_faults_per_crossbar.flatten().tolist())
        # XB_fault_rate.extend(self.fc2.w2g.SAF_neg.total_faults_per_crossbar.flatten().tolist())
        # NO_OF_XBs.add_(self.fc2.crxb_col * self.fc2.crxb_row * 2)
        # mean_faults= torch.mean(torch.tensor(XB_fault_rate))
        # var_faults = torch.var(torch.tensor(XB_fault_rate))
        # beta = torch.sqrt(mean_faults/var_faults)
        # print("Beta for the Cluster Distribution:", beta.item())
        # print("sum =", sum(XB_fault_rate))
        # print("Fault Density", sum(XB_fault_rate) * 100 /(128*128*NO_OF_XBs.item()))
        # # print(XB_fault_rate)
        self.bn1 = nn.BatchNorm2d(num_features=10)
        self.bn2 = nn.BatchNorm2d(num_features=20)

        self.bn3 = nn.BatchNorm1d(num_features=50)

        ##Determining B of the clustering fault distribution 

    # def forward_hook(self, module, input, output):
    #     self.forward_pass_count += 1
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        #quit(0)
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        x = x.view(-1, 320)
        #quit(0)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #print("end output", output)
        #quit(0)
        return output


no_of_forward_pass = 0
def forward_hook(model, input):
        global no_of_forward_pass
        if model.training:
            no_of_forward_pass += 1 


def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)

def train(model, device, criterion, optimizer, train_loader, epoch, batch_size):
    losses = AverageMeter()

    model.train()
    correct = 0
    # total_batches = len(train_loader)
    # train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):

        for name, module in model.named_modules():
            if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
                module._reset_delta()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        #print("target",target)
        # print(model.conv1.sa11_rate)
        output = model(data)
        #print("end output", output)
        loss = criterion(output, target)
        #quit(0)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.sampler.__len__(),
                       100. * batch_idx / len(train_loader), loss.item()))
        # if (epoch * total_batches + batch_idx) % 200 == 0 :
        #     print('No of batches processed: ', batch_idx)
        #     inject_fault_rate(model, dyn_fault_rate=0.00, dyn_xb_rate=1, prob_faults = [0.25,0.25,0.25,0.25])
        
        # with torch.no_grad():
            #     if epoch % 1 == 0:
            #         dyn_inject_fault_random_tile(model)
            #         fault_rate_dict = get_fault_rate(model, device)
            #         [print(key,': ',value) for key, value in fault_rate_dict.items()]

    print('\nTrain set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, train_loader.sampler.__len__(),
        100. * correct / train_loader.sampler.__len__()))

    return losses.avg


def validate(args, model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if args.ir_drop:
                print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
                    correct, val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1),
                             100. * correct / (val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1))))

        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__()))

        return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--crxb_size', type=int, default=128, help='corssbar size')
    parser.add_argument('--vdd', type=float, default=3.3, help='supply voltage')
    parser.add_argument('--gwire', type=float, default=0.0357,
                        help='wire conductacne')
    parser.add_argument('--gload', type=float, default=0.25,
                        help='load conductance')
    parser.add_argument('--gmax', type=float, default=0.000333,
                        help='maximum cell conductance')
    parser.add_argument('--gmin', type=float, default=0.000000333,
                        help='minimum cell conductance')
    parser.add_argument('--ir_drop', action='store_true', default=False,
                        help='switch to turn on ir drop analysis')
    parser.add_argument('--scaler_dw', type=float, default=1,
                        help='scaler to compress the conductance')
    parser.add_argument('--test', action='store_true', default=False,
                        help='switch to turn inference mode')
    parser.add_argument('--enable_noise', action='store_true', default=False,
                        help='switch to turn on noise analysis')
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable_ec_SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    parser.add_argument('--freq', type=float, default=10e6,
                        help='scaler to compress the conductance')
    parser.add_argument('--temp', type=float, default=300,
                        help='scaler to compress the conductance')
    parser.add_argument('--fault_dist', type=str, default="cluster",
                        help='fault distribution applied')
    parser.add_argument('--fault_rate', type=float, default=0.5,
                        help='fault rate per crossbar')


    args = parser.parse_args()

    # ddp_setup(rank,world_size)

    best_error = 0

    if args.ir_drop and (not args.test):
        warnings.warn("We don't recommend training with IR drop, too slow!")

    if args.ir_drop and args.test_batch_size > 150:
        warnings.warn("Reduce the batch size, IR drop is memory hungry!")

    if not args.test and args.enable_noise:
        raise KeyError("Noise can cause unsuccessful training!")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda:0,1" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    rank = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #load the network
    model = Net(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin, gwire=args.gwire, gload=args.gload,
                vdd=args.vdd, ir_drop=args.ir_drop, device=rank, scaler_dw=args.scaler_dw, freq=args.freq, temp=args.temp,
                enable_SAF=args.enable_SAF, enable_noise=args.enable_noise, enable_ec_SAF=args.enable_ec_SAF, fault_rate=args.fault_rate, fault_dist = args.fault_dist).to(rank)
    # if model.training:
    #     model.conv1.register_forward_pre_hook(forward_hook)
    #enable distributed model training
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert BatchNorm to SyncBatchNorm
    # model = DDP(model, device_ids=[rank])
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                           patience=2, verbose=True, threshold=0.5,
                                                           threshold_mode='rel', min_lr=1e-5)
    loss_log = []
    if not args.test:
        for epoch in range(args.epochs):
            print("epoch {0}, and now lr = {1:.4f}\n".format(epoch, optimizer.param_groups[0]['lr']))
            train_loss = train(model=model, device=rank, criterion=criterion,
                               optimizer=optimizer, train_loader=train_loader,
                               epoch=epoch, batch_size = args.batch_size)

            val_loss = validate(args=args, model=model, device=rank, criterion=criterion,
                                    val_loader=test_loader)

            scheduler.step(val_loss)
            # print("no of forward passes for an epoch", no_of_forward_pass)
            # break the training
            if optimizer.param_groups[0]['lr'] < ((scheduler.min_lrs[0] / scheduler.factor) + scheduler.min_lrs[0]) / 2:
                print("Accuracy not improve anymore, stop training!")
                break
            
            loss_log += [(epoch, train_loss, val_loss)]
            is_best = val_loss > best_error
            best_error = min(val_loss, best_error)
            
            filename = 'checkpoint_' + str(args.crxb_size) + 'test' + '.pth.tar'
            
            raw_model = model.module if hasattr(model, "module") else model
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': raw_model.state_dict(),
                'best_acc1': best_error,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=filename)
            

    elif args.test:
        modelfile = 'checkpoint_' + str(args.crxb_size) + '.pth.tar'
        if os.path.isfile(modelfile):
            print("=> loading checkpoint '{}'".format(modelfile))
            checkpoint = torch.load(modelfile)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'"
                  .format(modelfile))

            validate(args=args, model=model, device=rank, criterion=criterion,
                     val_loader=test_loader)
    # destroy_process_group()
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
    # world_size = int(2)
    # mp.spawn(main, args=(world_size,), nprocs=world_size)
