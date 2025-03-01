import argparse
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Custom warnings
def custom_formatwarnings(msg, *args, **kwargs):
    return 'WARNING: ' + str(msg) + '\n'

warnings.formatwarning = custom_formatwarnings

# Argument parser
parser = argparse.ArgumentParser(
    prog='python lab2.py',
    description='ResNet-18 training on CIFAR-10 dataset',
    epilog='------'
)

parser.add_argument('-d', '--device', choices=['cuda', 'cpu'], default='cpu',
                    required=False, dest='device', help='select the device for training (default: cpu)')
parser.add_argument('-dp', '--datapath', default='./data/', required=False,
                    dest='datapath', help='select the dataset path (default: ./data/)')
parser.add_argument('-w', '--workers', type=int, choices=[0, 1, 2, 4, 8, 12, 16, 20], default=2,
                    required=False, dest='workers', help='number of workers for data loading (default: 2)')
parser.add_argument('-op', '--optimizer', choices=['sgd', 'sgdnes', 'adagrad', 'adadelta', 'adam'],
                    default='sgd', required=False, dest='optimizer', help='optimizer for training (default: sgd)')
parser.add_argument('-v', '--verbose', action='store_true', required=False,
                    dest='verbose', help='print all logs on the console')
parser.add_argument('-e', '--experiment', choices=['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'all'], 
                    default='c2', required=False, dest='experiment', 
                    help='experiment to run (default: c2)')
parser.add_argument('-nb', '--no-batchnorm', action='store_true', required=False,
                    dest='no_batchnorm', help='disable batch normalization layers (for C7)')
parser.add_argument('-p', '--plot', action='store_true', required=False,
                    dest='plot', help='generate and save plots')
parser.add_argument('--profile', action='store_true', 
                    help='Enable PyTorch profiler and generate traces')

args = parser.parse_args()

# Handle datapath
if os.path.exists(args.datapath):
    datapath = args.datapath
else:
    datapath = './data/'
    warnings.warn(f"'{args.datapath}' doesn't exist, defaulting to './data/' datapath!")

# Handle device
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    warnings.warn("cuda is not available, running on cpu instead!")

# Create plots directory if it doesn't exist
if args.plot:
    os.makedirs('plots', exist_ok=True)

# ResNet-18 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if use_bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet-18 Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_bn=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet-18
def ResNet18(use_bn=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_bn=use_bn)

# DataLoader
def get_data_loader(datapath, workers, batch_size=128):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return train_loader


def count_parameters_and_gradients(use_bn=True):
    model = ResNet18(use_bn=use_bn)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable Parameters: {total_params}")
    
    # Dummy forward/backward pass to generate gradients
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (1,))
    output = model(dummy_input)
    loss = F.cross_entropy(output, dummy_target)
    loss.backward()
    
    total_gradients = sum(p.grad.numel() for p in model.parameters() if p.requires_grad)
    print(f"Gradients: {total_gradients}\n")

# Training function with accurate timing (C2)
def train_with_timing(model, train_loader, optimizer, criterion, device, verbose):
    model.train()
    
    # Results to return
    results = {
        'epoch_data_loading_time': [],
        'epoch_training_time': [],
        'epoch_total_time': [],
        'epoch_loss': [],
        'epoch_accuracy': []
    }
    
    for epoch in range(1, 6):  # 5 epochs
        running_loss = 0.0
        correct = 0
        total = 0

        # Timing variables
        epoch_start_time = time.time()
        data_loading_time = 0.0
        training_time = 0.0
        
        # For measuring data loading time accurately
        dataloader_iter = iter(train_loader)
        batch_idx = 0
        total_batches = len(train_loader)
        
        while batch_idx < total_batches:
            # Measure data loading time
            data_load_start = time.time()
            try:
                inputs, targets = next(dataloader_iter)
            except StopIteration:
                break
            data_loading_time += time.time() - data_load_start
            
            # Measure training time (including device transfer)
            training_start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_time += time.time() - training_start

            # Calculate accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if verbose and batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/5], Batch [{batch_idx}/{total_batches}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100. * correct / total:.2f}%')
            
            batch_idx += 1

        # Calculate total epoch time
        epoch_total_time = time.time() - epoch_start_time
        avg_loss = running_loss / total_batches
        accuracy = 100. * correct / total

        # Store results
        results['epoch_data_loading_time'].append(data_loading_time)
        results['epoch_training_time'].append(training_time)
        results['epoch_total_time'].append(epoch_total_time)
        results['epoch_loss'].append(avg_loss)
        results['epoch_accuracy'].append(accuracy)

        # Print timing results
        print('\n' + '='*50)
        print(f'Epoch [{epoch}/5] Summary:')
        print(f'Data Loading Time: {data_loading_time:.2f} seconds')
        print(f'Training Time: {training_time:.2f} seconds')
        print(f'Total Epoch Time: {epoch_total_time:.2f} seconds')
        print(f'Epoch Loss: {avg_loss:.4f}')
        print(f'Epoch Accuracy: {accuracy:.2f}%')
        print('='*50 + '\n')
    
    return results

# C3: I/O Optimization experiment
def run_experiment_c3(device, datapath, plot=False):
    print("\n== Running Experiment C3: I/O Optimization ==")
    
    # Workers to test
    workers_list = [0, 4, 8, 12, 16 ,20]
    loading_times = []
    
    for workers in workers_list:
        print(f"\nTesting with {workers} workers...")
        
        # Get data loader with specific number of workers
        train_loader = get_data_loader(datapath, workers)
        
        # Create model
        model = ResNet18().to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        # Train and measure timing
        results = train_with_timing(model, train_loader, optimizer, criterion, device, False)
        
        # Calculate average data loading time
        avg_loading_time = sum(results['epoch_data_loading_time']) / 5
        loading_times.append(avg_loading_time)
        print(f"Average data loading time with {workers} workers: {avg_loading_time:.2f} seconds")
    
    # Find best number of workers
    best_index = loading_times.index(min(loading_times))
    best_workers = workers_list[best_index]
    print(f"\nBest performance achieved with {best_workers} workers")
    
    # Plot results if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(workers_list, loading_times, 'o-', linewidth=2)
        plt.xlabel('Number of Workers')
        plt.ylabel('Average Data Loading Time (seconds)')
        plt.title('Data Loading Time vs. Number of Workers')
        plt.grid(True)
        plt.savefig('plots/c3_io_optimization.png')
        plt.close()
        print("Plot saved to plots/c3_io_optimization.png")
    
    return best_workers

# C4: Profiling experiment
def run_experiment_c4(device, datapath, best_workers):
    print("\n== Running Experiment C4: Profiling ==")
    
    # Test with 1 worker
    print("\nTesting with 1 worker...")
    train_loader_1 = get_data_loader(datapath, 1)
    model_1 = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.SGD(model_1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    results_1 = train_with_timing(model_1, train_loader_1, optimizer_1, criterion, device, False)
    
    # Test with best number of workers
    print(f"\nTesting with {best_workers} workers...")
    train_loader_best = get_data_loader(datapath, best_workers)
    model_best = ResNet18().to(device)
    optimizer_best = optim.SGD(model_best.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    results_best = train_with_timing(model_best, train_loader_best, optimizer_best, criterion, device, False)
    
    # Calculate averages
    avg_loading_1 = sum(results_1['epoch_data_loading_time']) / 5
    avg_training_1 = sum(results_1['epoch_training_time']) / 5
    avg_total_1 = sum(results_1['epoch_total_time']) / 5
    
    avg_loading_best = sum(results_best['epoch_data_loading_time']) / 5
    avg_training_best = sum(results_best['epoch_training_time']) / 5
    avg_total_best = sum(results_best['epoch_total_time']) / 5
    
    # Print comparison
    print("\nComparison between 1 worker and best number of workers:")
    print(f"1 Worker - Data Loading: {avg_loading_1:.2f}s, Training: {avg_training_1:.2f}s, Total: {avg_total_1:.2f}s")
    print(f"{best_workers} Workers - Data Loading: {avg_loading_best:.2f}s, Training: {avg_training_best:.2f}s, Total: {avg_total_best:.2f}s")
    print(f"Speedup - Data Loading: {avg_loading_1/avg_loading_best:.2f}x, Training: {avg_training_1/avg_training_best:.2f}x, Total: {avg_total_1/avg_total_best:.2f}x")
    
    # Analysis
    if avg_loading_1 > avg_loading_best:
        print("\nAnalysis: Using multiple workers significantly reduces data loading time, as it allows for parallel data processing and prefetching.")
    else:
        print("\nAnalysis: Using multiple workers did not improve data loading time, which may indicate I/O bottlenecks or CPU limitations.")
    
    return results_1, results_best

# C5: GPU vs CPU experiment (Modified Order)
def run_experiment_c5(datapath, best_workers, plot=False):
    print("\n== Running Experiment C5: GPU vs CPU Comparison ==")
    
    # Shared DataLoader
    train_loader = get_data_loader(datapath, best_workers)
    results_gpu = None
    
    # Test on GPU first if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        model_gpu = ResNet18().to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer_gpu = optim.SGD(model_gpu.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        results_gpu = train_with_timing(model_gpu, train_loader, optimizer_gpu, criterion, 'cuda', False)
    else:
        print("\nGPU not available - skipping GPU test")

    # Always test on CPU
    print("\nTesting on CPU...")
    model_cpu = ResNet18().to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    results_cpu = train_with_timing(model_cpu, train_loader, optimizer_cpu, criterion, 'cpu', False)

    # Compare if GPU test was done
    if results_gpu:
        avg_total_cpu = sum(results_cpu['epoch_total_time']) / 5
        avg_total_gpu = sum(results_gpu['epoch_total_time']) / 5
        
        print("\nCPU vs GPU Comparison:")
        print(f"GPU Average Epoch Time: {avg_total_gpu:.2f} seconds")
        print(f"CPU Average Epoch Time: {avg_total_cpu:.2f} seconds")
        print(f"Speedup: {avg_total_cpu/avg_total_gpu:.2f}x")

        if plot:
            # Bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(['GPU', 'CPU'], [avg_total_gpu, avg_total_cpu])
            plt.ylabel('Average Epoch Time (seconds)')
            plt.title('GPU vs CPU Training Time')
            plt.savefig('plots/c5_gpu_vs_cpu.png')
            plt.close()
            
            # Line plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 6), results_gpu['epoch_total_time'], 'o-', label='GPU')
            plt.plot(range(1, 6), results_cpu['epoch_total_time'], 'o-', label='CPU')
            plt.xlabel('Epoch')
            plt.ylabel('Total Time (seconds)')
            plt.title('GPU vs CPU Epoch Times')
            plt.legend()
            plt.grid(True)
            plt.savefig('plots/c5_gpu_vs_cpu_per_epoch.png')
            plt.close()
            
            print("Plots saved to plots/c5_gpu_vs_cpu.png and plots/c5_gpu_vs_cpu_per_epoch.png")

    return results_gpu if results_gpu else results_cpu

# C6: Optimizer comparison
def run_experiment_c6(device, datapath, best_workers, plot=False):
    print("\n== Running Experiment C6: Optimizer Comparison ==")
    
    optimizers = ['sgd', 'sgdnes', 'adagrad', 'adadelta', 'adam']
    results_dict = {}
    
    for opt_name in optimizers:
        print(f"\nTesting optimizer: {opt_name}")
        
        # Get data loader
        train_loader = get_data_loader(datapath, best_workers)
        
        # Create model
        model = ResNet18().to(device)
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Configure optimizer
        if opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        elif opt_name == 'sgdnes':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif opt_name == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
        elif opt_name == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
        elif opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
        
        # Train and measure timing
        results = train_with_timing(model, train_loader, optimizer, criterion, device, False)
        results_dict[opt_name] = results
        
        # Calculate averages
        avg_training_time = sum(results['epoch_training_time']) / 5
        avg_loss = sum(results['epoch_loss']) / 5
        avg_accuracy = sum(results['epoch_accuracy']) / 5
        
        print(f"Results for {opt_name.upper()}:")
        print(f"Average Training Time: {avg_training_time:.2f} seconds")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
    
    # Plot results if requested
    if plot:
        # Prepare data for plotting
        opt_names = list(results_dict.keys())
        avg_losses = [sum(results_dict[opt]['epoch_loss']) / 5 for opt in opt_names]
        avg_accuracies = [sum(results_dict[opt]['epoch_accuracy']) / 5 for opt in opt_names]
        avg_times = [sum(results_dict[opt]['epoch_training_time']) / 5 for opt in opt_names]
        
        # Plot loss comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(opt_names, avg_losses)
        plt.ylabel('Average Loss')
        plt.title('Optimizer Loss Comparison')
        
        # Plot accuracy comparison
        plt.subplot(1, 2, 2)
        plt.bar(opt_names, avg_accuracies)
        plt.ylabel('Average Accuracy (%)')
        plt.title('Optimizer Accuracy Comparison')
        
        plt.tight_layout()
        plt.savefig('plots/c6_optimizer_performance.png')
        plt.close()
        
        # Plot training time comparison
        plt.figure(figsize=(10, 6))
        plt.bar(opt_names, avg_times)
        plt.ylabel('Average Training Time (seconds)')
        plt.title('Optimizer Training Time Comparison')
        plt.savefig('plots/c6_optimizer_time.png')
        plt.close()
        
        print("Plots saved to plots/c6_optimizer_performance.png and plots/c6_optimizer_time.png")
    
    return results_dict

# C7: Without batch norm
def run_experiment_c7(device, datapath, best_workers):
    print("\n== Running Experiment C7: Without Batch Normalization ==")
    
    # Train with batch norm
    print("\nTraining with batch normalization...")
    train_loader = get_data_loader(datapath, best_workers)
    model_with_bn = ResNet18(use_bn=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    results_with_bn = train_with_timing(model_with_bn, train_loader, optimizer_with_bn, criterion, device, False)
    
    # Train without batch norm
    print("\nTraining without batch normalization...")
    model_no_bn = ResNet18(use_bn=False).to(device)
    optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    results_no_bn = train_with_timing(model_no_bn, train_loader, optimizer_no_bn, criterion, device, False)
    
    # Calculate averages
    avg_loss_with_bn = sum(results_with_bn['epoch_loss']) / 5
    avg_accuracy_with_bn = sum(results_with_bn['epoch_accuracy']) / 5
    
    avg_loss_no_bn = sum(results_no_bn['epoch_loss']) / 5
    avg_accuracy_no_bn = sum(results_no_bn['epoch_accuracy']) / 5
    
    # Print comparison
    print("\nComparison between with and without batch normalization:")
    print(f"With Batch Norm - Average Loss: {avg_loss_with_bn:.4f}, Average Accuracy: {avg_accuracy_with_bn:.2f}%")
    print(f"Without Batch Norm - Average Loss: {avg_loss_no_bn:.4f}, Average Accuracy: {avg_accuracy_no_bn:.2f}%")
    
    return results_with_bn, results_no_bn

# Main function
def main():
    if args.verbose:
        print("\n== Selected options :")
        print("Device    :", device)
        print("Data Path :", datapath)
        print("Workers   :", args.workers)
        print("Optimizer :", args.optimizer)
        print("Experiment:", args.experiment)

    
    count_parameters_and_gradients(use_bn=not args.no_batchnorm)

    # Run the selected experiment
    if args.experiment == 'c2' or args.experiment == 'all':
        # C2: Time Measurement
        print("\n== Running Experiment C2: Time Measurement ==")
        
        # DataLoader
        train_loader = get_data_loader(datapath, args.workers)

        # Model
        model = ResNet18(use_bn=not args.no_batchnorm).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == 'sgdnes':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

        # Train with timing
        results = train_with_timing(model, train_loader, optimizer, criterion, device, args.verbose)
        
        if args.experiment == 'all':
            # Continue with other experiments
            best_workers = run_experiment_c3(device, datapath, args.plot)
            run_experiment_c4(device, datapath, best_workers)
            run_experiment_c5(datapath, best_workers, args.plot)
            run_experiment_c6(device, datapath, best_workers, args.plot)
            run_experiment_c7(device, datapath, best_workers)
    
    elif args.experiment == 'c3':
        run_experiment_c3(device, datapath, args.plot)
    
    elif args.experiment == 'c4':
        best_workers = args.workers
        run_experiment_c4(device, datapath, best_workers)
    
    elif args.experiment == 'c5':
        best_workers = args.workers
        run_experiment_c5(datapath, best_workers, args.plot)
    
    elif args.experiment == 'c6':
        best_workers = args.workers
        run_experiment_c6(device, datapath, best_workers, args.plot)
    
    elif args.experiment == 'c7':
        best_workers = args.workers
        run_experiment_c7(device, datapath, best_workers)
    
    print("\n== Finished ==")

if __name__ == '__main__':
    main()