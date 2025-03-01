# ResNet-18 Training on CIFAR-10

This repository contains code for training a ResNet-18 model on the CIFAR-10 dataset, with various optimizations and experiments.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib (for plotting results)

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install torch torchvision matplotlib
```

## Running the Code

The main file `lab2.py` handles all experiments (C1-C7) through command-line arguments.

### Basic Usage

```bash
python lab2.py [options]
```

### Options

- `-d`, `--device`: Select device for training (`cuda` or `cpu`, default: `cpu`)
- `-dp`, `--datapath`: Path to dataset (default: `./data/`)
- `-w`, `--workers`: Number of workers for data loading (choices: 0, 1, 2, 4, 8, 12, 16, default: 2)
- `-op`, `--optimizer`: Optimizer for training (choices: `sgd`, `sgdnes`, `adagrad`, `adadelta`, `adam`, default: `sgd`)
- `-v`, `--verbose`: Print detailed logs on the console
- `-e`, `--experiment`: Experiment to run (choices: `c2`, `c3`, `c4`, `c5`, `c6`, `c7`, default: `c2`)
- `-nb`, `--no-batchnorm`: Disable batch normalization layers (for C7)
- `-p`, `--plot`: Generate and save plots (for experiments that produce graphs)
- `-profile`, `--profile `: Generate and save profiling for tensorboard(Extracredit )
### Running Specific Experiments

#### C2: Time Measurement
```bash
python lab2.py -e c2 -d cpu -w 2
```
<img width="409" alt="image" src="https://github.com/user-attachments/assets/8a1f0825-e20f-46de-b02d-f96bde745002" />

#### C3: I/O Optimization
```bash
python lab2.py -e c3 -d cpu -p
```

#### C4: Profiling
```bash
python lab2.py -e c4 -d cpu
```

#### C5: GPU vs CPU Comparison
```bash
python lab2.py -e c5 -p
```

#### C6: Optimizer Comparison
```bash
python lab2.py -e c6 -d cuda -w <optimal_workers> -p
```

#### C7: Without Batch Norm
```bash
python lab2.py -e c7 -d cuda -w <optimal_workers> -nb
```

### Running All Experiments
```bash
python lab2.py -e all -d cuda -p
```

## Output

Results will be printed to the console and, when the `-p` flag is used, plots will be saved in the `plots/` directory.
Profiling data saved to `profiles` directory
