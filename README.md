# Distributed-Training-Methods-for-Large-Models-and-Datasets

## Project Overview

This repository contains the code, experiments, and results for a comprehensive study on distributed training methods for large-scale machine learning models and datasets. The project explores the transition from CPU-based to GPU-based training, with a focus on distributed training techniques such as **data parallelism** and **model parallelism**. The experiments were conducted on six different models—**VGG11**, **NBoW**, **Seq2Seq**, **ConvAutoencoder**, **CNN**, and **MLP**—using both single-GPU and multi-GPU setups. The results demonstrate the effectiveness of distributed training in reducing training time while maintaining or improving model performance.

## Repository Structure

The repository is organized as follows:

```
C:.
│   Distributed Training Methods for Large Models and Datasets.pptx.pdf       ## Presentation of the results and methods used
│   distributed_ML_training.pdf         ## Full report of the work, including problem explanation, methods, and results
│
├───1 GPU                    ## Notebooks for running experiments on a single GPU
│       cnn_distribution.ipynb
│       ConvAutoencoder_distribution.ipynb
│       MLP_distribution.ipynb
│       NBoW_distribution.ipynb
│       seq2seq distribution.ipynb
│       vgg11-GPU1.ipynb
│
├───2 GPUs               ## Notebooks for running experiments on 2 GPUs
│       distributed-of-convolutionalautoencoder-training.ipynb
│       distributed-of-mlp-training.ipynb
│       distributed-of-nbow-training.ipynb
│       distributed-of-seq2seq-training.ipynb
│       distributed-of-vgg11-training.ipynb
│       project5-2gpus.ipynb
│
├───plot results            ## Results and plots from the experiments
│   │   plots.ipynb
│   │
│   ├───CNN               ## Results and plots for CNN experiments
│   │       CNN_20epochs_1GPU.json
│   │       CNN_20epochs_2GPU.json
│   │       CNN_training_metrics.png
│   │
│   ├───ConvAutoencoder
│   │       ConvAutoencoder_10epochs_1GPU.json
│   │       ConvAutoencoder_10epochs_2GPU.json
│   │       ConvAutoencoder_training_metrics.png
│   │
│   ├───MLP
│   │       MLP_10epochs_1GPU.json
│   │       MLP_10epochs_2GPU.json
│   │       MLP_10epochs_2GPU_DDP.json
│   │       MLP_training_metrics.png
│   │
│   ├───NBoW
│   │       NBoW_5epochs_1GPU.json
│   │       NBoW_5epochs_2GPU.json
│   │       NBoW_training_metrics.png
│   │
│   ├───Seq2Seq
│   │       seq2seq_10epochs_1GPU.json
│   │       seq2seq_10epochs_2GPU.json
│   │       Seq2Seq_training_metrics.png
│   │
│   └───VGG11
│       │   vgg11-3epoch-1gpu.json
│       │   vgg11-3epoch-2gpu.json
│       │   VGG11_training_metrics.png
│       │
│       └───.ipynb_checkpoints
└───the original note books     ## Original notebooks used to start the experiments
        Neural Bag of Words.ipynb
        Sequence to Sequence Learning with Neural Networks.ipynb
        mlp.ipynb
        vgg.ipynb
        cifar10_cnn_solution.ipynb
        Convolutional_Autoencoder_Solution.ipynb
```

## Key Features

- **Distributed Training Methods**: The project explores **data parallelism** and **model parallelism** for training large-scale machine learning models.
- **Six Models**: Experiments were conducted on **VGG11**, **NBoW**, **Seq2Seq**, **ConvAutoencoder**, **CNN**, and **MLP** models.
- **Single-GPU vs. Multi-GPU**: Comparison of training performance between single-GPU and multi-GPU setups.
- **Results and Plots**: Detailed results and visualizations for each model, including training time, loss, and accuracy metrics.
- **Original Notebooks**: The original notebooks used to start the experiments are included for reference.

## How to Use This Repository

1. **Single-GPU Experiments**: Navigate to the `1 GPU` folder to find notebooks for running experiments on a single GPU.
2. **Multi-GPU Experiments**: Navigate to the `2 GPUs` folder to find notebooks for running experiments on 2 GPUs.
3. **Results and Plots**: The `plot results` folder contains the results and plots for each experiment. Use the `plots.ipynb` notebook to visualize the results.
4. **Original Notebooks**: The `the original note books` folder contains the original notebooks used to start the experiments.

## Results and Findings

The experiments demonstrate that **distributed training** significantly reduces training time while maintaining or improving model performance in most cases. **Hybrid parallelism**, which combines data and model parallelism, shows particular promise in accelerating training for large models like MLP. However, challenges such as **synchronization overhead** and **model complexity** were observed, especially in models like CNN, where distributed training led to reduced performance.

For detailed results and analysis, refer to the [distributed_ML_training.pdf](distributed_ML_training.pdf) report and the [Distributed Training Methods for Large Models and Datasets.pptx.pdf](Distributed Training Methods for Large Models and Datasets.pptx.pdf) presentation.
