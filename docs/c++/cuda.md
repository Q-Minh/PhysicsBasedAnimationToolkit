# CUDA Installation Guide

## What is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. 
It allows developers to use GPUs for general-purpose processing, also known as GPU computing. CUDA is designed to work on NVIDIA GPUs.

## How to Install CUDA

### 1. **Windows Installation**

Follow these steps to install CUDA on Windows:

1. **Check GPU Compatibility**
   Make sure your system has an NVIDIA GPU that supports CUDA. You can check the list of supported GPUs on the [CUDA-Enabled GPUs page](https://developer.nvidia.com/cuda-gpus).

2. **Download the CUDA Toolkit**
   Visit the official NVIDIA CUDA Toolkit page:
   [Download CUDA Toolkit for Windows](https://developer.nvidia.com/cuda-toolkit-archive)

3. **Install CUDA Toolkit**
   Follow the installation instructions provided by NVIDIA:
   [Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

4. **Verify Installation**
   Once installed, verify CUDA by running the following command in the command prompt:
   ```bash
   nvcc --version
   ```

### 2. **Linux Installation**

To install CUDA on Linux, follow these steps:

1. **Check GPU Compatibility**
   Ensure your system has a supported NVIDIA GPU. Check the list of supported GPUs here:  
   [CUDA-Enabled GPUs](https://developer.nvidia.com/cuda-gpus)

2. **Download CUDA Toolkit**
   Visit the official NVIDIA CUDA Toolkit page:  
   [Download CUDA Toolkit for Linux](https://developer.nvidia.com/cuda-toolkit-archive)

3. **Install CUDA Toolkit**
   Follow the installation guide for Linux:  
   [Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

4. **Verify Installation**
   After installation, verify by running:
   ```bash
   nvcc --version
   ```

### 3. **macOS Users**

As of now, **CUDA is not supported on macOS**. NVIDIA stopped supporting CUDA for macOS after macOS 10.13 (High Sierra). If you are using a Mac, you won't be able to run CUDA applications natively, as Apple transitioned to its own GPUs (M1, M2) and discontinued official NVIDIA support.

However, if you need to work with CUDA, here are some alternatives:
- **Use a Cloud Service**: Platforms like [Google Colab](https://colab.research.google.com/) or [AWS EC2 with GPU instances](https://aws.amazon.com/ec2/instance-types/g4/) provide remote access to GPUs with CUDA support.
- **Use a Linux or Windows Virtual Machine (VM)**: Run a Linux or Windows VM on your Mac using software like [VirtualBox](https://www.virtualbox.org/) or [Parallels Desktop](https://www.parallels.com/), and install CUDA inside the VM (performance may vary).

For more details on CUDA-supported platforms and GPUs, visit:  
[NVIDIA CUDA Toolkit Official Site](https://developer.nvidia.com/cuda-toolkit)