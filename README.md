# UXU: Ultimate eXtensible UVM

UXU is a multi-level swapping for GPGPU memory, which is inspired from [DRAGON](README.org.md). Using UXU, large-scale GPGPU applications exceeding GPGPU physical memory can be easily executed. 

## Getting Started

This project composes of multiple components. To install DRAGON, follow the
instructions in the [INSTALL.md](INSTALL.md) file. To compile or run provided
example applications, see the [README.md](examples/README.md) file in the
*examples* folder.

### Install UXU kernel driver
UXU kernel driver is a kind of modified NVidia UVM(Unified Virtual Memory) in the same way by DRAGON. NVidia 440.82 kernel source is used as a codebase for our driver. Below installation procedure assumes Ubuntu 18.04. 

1. Prepare linux kernel build environment
    Most 4.15.x would be fine. One of tested versions is 4.15.0.140

2. Build kernel modules
     ```
     # cd kernel_nvidia
     # make -C /lib/modules/4.15.0-140-generic/build M=$PWD
   ```
     After successful build, several kernel modules will be generated. Two modules `nvidia.ko` and `nvidia-uvm.ko` are required to run UXU.

### Install libuxu library and examples
   User-land library and examples can built with autotools.
   ```
  # ./configure
  # make
  ```
