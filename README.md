# ArtTRT: Experimental Evaluation of Neural Networks Optimized for GPU Inference Using TensorRT.

## Introduction

This repository is built to conduct an experimental evaluation of TensorRT's optimization capabilities on GPUs of various ranges, from embedded systems to desktop computers, providing quantitative data on its advantages and limitations. The findings offer guidelines for optimizing customized algorithms in scientific and industrial applications beyond standardized benchmarking examples.

## Full Documentation

See the [Wiki](https://github.com/Juanx65/ArtTRT/wiki/) for full documentation, examples, operational details and other information.

## What does it do?


## Build

To build in a Jetson device, refer to the [Wiki](https://github.com/Juanx65/ArtTRT/wiki/)

### Prerequisites

* Linux ( based on Ubuntu 22.04 LTS )
* virtualenv 
* CUDA 12.4
* ultralytics ( only to test on yolov8 )

```
$ git clone git@github.com:Juanx65/ArtTRT.git
$ cd ArtTRT/
$ virtualenv env
$ source env/bin/activate
$ pip install --no-cache-dir -r requirements.txt
```

# Run Demo

Refer to `workflow.ipynb`.
