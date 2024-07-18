# ArtTRT: Experimental Evaluation of Neural Networks Optimized for GPU Inference Using TensorRT.

## Introduction

This repository is built to conduct an experimental evaluation of TensorRT's optimization capabilities on GPUs of various ranges, from embedded systems to desktop computers, providing quantitative data on its advantages and limitations. The findings offer guidelines for optimizing customized algorithms in scientific and industrial applications beyond standardized benchmarking examples.

## Full Documentation

See the [Wiki](https://github.com/Juanx65/ArtTRT/wiki/) for full documentation, examples, operational details and other information.

## What does it do?


## Build

To build in a Jetson device, refer to the [Wiki](https://github.com/Juanx65/ArtTRT/wiki/)

### Prerequisites

* CUDA 12.4
* TensorRT 8.6.1.post1
* torch 2.2.2
* torchvision 0.17.2
* onnx 1.16.0
* requests 2.31.0
* scipy 1.13.0
* polygraphy 0.49.0
* onnx_opcounter 0.0.3
* ultralytics ( only to test on yolov8 )

```
$ git clone git@github.com:Juanx65/ArtTRT.git
$ cd ArtTRT/
$ virtualenv env
$ pip install -r requiements.txt
```

# Run Demo

Refer to `workflow.ipynb`.