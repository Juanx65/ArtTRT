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
* python 3.10
* virtualenv 
* CUDA 12.4
* ultralytics ( only to test on yolov8 )

```
$ git clone git@github.com:Juanx65/ArtTRT.git
$ cd ArtTRT/
$ virtualenv -p /usr/bin/python3.10 env
$ source env/bin/activate
$ pip install --no-cache-dir --use-pep517 -r requirements.txt
```

# Run Demo

Refer to `workflow.ipynb`.

# Links to datasets used on this work

* [Validation dataset ImageNet-1k (ILSVRC 2012)](https://usmcl-my.sharepoint.com/:f:/g/personal/juan_aguileraca_sansano_usm_cl/En76lQlPb6RFid5DN5gGTUMBDl7xnLnIa3PTcizcG4em0A?e=0hFf4u)
* [Subset of the validation ImageNet-1k (ILSVRC 2012)](https://usmcl-my.sharepoint.com/:f:/g/personal/juan_aguileraca_sansano_usm_cl/EqmQr04DEg9AhhWLpJcTYfcBzfi-r3T2WxN-PTH0abdDng?e=e99n3D)
* [Calibration dataset ImageNet-1k](https://usmcl-my.sharepoint.com/:f:/g/personal/juan_aguileraca_sansano_usm_cl/EkCOXMTdrGpHhf1tCMpJF3YBPtNcBoODugY34J2VDpiL4g?e=wVOFUq)
