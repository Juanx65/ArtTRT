# ArtTRT
TensorRT state of the art

# Results
## Today State

| Data Set           | Precision    | Workflow             | Metrics            | Platform          | Network           | Batch Size |
|--------------------|--------------|----------------------|--------------------|-------------------|-------------------|------------|
| ImageNet &#x2713;  | fp32 &#x2713;| PyTorch-ONNX &#x2713;| Accuracy &#x2713;  | RTX 3060 &#x2713; | ResNet18 &#x2713; | 1 &#x2713; |
|                    | fp16 &#x2713;| PyTorch Runtime      | Latency  &#x2713;  | Xavier            | MobileNet &#x2713;|32  &#x2713;|
|                    | int8 &#x2713;|                      | Throughput         |                   |                   | 64 &#x2713;|
|                    |              |                      | Model Size &#x2713;|                   |                   |128 &#x2713;|
|                    |              |                      |                    |                   |                   |256 &#x2713;|

Note: Results were obtained using a 50k validation image set from the ImageNet-1k dataset with the pretrained models available on torch.hub.

Note: We are using a warm-up for 10% of the batches to achieve a better latency estimation.

Note: Latency shows the minimum / average / maximum time per batch after warm-up.

## MobileNet_V2
### Batch Size 1

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     |4.0/7.5/131.4   |13.92      |72.02                 |90.63                |
| TRT fp32    |1.0/3.1/95.8    |15.34      |72.02                 |90.61                |
| TRT fp16    |0.9/3.2/105.0   |10.64      |71.99                 |90.63                |
| TRT int8    |1.0 /3.3/104.0  |14.86      |72.02                 |90.62                |

### Batch Size 32

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     |28.0/48.4/372   |13.92      |72.02                 |90.63                |
| TRT fp32    |16.0/41.7/254   |15.04      |72.02                 |90.63                |
| TRT fp16    |15.0/41.8/265   |9.46       |72.00                 |90.64                |
| TRT int8    |16.0/41.6/239   |15.18      |72.03                 |90.63                |

### Batch Size 64

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     |54.3/101/404    |13.92      |72.02                 |90.63                |
| TRT fp32    |29.0/81.9/403   |14.84      |72.03                 |90.63                |
| TRT fp16    |26.0/82.7/414   |9.35       |71.99                 |90.63                |
| TRT int8    |31.0/83.0/392   |14.88      |72.03                 |90.63                |

### Batch Size 128

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     |112/162/513     |13.92      |72.06                 |90.64                |
| TRT fp32    |59.0/162/750    |14.94      |72.06                 |90.64                |
| TRT fp16    |50.0/160/692    |9.32       |72.04                 |90.64                |
| TRT int8    |61.0/161/651    |14.90      |72.06                 |90.64                |

### Batch Size 256

|  Model      | Latency (ms)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 309/358/677   |13.92      |72.06                 |90.64                |
| TRT fp32    | 168/357/1269  |15.36      |72.06                 |90.64                |
| TRT fp16    | 120/372/1218  |9.25       |72.04                 |90.67                |
| TRT int8    | 171/391/1317  |15.32      |72.08                 |90.65                |

## ResNet18
### Batch Size 1

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     |2.8 / 5.1 / 84.6|45.73      |69.76                 |89.00                |
| TRT fp32    |2.0 / 4.2 / 91.6|68.17      |69.75                 |89.09                |
| TRT fp16    |  1/ 2.6 / 99.2 |23.59      |69.74                 |89.09                |
| TRT int8    |0.5 / 2.6 /105.3|14.45      |68.85                 |88.45                |

### Batch Size 256

|  Model      | Latency (ms)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 373/407/494   |45.73      |69.80                 |89.10                |
| TRT fp32    | 300/329/598   |47.65      |69.80                 |89.10                |
| TRT fp16    | 138/330/1121  |24.34      |69.80                 |89.10                |
| TRT int8    | 107/343/1118  |14.10      |68.88                 |88.47                |

## ResNet34
### Batch Size 1

|  Model      | Latency (ms)      | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|-------------------|-----------|----------------------|---------------------|
| Vanilla     | 5.0 / 8.5 /  66.0 |45.75      |73.29                 |91.42                |
| TRT fp32    | 3.5 / 6.7 / 77.2  |132.63     |73.29                 |91.42                |
| TRT fp16    | 1.0 / 3.6 / 103.0 |44.40      |73.29                 |91.43                |
| TRT int8    | 3.0 / 5.3 / 86.0  |64.008     |72.62                 |91.06                |

### Batch Size 256

|  Model      | Latency (ms)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 534/613/761   |45.75      |73.34                 |91.43                |
| TRT fp32    | 455/487/584   |86.53      |73.35                 |91.43                |
| TRT fp16    | 182/322/898   |44.30      |73.37                 |91.44                |
| TRT int8    | 277/325/696   |50.10      |72.83                 |91.06                |

## ResNet50
### Batch Size 1

|  Model      | Latency (ms) | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|--------------|-----------|----------------------|---------------------|
| Vanilla     | 6.0/9.6/64.0 |100.14     |80.35                 |95.13                |
| TRT fp32    | 4.5/7.0/74.5 |110.46     |80.34                 |95.13                |
| TRT fp16    | 2.0/4.0/96.0 |51.83      |80.35                 |95.13                |
| TRT int8    | 4.0/6.0/8.6  |82.57      |2.55                  |6.34                 |

Note: The TRT int8 model was missing a lot of layers that may cause the results in the table

## ResNet101
### Batch Size 1

|  Model      | Latency (ms) | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|--------------|-----------|----------------------|---------------------|
| Vanilla     | 8.0/16.1/40.4|174.63     |81.68                 |95.66                |
| TRT fp32    | 8.0/11.7/55.0|215.20     |81.66                 |95.67                |
| TRT fp16    | 3.0/5.5/83.0 |89.03      |81.65                 |95.67                |
| TRT int8    | 7.0/10.2/73.0|188.00     |29.93                 |48.43                |

## ResNet152
### Batch Size 1

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|-----------|----------------------|---------------------|
| Vanilla     | 13.0/22.3/51.6 |236.00     |82.34                 |95.92                |
| TRT fp32    | 11.0/15.4/39.0 |313.93     |82.34                 |95.92                |
| TRT fp16    | 4.0/6.9/87.0   |119.78     |82.34                 |96.91                |
| TRT int8    | 11.0/15.0/45.0 |272.52     |20.37                 |35.97                |


---
# Compare and Validate on a pretrained model of the ImagNet-1k (2012)

## Comparison

Here we compare the output value of the vanilla model vs the TensorRT optimizated model with the function numpy.isclose() as described in `https://ieeexplore.ieee.org/document/10074837` this paper.

```
python .\main.py -trt --compare --batch_size=1 --network="resnet18" -rtol=1e-2
```

To comapre using a validation dataset instead of the random generated inputs, you can use this

```
python .\main.py --batch_size=1 --network="resnet18" -trt -rtol=1e-3 --compare --val_dataset --dataset='val_images/'
```

Note: We use the numpy.isclose() function, which returns True or False based on the following condition:

```
 absolute(a - b) <= (atol + rtol * absolute(b)) 
```

In this equation, a represents the output of the vanilla model, b is the output of the TRT optimized model, atol is the absolute tolerance set to 1e-8, and rtol is the relative tolerance set to 1e-3. For the TRT optimized model with FP32 precision, we observed a non-equal percentage of 6.50% with a rtol of 1e-2. Note that this result may change upone re build of the engine.


## Validation

To validate the models ( vanilla and trt ) with a validation set of the ImageNet-1k dataset:

### Vanilla

```
python .\main.py -v --batch_size=1 --dataset=val_images --network="resnet18"
```

### TensorRT optimization
```
python .\main.py -v --batch_size=1 --dataset=val_images --network="resnet18" -trt
```

---
Note: We downloaded a part of the Imagnet Dataset from `https://huggingface.co/datasets/imagenet-1k/viewer/default/validation` and saved it in the `val_images` folder. 

For the labels to function correctly, we utilize the script `format_dataset.py`. This script moves each image into its respective label folder, ensuring our code operates as expected. Ultimately, the dataset should adopt the following structure:

```
val_images/
│
└───n01440764/
    │
    ├── ILSVRC2012_val_00000293_n01440764.JPEG
    │
    ├── ...
│
└───nXXXXXXXX/
    │
    ├── ILSVRC2012_val_00000XXX_nXXXXXXXX.JPEG
    │
    ├── ...
│
└───...
```

---
# Train on a Subset of ImageNet Dataset

## Train Vanilla ResNet18

```
python main_own_trained_model.py --dataset='dataset/' --batch_size='256' --epoch=90 --wd=1e-4 --momentum=0.9 --lr=0.001 --weights='weights/best.pth' -m
```

## Evaluate Vanilla ResNet18

```
python main_own_trained_model.py --dataset='dataset/' --batch_size=256 --evaluate
```

## Evaluate TensorRT ResNet18

```
python main_own_trained_model.py --dataset='dataset/' --batch_size=1 --evaluate --trt --weights='weights/best.engine'
```

---

# TensorRT Optimization

## Transform PyTorch to ONNX

To transform the pretrained weights `.pth` to `.onnx` format:

```
python onnx_transform.py --batch_size=1 --weights="weights/best.pth" --pretrained --network="resnet18"
```
Note: Here we are downloading the weights form torch.hub.load, we only inform the `--weights="weights/best.pth"` value to indicate where to save the onnx value later.

To transform your own weights, you can use:

```
python onnx_transform.py --batch_size=1 --weights="weights/best.pth"
```

## Create the TRT Engine

```
python build_trt.py --fp16 --input_shape 256 3 224 224
```
Note: 256 is the batch size, 3 the number of channels, 224 is the height and width of the input.


---

# Prerequisites (Windows 10/11)

* CUDA 12.2
* cudnn
* TensorRT 8.6
* pytorch
* Microsoft C++ Build Tools (requiered to install pycuda)
* Pycuda
* onnx
* cv2

## TensorRT Installation

### Windows

Follow the installation guide at `https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html`.

#### or

TensorRT for Windows can only be installed via ZIP File installation:

* First, install the latest CUDA version for your device. Then, download TensorRT 8.x from this link: `https://developer.nvidia.com/nvidia-tensorrt-8x-download`.

* Unzip the `TensorRT-8.x.x.x.Windows10.x86_64.cuda-x.x.zip` file to the location of your choice. Where:
    * `8.x.x.x` is your TensorRT version
    * `cuda-x.x` is your CUDA version (either 11.8 or 12.0)

* Add the TensorRT library files to your system PATH (add `<installpath>/lib` to your system PATH).

* If you are using an environment like `virtualenv`, make sure to install the pip package located inside the previously installed TensorRT files:

    Install one of the TensorRT Python wheel files from `<installpath>/python` (replace `cp3x` with the desired Python version, for example, `cp310` for Python 3.10):

    ```bash
    python.exe -m pip install tensorrt-*-cp3x-none-win_amd64.whl
    ```

---

# References

* ResNet-ImageNet: https://github.com/jiweibo/ImageNet
* ImageNet subset: https://github.com/fastai/imagenette
* TensorRT functions (engine in utils): https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/models/engine.py
* TensorRT installation guide: https://developer.nvidia.com/nvidia-tensorrt-8x-download
* Microsoft C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
* val_image dataset: https://huggingface.co/datasets/imagenet-1k/viewer/default/validation