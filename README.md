# ArtTRT
TensorRT state of the art

## Results

### Today State

| Data Set           | Precision    | Workflow             | Metrics            | Platform          | Network           | Batch Size |
|--------------------|--------------|----------------------|--------------------|-------------------|-------------------|------------|
| ImageNet &#x2713;  | fp32 &#x2713;| PyTorch-ONNX &#x2713;| Accuracy &#x2713;  | RTX 3060 &#x2713; | ResNet18 &#x2713; | 1 &#x2713; |
|                    | fp16 &#x2713;| PyTorch Runtime      | Latency  &#x2713;  | Xavier            | MobileNet         | 32         |
|                    | int8 &#x2713;|                      | Throughput         |                   |                   | 64         |
|                    |              |                      | Model Size &#x2713;|                   |                   | 256 &#x2713;|

### Table of results for Batch Size 1

|  Model      | Latency (s)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|--------------|-----------|----------------------|---------------------|
| Vanilla     | 0.015        |45.75      |76.05                 |96.66                |
| TRT fp32    | 0.006        |68.17      |76.05                 |96.66                |
| TRT fp16    | 0.007        |23.88      |76.08                 |96.64                |
| TRT int8    | 0.006        |14.45      |75.98                 |96.61                |

### Table of results for Batch Size 256

|  Model      | Latency (s)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|--------------|-----------|----------------------|---------------------|
| Vanilla     | 1.284        |45.75      |76.05                 |96.66                |
| TRT fp32    | 0.905        |48.75      |76.02                 |96.67                |
| TRT fp16    | 0.899        |24.48      |76.02                 |96.64                |
| TRT int8    | 0.874        |12.66      |75.99                 |96.64                |

obs: Latency as, time per batch (of 256)

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
python build_trt.py --fp16
```
You may need to change the batch size and input size manually.


---

# Prerequizitos (Windows 10/11)

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