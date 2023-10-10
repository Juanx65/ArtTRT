# ArtTRT
TensorRT state of the art

# Results
## Today State

| Data Set           | Precision    | Workflow             | Metrics            | Platform          | Network           | Batch Size |
|--------------------|--------------|----------------------|--------------------|-------------------|-------------------|------------|
| ImageNet &#x2713;  | fp32 &#x2713;| PyTorch-ONNX &#x2713;| Accuracy &#x2713;  | RTX 3060 &#x2713; | ResNet18 &#x2713; | 1 &#x2713; |
|                    | fp16 &#x2713;| PyTorch Runtime      | Latency  &#x2713;  | Xavier            | MobileNet &#x2713;|32  &#x2713;|
|                    | int8 &#x2713;|                      | Throughput         |                   | YOLOv8 &#x2713;   | 64 &#x2713;|
|                    |              |                      | Model Size &#x2713;|                   |                   |128 &#x2713;|
|                    |              |                      |                    |                   |                   |256 &#x2713;|

Note: 

* Results were obtained using a 50k validation image set from the ImageNet-1k dataset with the pretrained models available on torch.hub.

* We are using a warm-up for 10% of the batches to achieve a better latency estimation.

*  Latency shows the minimum / average / maximum time per batch after warm-up.

* For every engine of int8 precision with different batch size created with build_trt, you need to delete the `cache` file for the script to create one new with the correct batch size, in the future this will have a flag to restore cache option.

<details><summary> YOLOv8 </summary>

### Reference results
Results from the ultralyric github page https://github.com/ultralytics/ultralytics

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

<details><summary> YOLOv8n-cls </summary>

### Batch Size 1

</details>


<details><summary> YOLOv8n-cls </summary>

### Batch Size 1


</details>

</details>

<details><summary> MobileNet_V2 </summary>

### Batch Size 1

</details>

<details><summary>  ResNet </summary>

<details><summary> ResNet18 </summary>

### Batch Size 1

</details>

<details><summary> ResNet34 </summary>

### Batch Size 1

</details>

<details><summary> ResNet50 </summary>

### Batch Size 1

</details>

<details><summary> ResNet101 </summary> 

### Batch Size 1

</details>

<details><summary>  ResNet152 </summary> 

### Batch Size 1

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 7.7/8.4/20.6  |241.7      |82.34                 |95.92                |
| TRT fp32    | 13.1/14.4/20.8|243.3      |82.34                 |95.92                |
| TRT fp16    | 6.7/7.1/12.9  |122.3      |82.34                 |95.90                |

### Batch Size 32 

|  Model      | Latency (ms)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 141/144/141   |241.7      |82.34                 |95.93                |
| TRT fp32    | 72/78/119     |243.3      |82.34                 |95.92                |
| TRT fp16    | 29/31/52      |122.3      |82.34                 |95.90                |

### Batch Size 64

|  Model      | Latency (s)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 0.3/0.3/0.3   |241.7      |82.34                 |95.93                |
| TRT fp32    | 0.1/0.1/0.2   |243.3      |82.34                 |95.92                |
| TRT fp16    | 55/57/69 (ms) |122.3      |82.34                 |95.90                |


### Batch Size 128

|  Model      | Latency (s)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 0.5/0.6/0.7   |241.7      |82.38                 |95.93                |
| TRT fp32    | 0.3/0.3/0.3   |243. 3     |82.38                 |95.92                |
| TRT fp16    | 0.1/0.1/0.14  |122.3      |82.37                 |95.90                |

### Batch Size 256

|  Model      | Latency (s)   | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|---------------|-----------|----------------------|---------------------|
| Vanilla     | 1.1/1.1/1.3   |241.7      |82.38                 |95.93                |
| TRT fp32    | 0.6/0.6/0.6   |243.3      |82.38                 |95.92                |
| TRT fp16    | 0.2/0.2/0.2   |122.3      |82.37                 |95.90                |

</details>

</details>

---
# Validation

To validate the models ( vanilla and trt ) with a validation set of the ImageNet-1k dataset:

## Vanilla

```
python main.py -v --batch_size=1 --dataset='dataset/val' --network="resnet18"
```

YOLOv8 example:

```
python main.py -v --batch_size=1 --dataset='dataset/val' --network="yolo" --weights='weights/yolov8n-cls.pt'
```

## TensorRT optimization
```
python main.py -v --batch_size=1 --dataset='dataset/val' --network="resnet18" -trt
```

---
Notes:

* For the YOLOv8: First, download the pretrained weights in the classification (imageNet) section here https://github.com/ultralytics/ultralytics in the weights folder.

* We downloaded a part of the Imagnet Dataset from `https://huggingface.co/datasets/imagenet-1k/viewer/default/validation` and saved it in the `dataset/val` folder. 

* For the labels to function correctly, we utilize the script `format_dataset.py`. This script moves each image into its respective label folder, ensuring our code operates as expected. Ultimately, the dataset should adopt the following structure:

    ```
    dataset/val/
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

# Comparison

Here we compare the output value of the vanilla model vs the TensorRT optimizated model with the function numpy.isclose() as described in `https://ieeexplore.ieee.org/document/10074837` this paper.

 Note: For better performance, we use torch.isclose(), which performs the same function as the NumPy function.

```
python main.py -trt --compare --batch_size=1 --network="resnet18" -rtol=1e-2
```

To comapre using a validation dataset instead of the random generated inputs, you can use this

Note: Currently, comparing the MSE of the top 5 classes, as the isclose() approach in the paper didn't yield good results.

```
python main.py --batch_size=1 --network="resnet18" -trt -rtol=1e-3 --compare --val_dataset --dataset='val_images/'
```

Note: We use the numpy.isclose() function, which returns True or False based on the following condition:

```
 absolute(a - b) <= (atol + rtol * absolute(b)) 
```

In this equation, a represents the output of the vanilla model, b is the output of the TRT optimized model, atol is the absolute tolerance set to 1e-8, and rtol is the relative tolerance set to 1e-3. For the TRT optimized model with FP32 precision, we observed a non-equal percentage of 6.50% with a rtol of 1e-2. Note that this result may change upone re build of the engine.

---

<details><summary> Train on a Subset of ImageNet Dataset </summary>

As it is easyer to work with pre trained datasets, I stoped working with this...

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

</details>

---

# TensorRT Optimization

## Transform PyTorch to ONNX

To transform the pretrained weights `.pth` to `.onnx` format:

```
python onnx_transform.py --weights="weights/best.pth" --pretrained --network="resnet18"
```
Note: Here we are downloading the weights form torch.hub.load, we only inform the `--weights="weights/best.pth"` value to indicate where to save the onnx value later.

To transform your own weights, you can use:

```
python onnx_transform.py --weights="weights/best.pth"
```

## Create the TRT Engine

```
python build_trt.py --fp16 --input_shape -1 3 224 224
```
Note: -1 (for dynamic) is the batch size, 3 the number of channels, 224 is the height and width of the input.

## Profile with pytorch 

https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

to use this firs do a `pip uninstall tensorboard` then do `pip install torch-tb-profiler`


---

<details><summary>  Prerequisites and Installation (Windows 10/11) </summary>

## Prerequisites

* CUDA 12.2
* cudnn
* TensorRT 8.6
* pytorch
* ultralyrtics
* Microsoft C++ Build Tools (requiered to install pycuda)
* onnx

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

</details>

---

# References

* ResNet-ImageNet: https://github.com/jiweibo/ImageNet
* ImageNet subset: https://github.com/fastai/imagenette
* TensorRT functions (engine in utils): https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/models/engine.py
* TensorRT installation guide: https://developer.nvidia.com/nvidia-tensorrt-8x-download
* Microsoft C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
* val_image dataset: https://huggingface.co/datasets/imagenet-1k/viewer/default/validation
* pytorch profiler: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html