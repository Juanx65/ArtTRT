# ArtTRT
TensorRT state of the art

## Results

### Today State

| Data Set           | Precision    | Workflow             | Metrics            | Platform          | Network           | Batch Size |
|--------------------|--------------|----------------------|--------------------|-------------------|-------------------|------------|
| ImageNet &#x2713;  | fp32 &#x2713;| PyTorch-ONNX &#x2713;| Accuracy &#x2713;  | RTX 3060 &#x2713; | ResNet18 &#x2713; | 1 &#x2713; |
|                    | fp16 &#x2713;| PyTorch Runtime      | Latency  &#x2713;  | Xavier            | MobileNet         | 32         |
|                    | int8         |                      | Throughput         |                   |                   | 64         |
|                    |              |                      | Model Size &#x2713;|                   |                   | 264        |

### Table of results for Batch Size 1

|  Model      | Latency (s)  | size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|--------------|-----------|----------------------|---------------------|
| Vanilla     | 0.015        |45.75      |76.05                 |96.66                |
| TRT fp32    | 0.006        |68.17      |76.05                 |96.66                |
| TRT fp16    | 0.007        |23.88      |76.08                 |96.64                |
| TRT int8    | 0.006        |14.45      |75.975                |96.611               |

---
# Train on a Subset of ImageNet Dataset

## Train Vanilla ResNet18

```
python main.py --dataset='dataset/' --batch_size='256' --epoch=90 --wd=1e-4 --momentum=0.9 --lr=0.001 --weights='weights/best.pth' -m
```

## Evaluate Vanilla ResNet18

```
python main.py --dataset='dataset/' --batch_size=256 --evaluate
```

## Evaluate TensorRT ResNet18

```
python main.py --dataset='dataset/' --batch_size=1 --evaluate --trt --weights='weights/best.engine'
```

---

# TensorRT Optimization

## Transform PyTorch to ONNX

To transform the trained weights `.pth` to `.onnx` format:

```
python onnx_transform.py
```

You may need to change the batch size and input size manually.

## Create the TRT Engine

```
python build_trt.py --fp16 --input_shape=[BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
```

---

# References

* ResNet-ImageNet: https://github.com/jiweibo/ImageNet
* ImageNet subset: https://github.com/fastai/imagenette
* TensorRT functions (engine in utils): https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/models/engine.py