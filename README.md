# ArtTRT
Estado del arte de TensorRT

---


# Train on a subset of ImagNet  Dataset

## Train Vanilla ResNet18

```
python main.py --dataset='dataset/' --batch_size='256' --epoch=90 --wd=1e-4 --momentum=0.9 --lr=0.001 --weights='weights/best.pth' -m
```

## Eval Vanilla ResNet18

```
python main.py --dataset='dataset/' --batch_size=256 --evaluate
```

## Evaluate TensorRT ResNet18

```
python main.py --dataset='dataset/' --batch_size=1 --evaluate -trt
```

---

# TensorRT Optimization

## Transform pytorch to onnx

To transform the trained weights `.pth` to `.onnx` format:

```
python onnx_transform.py
```

you may need to change the batch size and input size manually

## Create the TRT Engine

```
python build_trt.py --fp16 --input_shape=[BATCH_SIZE,CHANNELS, HEIGHT,WIDTH]
```

---
# REFS

* resnet-imagnet: https://github.com/jiweibo/ImageNet
* imagnet subset: https://github.com/fastai/imagenette
* TensorRT functions (engine in utils): https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/models/engine.py