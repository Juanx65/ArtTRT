# ArtTRT
TensorRT state of the art

## Notes:

Avances: `outputs/avances.md`
Tablas resumen: `outputs/talbe_outputs/`

## Prerequisites

* CUDA 12.4
* cudnn?
* TensorRT 8.6.1.post1
* torch 2.2.2
* torchvision 0.17.2
* onnx 1.16.0
* requests 2.31.0
* scipy 1.13.0

to counnt layers and parameters:

* polygraphy 0.49.0
* onnx_opcounter 0.0.3

optionals

* ultralytics ( to test on yolov8 )

# Results

![Device part.](/outputs/img_readme/inference_time_bar_all.png)

## Today State

| Data Set           | Precision    | Workflow             | Metrics            | Platform          | Network           | Batch Size |
|--------------------|--------------|----------------------|--------------------|-------------------|-------------------|------------|
| ImageNet &#x2713;  | fp32 &#x2713;| PyTorch-ONNX &#x2713;| Accuracy &#x2713;  | RTX 3060 &#x2713; | ResNet &#x2713;   | 1 &#x2713; |
|                    | fp16 &#x2713;| PyTorch Runtime      | Latency  &#x2713;  | Xavier            | MobileNet &#x2713;|32  &#x2713;|
|                    | int8 &#x2713;|                      | Throughput&#x2713; |                   | YOLOv8 &#x2713;   | 64 &#x2713;|
|                    |              |                      | Model Size &#x2713;|                   |                   |128 &#x2713;|
|                    |              |                      |                    |                   |                   |256 &#x2713;|

Note: 

* Validation results were obtained using a 50k validation image set from the ImageNet-1k dataset with the pretrained models available on torch.hub.

* We are using a warm-up for 10% of the dataset to achieve a better latency estimation.

* Latency-all shows the average and maximum time per batch after the warm-up, accounting for both CPU-GPU communication and model processing time.

* Latency-model displays the average and maximum time per batch following the model's warm-up.

* For every engine of int8 precision with different batch size created with build_trt, you need to delete the `cache` file for the script to create one new with the correct batch size, in the future this will have a flag to restore cache option.

<details><summary><span style="font-size:1.8em;">YOLOv8</span></summary>

### Reference results
Results from the ultralyric github page https://github.com/ultralytics/ultralytics

<details><summary> YOLOv8n-cls </summary>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| # layers  | # parameters|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|-----------|-------------|
| Vanilla         |   1.9 / 10.2   |  1.5 / 9.7       | 5.3       | 65.96                | 86.55               | 332       | 2715880     |
| TRT fp32        |     0.9 / 2.6  |  0.5 / 1.7       | 12.8      | 65.96                | 86.56               |81         | 2711472     |
| TRT fp16        |    0.8 / 5.3   |  0.4 / 4.8       |   6.2     | 65.96                | 86.55               |59         | 2711472     |
| TRT int8        |    0.8 / 3.3   |  0.3 / 2.7       | 5.3       | 63.38                | 84.63               |51         | 2711472     |

</details>


<details><summary> YOLOv8x-cls </summary>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |   5.9 / 13.4   |  5.5 / 12.1      | 109.7     | 78.66                | 94.29               |
| TRT fp32        |    4.3 / 9.2   |  3.9 / 7.0       | 264.0     | 78.65                | 94.28               |
| TRT fp16        |   1.8 / 3.8    |  1.3 / 2.6       | 111.7     | 78.66                | 94.29               |
| TRT int8        |     1.4 / 4.7  |  1.0 / 4.2       | 59.6      | 77.38                | 93.70               |

</details>

</details>

<details><summary><span style="font-size:1.8em;">MobileNet V2</span></summary>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |   2.8 / 11.5   |   2.3 / 11.0     | 13.6      | 72.02                | 90.63               |
| TRT fp32        |    1.0 / 2.5   |  0.5 / 1.6       | 14.1      | 72.02                | 90.62               |
| TRT fp16        |    0.8 / 4.1   |  0.4 / 3.6       | 8.5       | 71.99                | 90.62               |
| TRT int8        |    0.8 / 2.2   |  0.4 / 1.7       | 5.9       | 71.44                | 90.38               |

</details>

<details><summary><span style="font-size:1.8em;">ResNet</span></summary>

<details><summary> ResNet18 </summary>

### Inference Throughput
![Throughput stats.](/outputs/img_readme/inference_throughput_vs_batch_size_resnet18.png)


### GPU memory usage and SM Utilization over time
256 Vanilla            |  256 fp16 TRT
:-------------------------:|:-------------------------:
![](/outputs/img_readme/gpu_metrics_resnet18_256_vanilla.png)|![](/outputs/img_readme/gpu_metrics_resnet18_256_trt.png)


<details><summary>  Comparisons of ResNet18 Vanilla vs TRT Results </summary> 

* With relative tolearance of 1e-3 over 5k random inputs

| Vanilla VS | equality (%) |
|------------|--------------|
| TRT fp32   | 71.09        |
| TRT fp16   | 20.73        |
| TRT int8   |  1.12        |

*  TOP 10 MAE Error and disagriment % Comparison (over 5k examples)

|  Rank  | MAE / Disg  fp32 | MAE / Disg  fp16 | MAE / Disg  int8 |
|--------|------------------|------------------|------------------|
|      1 | 0.0017 / 0.06    | 0.0089 / 0.32    | 0.1062 / 6.18    |
|      2 | 0.0008 / 0.46    | 0.0043 / 1.94    | 0.0933 / 29.50   |
|      3 | 0.0011 / 0.72    | 0.0046 / 3.30    | 0.0899 / 48.56   |
|      4 | 0.0011 / 0.50    | 0.0045 / 2.96    | 0.0821 / 50.18   |
|      5 | 0.0012 / 0.24    | 0.0048 / 1.62    | 0.0811 / 37.58   |
|      6 | 0.0012 / 0.44    | 0.0044 / 2.40    | 0.0774 / 35.06   |
|      7 | 0.0012 / 0.90    | 0.0042 / 4.90    | 0.0742 / 53.38   |
|      8 | 0.0012 / 1.26    | 0.0043 / 5.98    | 0.0727 / 60.18   |
|      9 | 0.0014 / 1.26    | 0.0045 / 6.72    | 0.0738 / 62.36   |
|     10 | 0.0015 / 1.30    | 0.0047 / 7.58    | 0.0734 / 64.96   |


</details>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| # layers | # parameters |
|-----------------|----------------|------------------|-----------|----------------------|---------------------|----------|--------------|
| Vanilla         |     1.9 / 11.8 |  1.5 / 10.4      | 44.7      | 69.76                | 89.08               |53        |11684712      |
| TRT fp32        |     1.3 / 4.7  |  0.9 / 4.2       | 67.9      | 69.76                | 89.08               |28        |11678912      |
| TRT fp16        |  0.8 / 27.9    |  0.4 / 27.5      | 25.0      | 69.77                | 89.09               |27        |11678912      |
| TRT int8        |   0.7 / 9.4    |  0.3 / 8.8       | 13.3      | 69.56                | 88.91               |25        |11669504      |

### Batch Size 32

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 26.0 / 38.0    | 19.3 / 26.1      | 44.7      | 69.77                | 89.09               |
| TRT fp32        | 19.1 / 55.0    | 12.4 / 46.6      | 46.1      | 69.76                | 89.09               |
| TRT fp16        | 10.7 / 13.7    |  4.1 / 5.1       | 23.8      | 69.78                | 89.07               |
| TRT int8        |  8.3 / 10.1    |  1.8 / 2.5       | 12.5      | 69.51                | 88.98               |

### Batch Size 64

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 48.1 / 52.5    | 35.3 / 40.0      | 44.7      | 69.77                | 89.08               |
| TRT fp32        | 34.8 / 42.1    | 22.0 / 29.5      | 46.1      | 69.76                | 89.09               |
| TRT fp16        | 20.3 / 22.7    |  7.6 / 9.5       | 23.8      | 69.78                | 89.07               |
| TT int8         | 15.8 / 20.0    |  3.2 / 4.5       | 12.4      | 69.60                | 88.99               |

### Batch Size 128

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 89.2 / 94.0    | 64.5 / 69.7      | 44.7      | 69.80                | 89.10               |
| TRT fp32        | 66.9 / 70.5    | 41.9 / 43.9      | 46.1      | 69.79                | 89.11               |
| TRT fp16        | 38.3 / 40.9    | 13.6 / 15.6      | 23.8      | 69.81                | 89.09               |
| TRT int8        | 30.3 / 32.1    |  5.8 / 6.2       | 12.4      | 69.56                | 89.04               |

### Batch Size 256

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 177.5 / 181.6  | 128.5 / 133.0    | 44.7      | 69.80                | 89.10               |
| TRT fp32        | 133.2 / 135.7  | 84.3 / 86.1      | 46.1      | 69.79                | 89.11               |
| TRT fp16        | 75.7 / 79.0    | 26.0 / 27.5      | 23.8      | 69.81                | 89.09               |

</details>

<details><summary> ResNet34 </summary>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |    3.1 / 7.2   |  2.1 / 6.3       |87.3       |73.29                 |91.52                |
| TRT fp32        |    2.1 / 7.5   |  1.7 / 3.5       |134.5      |73.29                 |91.52                |
| TRT fp16        |    1.0 / 3.9   |  0.6 / 3.3       |46.5       |73.30                 |91.54                |
| TRT int8        |    0.8 / 6.6   |  0.4 / 6.0       |24.4       |73.23                 |91.40                |

</details>

<details><summary> ResNet50 </summary>

### Inference Throughput
![Device part.](/outputs/img_readme/inference_throughput_vs_batch_size_resnet50.png)

### GPU memory usage and SM Utilization over time
256 Vanilla            |  256 fp16 TRT
:-------------------------:|:-------------------------:
![](/outputs/img_readme/gpu_metrics_resnet50_256_vanilla.png)|![](/outputs/img_readme/gpu_metrics_resnet50_256_trt.png)

<details><summary>  Comparisons of ResNet50 Vanilla vs TRT Results </summary> 

* With relative tolearance of 1e-3 over 5k random inputs

| Vanilla VS | equality (%) |
|------------|--------------|
| TRT fp32 | 28.56 |
| TRT fp16 | 26.34 |
| TRT int8 | 0.08 |

*  TOP 10 MAE Error and disagriment % Comparison (over 5k examples)

|  Rank  | MAE / Disg  fp32 | MAE / Disg  fp16 | MAE / Disg  int8 |
|--------|------------------|------------------|------------------|
|      1 | 0.0045 / 0.02    | 0.0040 / 0.00    | 0.8008 / 99.86   |
|      2 | 0.0026 / 1.10    | 0.0036 / 1.82    | 1.1365 / 72.62   |
|      3 | 0.0028 / 1.88    | 0.0039 / 2.36    | 0.8114 / 91.20   |
|      4 | 0.0043 / 1.90    | 0.0038 / 1.26    | 0.8880 / 99.94   |
|      5 | 0.0043 / 3.04    | 0.0037 / 2.30    | 1.0260 / 99.02   |
|      6 | 0.0037 / 4.76    | 0.0035 / 3.96    | 1.0319 / 94.86   |
|      7 | 0.0038 / 5.44    | 0.0036 / 5.00    | 1.0555 / 94.26   |
|      8 | 0.0035 / 3.58    | 0.0035 / 3.68    | 1.0912 / 92.94   |
|      9 | 0.0030 / 3.32    | 0.0036 / 3.44    | 1.1738 / 97.36   |
|     10 | 0.0026 / 3.92    | 0.0037 / 3.96    | 1.1026 / 98.68   |

</details>

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters |
|-----------------|----------------|------------------|-----------|----------------------|---------------------|---------|-------------|
| Vanilla         |   3.7 / 16.7   |  2.8 / 16.3      |102.5      |80.34                 |95.12                |126      | 25530472    |
| TRT fp32        |   2.2 / 4.2    |  1.9 / 3.8       |113.2      |80.34                 |95.13                |79       | 25502912    |
| TRT fp16        |   1.0 / 1.1    |  0.8 / 3.1       |53.2       |80.34                 |95.13                |60       | 25502912    |
| TRT int8        |  0.9 / 2.3     |  0.5 / 1.8       |29.3       |78.57                 |94.94                |58       | 25493504    |

### Batch Size 32

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 62.1 / 83.2    | 55.7 / 76.6      | 97.8      | 80.35                | 95.13               |
| TRT fp32        | 34.3 / 44.1    | 27.7 / 36.3      | 99.7      | 80.35                | 95.13               |
| TRT fp16        | 17.2 / 29.1    | 10.4 / 14.3      | 51.0      | 80.38                | 95.14               |
| TRT int8        | 11.6 / 15.4    |  4.8 / 7.0       | 27.1      | 78.58                | 94.96               |

### Batch Size 64

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 123.7 / 164.0  | 110.8 / 150.1    | 97.8      | 80.35                | 95.13               |
| TRT fp32        | 67.0 / 81.0    | 54.1 / 66.0      | 99.7      | 80.35                | 95.13               |
| TRT fp16        | 32.5 / 43.7    | 19.4 / 25.8      | 51.0      | 80.38                | 95.14               |
| TRT int8        | 21.4 / 33.8    |  8.6 / 12.2      | 27.1      | 78.61                | 94.98               |

### Batch Size 128

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 240.4 / 313.7  | 215.7 / 288.9    | 97.8      | 80.39                | 95.13               |
| TRT fp32        | 123.9 / 158.9  | 99.1 / 132.0     | 99.7      | 80.39                | 95.13               |
| TRT fp16        | 61.7 / 81.5    | 36.1 / 49.3      | 51.0      | 80.42                | 95.14               |

### Batch Size 256

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 475.9 / 558.8  | 426.3 / 508.3    | 97.8      | 80.39                | 95.13               |
| TRT fp32        | 250.4 / 269.7  | 199.9 / 218.3    | 99.7      | 80.39                | 95.13               |
| TRT fp32        | 120.5 / 135.4  | 70.8 / 83.4      | 51.0      | 80.42                | 95.14               |


</details>

<details><summary> ResNet101 </summary> 

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |   5.8 / 15.2   |  4.3 / 14.3      |178.8      |81.67                 |95.65                |
| TRT fp32        |   3.8 / 8.8    |  3.5 / 8.4       |220.5      |81.67                 |95.66                |
| TRT fp16        |   1.7 / 7.1    |  1.3 / 3.1       |91.1       |81.65                 |95.66                |
| TRT int8        |   1.2 / 8.6    |  0.9 / 7.8       |49.0       |79.89                 |95.59                |

</details>

<details><summary>ResNet152</summary>

### Inference Throughput
![Device part.](/outputs/img_readme/inference_throughput_vs_batch_size_resnet152.png)

### GPU memory usage and SM Utilization over time
256 Vanilla            |  256 fp16 TRT
:-------------------------:|:-------------------------:
![](/outputs/img_readme/gpu_metrics_resnet152_256_vanilla.png)|![](/outputs/img_readme/gpu_metrics_resnet152_256_trt.png)


<details><summary>  Comparisons of ResNet152 Vanilla vs TRT Results </summary> 

* With relative tolearance of 1e-3 over 5k random inputs

| Vanilla VS | equality (%) |
|------------|--------------|
| TRT fp32 | 48.79 |
| TRT fp16 | 20.69 |
| TRT int8 | 0.13 |

*  TOP 10 MAE Error Comparison (over 5k examples)

|  Rank  | MAE / Disg  fp32 | MAE / Disg  fp16 | MAE / Disg  int8 |
|--------|------------------|------------------|------------------|
|      1 | 0.0017 / 0.62    | 0.0052 / 3.26    | 1.2601 / 50.56   |
|      2 | 0.0017 / 1.14    | 0.0052 / 4.28    | 1.4691 / 50.94   |
|      3 | 0.0025 / 0.54    | 0.0070 / 1.24    | 1.6041 / 39.00   |
|      4 | 0.0021 / 0.88    | 0.0044 / 2.52    | 1.2059 / 77.54   |
|      5 | 0.0021 / 1.40    | 0.0042 / 3.98    | 1.2333 / 76.00   |
|      6 | 0.0020 / 1.22    | 0.0040 / 3.30    | 1.1825 / 83.14   |
|      7 | 0.0021 / 1.26    | 0.0041 / 3.24    | 1.1236 / 77.52   |
|      8 | 0.0016 / 1.60    | 0.0038 / 3.76    | 1.1334 / 92.12   |
|      9 | 0.0014 / 1.90    | 0.0036 / 4.94    | 1.1734 / 97.06   |
|     10 | 0.0013 / 1.46    | 0.0035 / 5.18    | 1.1071 / 98.18   |

* TOP 10 output example comparison:

|  Rank           |Vanilla Score   | Vanilla Label    |TRT fp32 Score  |TRT fp32 label |
|-----------------|----------------|------------------|----------------|---------------|
| 1               |    5.16475     |  92              |5.16393         |92             |
| 2               |    5.12132     |  892             |5.11767         |892            |
| 3               |    4.94796     |  21              |4.94968         |21             |
| 4               |    4.21629     |  679             |4.21605         |679            |
| 5               |    3.97607     |  128             |3.97772         |128            |
| 6               |    3.93793     |  127             |3.94031         |127            |
| 7               |    3.88538     |  812             |3.8876          |23             |
| 8               |    3.88403     |  23              |3.88676         |812            |
| 9               |    3.71139     |  22              |3.71399         |22             |
| 10              |    3.64397     |  16              |3.64568         |16             |

</details>

## Validation results

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |    8.1 / 16.3  |  6.0 / 15.9      |241.7      |82.34                 |95.92                |
| TRT fp32-dynamic|    13.2 / 19.7 |  12.9 / 19.2     |243.3      |82.34                 |95.92                |
| TRT fp32-static |    5.5 / 10.0  |  5.1 / 9.5       |243.3      |82.34                 |95.92                |
| TRT fp16-dynamic|    7.1 / 13.2  |  6.8 / 11.6      |123.0      |82.31                 |95.91                |
| TRT fp16-static |    2.2 / 8.6   |  1.8 / 8.1       |122.6      |82.32                 |95.90                |
| TRT int8-static |    1.5 / 4.3   |  1.2 / 3.9       |65.5       |79.99                 |95.74                |

Note: 

* Here, we compare the dynamic batch engine with the static batch engine. As the dynamic batch engine is optimized for a batch size of 256, it is not optimal for this example.

* For all subsequent experiments, we utilize a dynamic batch size for every engine except the int8 ones.

### Batch Size 1 - sync

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |  9.3 / 13.6    |  8.8 / 13.0      | 230.5     | 82.34                | 95.92               |
| TRT fp32       |  5.7 / 9.7  |  5.2 / 9.0  | 293.2   | 82.34                | 95.92               |
| TRT fp16       |  2.2 / 3.4  |  1.8 / 2.5  | 116.8   | 82.34                | 95.91               |
| TRT int8       |  1.6 / 5.8  |  1.2 / 5.3  | 62.2    | 79.99                | 95.74               |

### Batch Size 32 

|  Model      |Latency-all (ms)|Latency-model (ms)| size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla     | 141 / 181      |  6.3 / 12.3      |241.7      |82.34                 |95.93                |
| TRT fp32    | 75.6 / 96.2    |   69.3 / 89.8    |243.3      |82.34                 |95.92                |
| TRT fp16    | 30.6 / 55.1    | 24.2 / 48.8      |123.0      |82.32                 |95.91                |
| TRT int8    | 18.1 / 36.4    |  11.6 / 25.0     |64.6       |80.01                 |95.79                |

### Batch Size 32 - sync

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 161.5 / 185.2  | 154.9 / 177.9    | 230.5     | 82.35                | 95.93               |
| resnet152       | 73.6 / 84.5 | 67.2 / 78.1 | 231.2   | 82.34                | 95.92               |
| TRT fp16        | 33.2 / 46.0    | 26.4 / 34.1      | 117.7     | 82.34                | 95.90               |

### Batch Size 64

|  Model      |Latency-all (ms)|Latency-model (ms)|size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla     | 283  / 355     |  6.3 / 11.1      |241.7      |82.34                 |95.93                |
| TRT fp32    |135.2 / 161.4   |  122.9 / 149.1   |243.3      |82.34                 |95.92                |
| TRT fp16    | 59.4 / 83.4    |  46.8 / 65.3     |123.0      |82.32                 |95.91                |

* Note: Unable to create a static batch size int8 engine due to the following error:

    torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.00 MiB. GPU 0 has a total capacty of 11.75 GiB of which 52.00 MiB is free. Including non-PyTorch memory, this process has 11.02 GiB memory in use. Of the allocated memory 9.60 GiB is allocated by PyTorch, and 310.22 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

* Another reason to use a dynamic batch size is to avoid that error.

### Batch Size 64 - sync

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla       | 297.8 / 335.1 | 285.1 / 322.2 | 230.5   | 82.35                | 95.93               |
| TRT fp32       | 136.3 / 163.5 | 123.9 / 150.9 | 231.2   | 82.34                | 95.92               |
| TRT fp16        | 60.3 / 74.4    | 47.4 / 60.6      | 117.7     | 82.34                | 95.90               |

### Batch Size 128

|  Model      |Latency-all (ms)|Latency-model (ms)| size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla     | 555.1 / 620    |  6.0 / 9.5       |241.7      |82.39                 |95.93                |
| TRT fp32    | 269.3 / 336.2  |  244.9 / 311.7   |243.3      |82.38                 |95.93                |
| TRT fp16    | 108.3 / 127.8  |   83.4 / 100.0   |123.0      |82.36                 |95.91                |

### Batch Size 128 - sync

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         | 530.7 / 623.1  | 506.3 / 598.5    | 230.5     | 82.39                | 95.93               |
| TRT fp32        | 267.9 / 325.2 | 243.5 / 300.6 | 231.2   | 82.38                | 95.92               |
| TRT fp16        | 113.6 / 130.9  | 88.4 / 103.8     | 117.7     | 82.38                | 95.90               |

### Batch Size 256

|  Model      |Latency-all (ms)|Latency-model (ms)| size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla     | 1072/1145      |  5.9 / 8.6       |241.7      |82.38                 |95.93                |
| TRT fp32    | 592 / 689      |  543 / 641       |243.3      |82.38                 |95.92                |
| TRT fp16    | 215.2 / 258.3  |  165.7 / 208.3   |123.0      |82.36                 |95.91                |

### Batch Size 256 - sync

|  Model      |Latency-all (ms)|Latency-model (ms)| size (MB) | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla     | 1068.7 / 1229.6| 1020.1 / 1179.9  | 230.5     | 82.38                | 95.93               |
| TRT fp32    | 593.7 / 693.7  |  541.2 / 643.2   | 231.8     |82.38                 |95.92                |
| TRT fp16    | 244.0 / 294.6  | 193.4 / 235.6    | 117.7     | 82.38                | 95.90               |

</details>

</details>

---
# Validation

To validate the models ( vanilla and trt ) with a validation set of the ImageNet-1k dataset:

## Vanilla

```
python main.py -v --batch_size=1 --dataset='datasets/dataset_val/val' --network="resnet18"
```

YOLOv8 example:

```
python main.py -v --batch_size=1 --dataset='datasets/dataset_val/val' --network="yolo" --weights='weights/yolov8n-cls.pt'
```

## TensorRT optimization
```
python main.py -v --batch_size=1 --dataset='datasets/dataset_val/val' --network="resnet18" -trt
```

---
Notes:

* For the YOLOv8: First, download the pretrained weights in the classification (imageNet) section here https://github.com/ultralytics/ultralytics in the weights folder.

* We downloaded a part of the Imagnet Dataset from `https://huggingface.co/datasets/imagenet-1k/viewer/default/validation` and saved it in the `dataset/val` folder. 

* For the labels to function correctly, we utilize the script `format_dataset.py`. This script moves each image into its respective label folder, ensuring our code operates as expected. Ultimately, the dataset should adopt the following structure:

    <details><summary> dataset organization </summary>

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

    </details>

---

<details><summary> Comparison </summary>

Here we compare the output value of the vanilla model vs the TensorRT optimizated model with the function numpy.isclose() as described in `https://ieeexplore.ieee.org/document/10074837` this paper.

 Note: For better performance, we use torch.isclose(), which performs the same function as the NumPy function.

```
python main.py -trt --compare --batch_size=1 --network="resnet18" -rtol=1e-2
```

To comapre using a validation dataset instead of the random generated inputs, you can use this

Note: Currently, comparing the MAE of the top 5 classes, as the isclose() approach in the paper didn't yield good results.

```
python main.py --batch_size=1 --network="resnet18" -trt -rtol=1e-3 --compare --val_dataset --dataset='val_images/'
```

Note: We use the numpy.isclose() function, which returns True or False based on the following condition:

```
 absolute(a - b) <= (atol + rtol * absolute(b)) 
```

In this equation, a represents the output of the vanilla model, b is the output of the TRT optimized model, atol is the absolute tolerance set to 1e-8, and rtol is the relative tolerance set to 1e-3. For the TRT optimized model with FP32 precision, we observed a non-equal percentage of 6.50% with a rtol of 1e-2. Note that this result may change upone re build of the engine.

</details>

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
python onnx_transform.py --weights="weights/best.pth" --pretrained --network="resnet18" --input_shape -1 3 224 224
```

Note: 

* Here, we use the input shape (-1, 3, 224, 224), where '-1' denotes the dynamic batch size.

* Here we are downloading the weights form torch.hub.load, we only inform the `--weights="weights/best.pth"` value to indicate where to save the onnx value later.

To transform your own weights, you can use:

```
python onnx_transform.py --weights="weights/best.pth" --input_shape 1 3 224 224
``` 

## Create the TRT Engine

```
python build_trt.py --fp16 --input_shape -1 3 224 224
```
Note: -1 (for dynamic) is the batch size, 3 the number of channels, 224 is the height and width of the input.

# Annex

<details><summary>  Polygraphy: Engine and Model Summary </summary>

### Installation:

follow `https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy#installation`, or:

inside the env using pip install:

```
pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

### ONNX SUMMARY

In the command line on the proyect dir use:

```
polygraphy inspect model weights/best.onnx --show layers attrs --display-as=trt > onnx_summ.txt
```

### TRT SYMMARY

In the command line on the proyect dir use:

```
polygraphy inspect model weights/best.engine --show layers attrs > trt_summ.txt
```

Note: For less information, just omit the --show layers attrs flag.

## Profile with pytorch 

https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

to use this first do a `pip uninstall tensorboard` then do `pip install torch-tb-profiler`

</details>

---

<details><summary>  Installation </summary>


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

<details><summary> NSight dl  </summary>

No lo usaremos porque en general, solo sirve para probar y evaluar redes generadas en el mismo programa (tipo, programacion por bloques).

Note; completar esto en el futuro.

</details>

<details><summary> trtexec </summary>

No lo usaremos porque en general, sirve para generar los engine de manrea más user friendly, sin tener que hacer todo el proceso que hacemos con onnx y build, permire hacer benchmarks pero nos quita flexibilidad.

Nota: completar esto en el futuro.

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
* how to measure latency: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
* cuda Event time measurement: https://pytorch.org/docs/stable/generated/torch.cuda.Event.html