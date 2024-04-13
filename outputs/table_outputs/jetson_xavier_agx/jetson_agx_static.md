# mobilenet bs 1
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  66,7  +11,6 -15,6 |  15.0 / 57.2    +3.1 -2.8 |   13.9 / 56.1    |  13.6      | 72.01                | 90.62               | 0       | 0          |
| TRT_fp32        |  110,0  +19,1 -25,9 |   9.1 / 28.0    +1.9 -1.7 |    7.9 / 23.2    |  14.3      | 72.01                | 90.62               | 0       | 3469760    |
| TRT_fp16        |  187,5  +47,2 -75,9 |   5.3 / 26.4    +1.8 -1.5 |    4.1 / 22.1    |  7.5       | 72.00                | 90.63               | 0       | 3469760    |
| TRT_int8        |  236,0  +68,8 -122,3 |   4.2 / 26.6    +1.7 -1.4 |    3.0 / 18.1    |  4.4       | 71.42                | 90.32               | 0       | 3469760    |
 
# mobilenet bs 32
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  224,9  +12,9 -14,2 | 142.3 / 191.0   +8.7 -8.4 |  131.7 / 162.4   |  13.6      | 72.02                | 90.63               | 0       | 0          |
| TRT_fp32        |  439,4  +64,5 -82,7 |  72.8 / 121.1   +12.5 -11.5 |   61.0 / 80.6    |  13.6      | 72.02                | 90.63               | 0       | 3469760    |
| TRT_fp16        |  807,3  +176,3 -262,6 |  39.6 / 108.1   +11.1 -9.7 |   28.4 / 53.0    |  7.1       | 72.04                | 90.64               | 0       | 3469760    |
| TRT_int8        |  1.012,7  +270,9 -452,8 |  31.6 / 91.0    +11.5 -9.8 |   20.0 / 59.0    |  4.0       | 71.40                | 90.35               | 0       | 3469760    |
 
# mobilenet bs 64
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  233,6  +14,7 -16,2 | 274.0 / 322.1   +18.3 -17.8 |  251.5 / 264.7   |  13.6      | 72.02                | 90.63               | 0       | 0          |
| TRT_fp32        |  456,3  +54,2 -66,0 | 140.3 / 190.3   +18.9 -17.7 |  117.0 / 129.2   |  13.6      | 72.02                | 90.63               | 0       | 3469760    |
| TRT_fp16        |  846,3  +166,0 -235,5 |  75.6 / 122.5   +18.5 -16.5 |   53.1 / 66.1    |  7.1       | 72.04                | 90.64               | 0       | 3469760    |
| TRT_int8        |  1.260,8  +335,5 -558,7 |  50.8 / 100.4   +18.4 -15.6 |   28.9 / 48.0    |  4.0       | 71.42                | 90.32               | 0       | 3469760    |
 
# mobilenet bs 128
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  237,6  +11,7 -12,6 | 538.6 / 624.7   +27.9 -27.2 |  499.2 / 531.2   |  13.6      | 72.05                | 90.64               | 0       | 0          |
| TRT_fp32        |  476,3  +38,6 -44,0 | 268.8 / 323.3   +23.7 -22.7 |  231.8 / 266.9   |  13.6      | 72.05                | 90.64               | 0       | 3469760    |
| TRT_fp16        |  899,5  +150,5 -201,1 | 142.3 / 229.9   +28.6 -26.0 |  103.9 / 129.1   |  7.2       | 72.03                | 90.65               | 0       | 3469760    |
| TRT_int8        |  1.376,7  +297,3 -440,3 |  93.0 / 150.5   +25.6 -22.5 |   55.5 / 89.0    |  4.0       | 71.55                | 90.34               | 0       | 3469760    |
 
# mobilenet bs 256
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  239,3  +9,9 -10,6 | 1069.8 / 1159.8  +46.2 -45.3 |  995.2 / 1006.4  |  13.6      | 72.05                | 90.64               | 0       | 0          |
Memoria excedida (148040 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 291261
Memoria excedida (151864 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 291524
Memoria excedida (152720 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 291530
Memoria excedida (150320 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 291534
Memoria excedida (149788 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 291540
Memoria excedida (147188 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 291545
Memoria excedida (142680 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 291551
Memoria excedida (144212 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 291555
Memoria excedida (166136 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 291559
 
# resnet18 bs 1
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  86,4  +16,7 -23,6 |  11.6 / 39.6    +2.8 -2.5 |   10.3 / 34.2    |  44.7      | 69.76                | 89.08               | 0       | 0          |
| TRT_fp32        |  103,5  +22,8 -34,2 |   9.7 / 27.4    +2.7 -2.4 |    8.4 / 25.2    |  66.3      | 69.76                | 89.08               | 0       | 11678912   |
| TRT_fp16        |  166,2  +43,6 -71,9 |   6.0 / 28.0    +2.1 -1.8 |    4.7 / 19.6    |  23.1      | 69.75                | 89.09               | 0       | 11678912   |
| TRT_int8        |  238,6  +56,5 -87,8 |   4.2 / 18.8    +1.3 -1.1 |    3.0 / 12.4    |  11.9      | 69.55                | 89.02               | 0       | 11669504   |
 
# resnet18 bs 32
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  214,2  +13,4 -14,8 | 149.4 / 196.3   +10.0 -9.6 |  138.4 / 158.7   |  44.7      | 69.76                | 89.09               | 0       | 0          |
| TRT_fp32        |  209,6  +8,7 -9,3 | 152.7 / 180.8   +6.6 -6.5 |  142.7 / 158.0   |  45.1      | 69.76                | 89.09               | 0       | 11678912   |
| TRT_fp16        |  687,0  +130,8 -183,4 |  46.6 / 113.5   +11.0 -9.8 |   35.7 / 64.5    |  22.5      | 69.75                | 89.10               | 0       | 11678912   |
| TRT_int8        |  1.091,6  +266,7 -421,4 |  29.3 / 70.7    +9.5 -8.2 |   18.8 / 34.0    |  11.5      | 69.62                | 88.92               | 0       | 11669504   |
 
# resnet18 bs 64
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  220,5  +10,0 -10,8 | 290.3 / 336.1   +13.8 -13.5 |  270.2 / 283.1   |  44.7      | 69.76                | 89.08               | 0       | 0          |
| TRT_fp32        |  220,2  +12,9 -14,2 | 290.6 / 360.7   +18.1 -17.6 |  269.5 / 283.8   |  45.1      | 69.76                | 89.09               | 0       | 11678912   |
| TRT_fp16        |  704,3  +134,4 -188,6 |  90.9 / 149.9   +21.4 -19.2 |   66.6 / 84.9    |  22.6      | 69.79                | 89.09               | 0       | 11678912   |
| TRT_int8        |  1.073,9  +270,7 -435,8 |  59.6 / 92.2    +20.1 -17.2 |   31.5 / 55.1    |  11.4      | 69.56                | 88.92               | 0       | 11669504   |
 
# resnet18 bs 128
 
CUDA is available.
No se encontró el número de parametros
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  225,0  +7,5 -7,8 | 568.9 / 620.0   +19.5 -19.2 |  533.4 / 582.1   |  44.7      | 69.80                | 89.10               | 0       | 0          |
| TRT_fp32        |  228,8  +9,5 -10,1 | 559.5 / 624.3   +24.2 -23.6 |  522.6 / 534.9   |  44.9      | 69.80                | 89.10               | 0       | 11678912   |
| TRT_fp16        |  762,4  +116,7 -151,6 | 167.9 / 249.7   +30.3 -27.8 |  128.9 / 185.7   |  22.7      | 69.79                | 89.11               | 0       | 11678912   |
| TRT_int8        |  1.250,3  +295,1 -457,5 | 102.4 / 174.4   +31.6 -27.4 |   61.2 / 121.5   |  11.5      | 69.65                | 88.98               | 0       | 11669504   |
 
# resnet18 bs 256
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  231,2  +1,8 -1,8 | 1107.3 / 1138.5  +8.8 -8.8 | 1047.7 / 1055.5  |  44.7      | 69.80                | 89.10               | 0       | 0          |
| TRT_fp32        |  235,7  +4,8 -5,0 | 1086.1 / 1156.4  +22.7 -22.4 | 1024.3 / 1030.1  |  44.9      | 69.80                | 89.10               | 0       | 11678912   |
| TRT_fp16        |  787,3  +94,1 -114,7 | 325.2 / 396.6   +44.1 -41.4 |  252.8 / 285.8   |  22.6      | 69.82                | 89.11               | 0       | 11678912   |

# resnet34 bs 1

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  56,2  +1,7 -1,8 |  17.8 / 23.8    +0.5 -0.5 |   16.8 / 22.9    |  83.3      | 73.30                | 91.42               | 0       | 0          |
| TRT_fp32        |  84,6  +5,8 -6,5 |  11.8 / 14.6    +0.9 -0.8 |   10.9 / 13.5    |  127.8     | 73.30                | 91.42               | 0       | 21779648   |
| TRT_fp16        |  133,3  +14,9 -17,9 |   7.5 / 15.3    +0.9 -0.9 |    6.5 / 10.1    |  42.5      | 73.29                | 91.42               | 0       | 21779648   |
| TRT_int8        |  165,3  +19,4 -23,5 |   6.0 / 19.2    +0.8 -0.8 |    5.0 / 9.6     |  21.7      | 73.24                | 91.35               | 0       | 21770240   |

# resnet34 bs 32

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  130,9  +0,5 -0,5 | 244.5 / 246.9   +0.8 -0.8 |  236.1 / 236.9   |  83.3      | 73.31                | 91.42               | 0       | 0          |
| TRT_fp32        |  117,6  +0,7 -0,7 | 272.1 / 276.8   +1.7 -1.7 |  263.8 / 267.8   |  83.9      | 73.31                | 91.42               | 0       | 21779648   |
| TRT_fp16        |  427,3  +41,2 -48,2 |  74.9 / 98.2    +8.0 -7.6 |   64.9 / 80.7    |  42.0      | 73.31                | 91.43               | 0       | 21779648   |
| TRT_int8        |  775,1  +115,1 -148,3 |  41.3 / 58.9    +7.2 -6.6 |   31.6 / 48.4    |  21.2      | 73.27                | 91.38               | 0       | 21770240   |

# resnet34 bs 64

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  134,0  +0,6 -0,6 | 477.6 / 487.5   +2.0 -2.0 |  462.3 / 464.0   |  83.3      | 73.31                | 91.42               | 0       | 0          |
| TRT_fp32        |  128,7  +0,4 -0,4 | 497.2 / 500.2   +1.6 -1.6 |  482.0 / 483.8   |  83.9      | 73.31                | 91.42               | 0       | 21779648   |
| TRT_fp16        |  431,2  +42,6 -50,0 | 148.4 / 167.1   +16.3 -15.4 |  121.1 / 132.2   |  41.9      | 73.30                | 91.42               | 0       | 21779648   |
| TRT_int8        |  757,2  +126,9 -169,7 |  84.5 / 108.7   +17.0 -15.5 |   57.8 / 60.4    |  21.2      | 73.21                | 91.37               | 0       | 21770240   |


# resnet34 bs 128

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  138,4  +0,4 -0,4 | 925.1 / 932.8   +2.6 -2.6 |  897.0 / 897.9   |  83.3      | 73.35                | 91.43               | 0       | 0          |
| TRT_fp32        |  135,7  +0,5 -0,5 | 943.5 / 951.9   +3.3 -3.3 |  915.0 / 918.2   |  83.6      | 73.35                | 91.43               | 0       | 21779648   |
| TRT_fp16        |  461,1  +37,6 -42,9 | 277.6 / 309.1   +24.6 -23.6 |  234.9 / 248.7   |  42.0      | 73.33                | 91.42               | 0       | 21779648   |
| TRT_int8        |  852,6  +123,6 -158,1 | 150.1 / 194.4   +25.5 -23.5 |  111.2 / 133.5   |  21.2      | 73.30                | 91.39               | 0       | 21770240   |

# resnet34 bs 256

No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  142,2  +0,4 -0,4 | 1799.7 / 1805.9  +4.6 -4.6 | 1745.1 / 1747.4  |  83.3      | 73.35                | 91.43               | 0       | 0          |
| TRT_fp32        |  139,6  +0,4 -0,4 | 1833.6 / 1839.7  +5.1 -5.1 | 1780.1 / 1782.6  |  83.6      | 73.35                | 91.43               | 0       | 21779648   |
| TRT_fp16        |  480,2  +31,7 -35,2 | 533.1 / 596.5   +37.7 -36.4 |  461.3 / 486.3   |  42.0      | 73.33                | 91.42               | 0       | 21779648   |

# resnet50 bs 1

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  40,9  +0,9 -0,9 |  24.4 / 31.5    +0.5 -0.5 |   23.4 / 30.4    |  97.8      | 80.35                | 95.13               | 0       | 0          |
| TRT_fp32        |  58,2  +2,2 -2,3 |  17.2 / 35.1    +0.7 -0.7 |   16.2 / 25.4    |  108.1     | 80.35                | 95.13               | 0       | 25502912   |
| TRT_fp16        |  105,3  +10,0 -11,6 |   9.5 / 19.9    +1.0 -0.9 |    8.4 / 15.5    |  49.8      | 80.35                | 95.12               | 0       | 25502912   |
[02/23/2024-16:13:15] [TRT] [E] 1: [stdArchiveReader.cpp::StdArchiveReader::57] Error Code 1: Serialization (Serialization assertion sizeRead == static_cast<uint64_t>(mEnd - mCurrent) failed.Size specified in header does not match archive size)
[02/23/2024-16:13:15] [TRT] [E] 4: [runtime.cpp::deserializeCudaEngine::65] Error Code 4: Internal Error (Engine deserialization failed.)

# resnet50 bs 32

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  65,5  +0,1 -0,1 | 488.2 / 491.8   +1.1 -1.1 |  480.5 / 481.9   |  97.8      | 80.36                | 95.13               | 0       | 0          |
| TRT_fp32        |  85,4  +0,4 -0,4 | 374.5 / 380.7   +1.6 -1.6 |  366.6 / 369.8   |  98.0      | 80.36                | 95.13               | 0       | 25502912   |
| TRT_fp16        |  335,7  +25,5 -28,8 |  95.3 / 116.4   +7.8 -7.5 |   85.2 / 104.9   |  49.2      | 80.34                | 95.14               | 0       | 25502912   |
| TRT_int8        |  581,9  +69,8 -85,2 |  55.0 / 76.6    +7.5 -7.0 |   45.0 / 57.8    |  25.3      | 79.83                | 95.07               | 0       | 25493504   |

# resnet50 bs 64

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  67,3  +0,1 -0,1 | 951.2 / 954.8   +1.7 -1.7 |  936.6 / 938.5   |  97.8      | 80.36                | 95.13               | 0       | 0          |
| TRT_fp32        |  89,3  +0,3 -0,3 | 716.6 / 723.7   +2.6 -2.6 |  701.6 / 709.4   |  97.9      | 80.36                | 95.13               | 0       | 25502912   |
| TRT_fp16        |  341,2  +27,0 -30,6 | 187.6 / 213.6   +16.1 -15.4 |  165.0 / 179.9   |  49.4      | 80.34                | 95.14               | 0       | 25502912   |
| TRT_int8        |  594,6  +80,0 -100,4 | 107.6 / 133.1   +16.7 -15.5 |   84.9 / 99.3    |  25.2      | 79.92                | 95.07               | 0       | 25493504   |


# resnet50 bs 128

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  68,2  +0,1 -0,1 | 1875.5 / 1880.9  +2.3 -2.3 | 1848.8 / 1850.6  |  97.8      | 80.39                | 95.13               | 0       | 0          |
| TRT_fp32        |  91,6  +0,2 -0,2 | 1397.6 / 1402.8  +3.0 -3.0 | 1369.9 / 1373.5  |  97.7      | 80.39                | 95.13               | 0       | 25502912   |
| TRT_fp16        |  358,1  +17,8 -19,2 | 357.4 / 388.2   +18.6 -18.2 |  323.1 / 331.3   |  49.4      | 80.38                | 95.14               | 0       | 25502912   |
| TRT_int8        |  637,6  +54,4 -62,4 | 200.7 / 236.6   +18.7 -17.9 |  166.6 / 170.9   |  25.3      | 78.75                | 95.00               | 0       | 25493504   |

# resnet50 bs 256

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  68,9  +0,1 -0,1 | 3714.6 / 3719.9  +3.1 -3.1 | 3664.0 / 3665.7  |  97.8      | 80.39                | 95.13               | 0       | 0          |

# resnet101 bs 1

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  24,0  +0,2 -0,2 |  41.7 / 49.2    +0.4 -0.4 |   40.8 / 45.7    |  170.5     | 81.67                | 95.66               | 0       | 0          |

# resnet101 bs 32

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  39,3  +0,1 -0,1 | 814.3 / 817.4   +1.4 -1.4 |  806.4 / 808.1   |  170.5     | 81.68                | 95.66               | 0       | 0          |

# resnet101 bs 64

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  40,2  +0,0 -0,0 | 1593.8 / 1597.3  +2.0 -2.0 | 1579.4 / 1582.0  |  170.5     | 81.68                | 95.66               | 0       | 0          |


# resnet101 bs 128

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  41,0  +0,0 -0,0 | 3121.1 / 3124.1  +2.0 -2.0 | 3095.4 / 3097.2  |  170.5     | 81.71                | 95.66               | 0       | 0          |

# resnet101 bs 256

CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  41,5  +0,0 -0,0 | 6163.5 / 6170.0  +3.0 -3.0 | 6113.4 / 6116.3  |  170.5     | 81.71                | 95.66               | 0       | 0          |


