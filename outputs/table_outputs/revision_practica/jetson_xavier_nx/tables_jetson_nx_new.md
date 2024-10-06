# mobilenet bs 1
 
CUDA is available.
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  38,3  +22,6 -155,3 |  26.1 / 320.9   +37.5 -20.9 |   24.4 / 313.1   |  13.6      | 72.01                | 90.62               | 93      | 21789160   |
| TRT_fp32        |  100,8  +52,2 -218,4 |   9.9 / 204.2   +10.6 -6.8 |    8.1 / 197.9   |  14.2      | 72.01                | 90.62               | 84      | 3469760    |
| TRT_fp16        |  111,8  +55,1 -203,3 |   8.9 / 174.0   +8.7 -5.8 |    7.1 / 130.9   |  7.5       | 71.98                | 90.64               | 58      | 3469760    |
| TRT_int8        |  144,6  +75,1 -317,4 |   6.9 / 154.6   +7.5 -4.8 |    5.1 / 130.0   |  4.4       | 71.44                | 90.33               | 57      | 3469760    |
 
# mobilenet bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  137,9  +18,8 -23,7 | 232.1 / 396.7   +36.7 -34.1 |  214.6 / 312.5   |  13.6      | 72.02                | 90.63               | 0       | 0          |
| TRT_fp32        |  265,0  +64,7 -102,2 | 120.8 / 288.8   +39.0 -33.6 |  102.7 / 210.6   |  13.6      | 72.02                | 90.63               | 80      | 3469760    |
| TRT_fp16        |  482,9  +200,5 -525,8 |  66.3 / 247.5   +47.1 -34.5 |   48.2 / 197.0   |  7.1       | 72.01                | 90.62               | 58      | 3469760    |
| TRT_int8        |  715,9  +342,6 -1.174,6 |  44.7 / 220.1   +41.0 -27.8 |   27.3 / 208.2   |  4.0       | 71.48                | 90.29               | 57      | 3469760    |
 
# mobilenet bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  145,7  +13,1 -15,2 | 439.3 / 610.6   +43.5 -41.5 |  407.5 / 524.9   |  13.6      | 72.02                | 90.63               | 0       | 0          |


Memoria excedida (134392 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 203557
Memoria excedida (143304 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 64 3 224 224, terminando PID 204341
Memoria excedida (171476 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 204345
Memoria excedida (184260 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 204349
Memoria excedida (191184 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 64 3 224 224, terminando PID 204353
Memoria excedida (166604 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 205426
Memoria excedida (150308 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 64 3 224 224, terminando PID 206486
Memoria excedida (154272 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 206490
 
# mobilenet bs 128
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  147,4  +9,5 -10,5 | 868.1 / 1107.5  +59.9 -57.9 |  813.8 / 895.6   |  13.6      | 72.05                | 90.64               | 0       | 0          |


Memoria excedida (191716 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 216606
Memoria excedida (200640 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 217928
Memoria excedida (160820 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 218976
Memoria excedida (154588 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 220066
Memoria excedida (154172 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 220070
Memoria excedida (187508 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 220075
Memoria excedida (192992 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 128 3 224 224, terminando PID 221640
Memoria excedida (200924 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 221646
 
# mobilenet bs 256
 
CUDA is available.
Memoria excedida (178172 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 221655
Memoria excedida (176480 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 223522
Memoria excedida (165604 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 223530
Memoria excedida (159024 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 223534
Memoria excedida (159768 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 223540
Memoria excedida (159808 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 223545
Memoria excedida (160976 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 223552
Memoria excedida (153420 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 223565
Memoria excedida (155072 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 223575
Memoria excedida (153284 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 223583
 
# resnet18 bs 1
 
Memoria excedida (143392 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet18 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 223604
Memoria excedida (139668 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet18 --input_shape 1 3 224 224, terminando PID 223615
Memoria excedida (139512 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 1 3 224 224, terminando PID 223624
Memoria excedida (138936 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 223635
Memoria excedida (136856 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet18 --input_shape 1 3 224 224, terminando PID 223644
Memoria excedida (136044 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 1 3 224 224, terminando PID 223651
Memoria excedida (135316 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 223658
Memoria excedida (139552 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet18 --input_shape 1 3 224 224, terminando PID 223675
 
# resnet18 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  133,6  +3,4 -3,6 | 239.6 / 267.8   +6.4 -6.3 |  229.5 / 238.8   |  44.7      | 69.76                | 89.09               | 0       | 0          |
| TRT_fp32        |  126,1  +2,5 -2,5 | 253.8 / 280.0   +5.1 -5.0 |  243.8 / 270.2   |  44.9      | 69.76                | 89.09               | 27      | 11678912   |
| TRT_fp16        |  467,9  +48,6 -57,6 |  68.4 / 94.4    +7.9 -7.5 |   57.2 / 84.6    |  22.6      | 69.77                | 89.09               | 33      | 11678912   |
| TRT_int8        |  835,4  +146,3 -198,6 |  38.3 / 78.4    +8.1 -7.4 |   27.5 / 41.9    |  11.4      | 69.55                | 88.95               | 25      | 11669504   |
 
# resnet18 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  135,5  +2,4 -2,5 | 472.2 / 497.1   +8.5 -8.5 |  452.2 / 458.4   |  44.7      | 69.76                | 89.08               | 0       | 0          |
| TRT_fp32        |  132,9  +2,1 -2,1 | 481.5 / 512.2   +7.7 -7.6 |  461.8 / 470.6   |  44.9      | 69.76                | 89.09               | 29      | 11678912   |
| TRT_fp16        |  488,0  +43,9 -50,7 | 131.2 / 173.6   +13.0 -12.4 |  109.5 / 146.1   |  22.7      | 69.79                | 89.09               | 39      | 11678912   |
| TRT_int8        |  871,0  +127,7 -163,9 |  73.5 / 126.9   +12.6 -11.6 |   51.8 / 57.3    |  11.4      | 69.59                | 88.93               | 25      | 11669504   |
 
# resnet18 bs 128
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  137,9  +1,8 -1,8 | 928.2 / 968.0   +12.2 -12.1 |  891.5 / 899.4   |  44.7      | 69.80                | 89.10               | 0       | 0          |
| TRT_fp32        |  138,0  +1,7 -1,8 | 927.9 / 963.8   +11.9 -11.8 |  891.9 / 902.3   |  44.7      | 69.80                | 89.10               | 27      | 11678912   |
| TRT_fp16        |  504,0  +34,9 -39,0 | 254.0 / 319.1   +18.9 -18.2 |  214.5 / 255.3   |  22.6      | 69.82                | 89.11               | 45      | 11678912   |
| TRT_int8        |  915,7  +107,5 -130,5 | 139.8 / 203.1   +18.6 -17.4 |   99.7 / 109.2   |  11.4      | 69.64                | 88.97               | 25      | 11669504   |
 
# resnet18 bs 256
 
CUDA is available.
Memoria excedida (179952 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet18 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 303278
Memoria excedida (180172 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 304056
Memoria excedida (204140 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 304060
Memoria excedida (174376 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 304599
Memoria excedida (173916 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 305126
Memoria excedida (172792 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 305136
Memoria excedida (160932 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 305141
Memoria excedida (142784 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 305883
Memoria excedida (143312 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 305887
 
# resnet34 bs 1
 
Memoria excedida (179708 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet34 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 305900
Memoria excedida (192652 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet34 --input_shape 1 3 224 224, terminando PID 305904


| TRT_fp16        |  118,8  +21,3 -29,1 |   8.4 / 42.7    +1.8 -1.7 |    7.3 / 41.2    |  42.4      | 73.30                | 91.43               | 42      | 21779648   |
| TRT_int8        |  146,0  +32,0 -47,6 |   6.8 / 36.9    +1.9 -1.7 |    5.6 / 33.0    |  21.6      | 73.22                | 91.42               | 41      | 21770240   |
 
# resnet34 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  80,2  +0,8 -0,9 | 399.2 / 418.9   +4.3 -4.2 |  389.9 / 397.7   |  83.3      | 73.31                | 91.42               | 0       | 0          |
| TRT_fp32        |  69,6  +0,5 -0,5 | 459.6 / 475.3   +3.1 -3.1 |  450.3 / 461.3   |  84.0      | 73.31                | 91.42               | 45      | 21779648   |
| TRT_fp16        |  274,5  +19,9 -22,3 | 116.6 / 168.1   +9.1 -8.8 |  105.2 / 130.6   |  42.0      | 73.30                | 91.42               | 63      | 21779648   |
| TRT_int8        |  518,0  +57,1 -68,5 |  61.8 / 82.8    +7.7 -7.2 |   50.8 / 59.4    |  21.2      | 73.21                | 91.34               | 41      | 21770240   |
 
# resnet34 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  81,0  +0,7 -0,7 | 790.0 / 819.7   +6.8 -6.8 |  771.8 / 779.9   |  83.3      | 73.31                | 91.42               | 0       | 0          |
| TRT_fp32        |  76,2  +0,5 -0,5 | 840.4 / 868.5   +5.7 -5.6 |  822.4 / 828.0   |  83.6      | 73.31                | 91.42               | 45      | 21779648   |
| TRT_fp16        |  285,5  +16,0 -17,5 | 224.2 / 267.9   +13.3 -12.9 |  202.5 / 208.9   |  42.0      | 73.30                | 91.42               | 70      | 21779648   |
| TRT_int8        |  540,9  +49,6 -57,6 | 118.3 / 156.7   +11.9 -11.4 |   97.0 / 107.0   |  21.2      | 73.19                | 91.41               | 41      | 21770240   |
 
# resnet34 bs 128
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  83,7  +0,4 -0,5 | 1529.7 / 1552.7  +8.2 -8.2 | 1496.9 / 1505.8  |  83.3      | 73.35                | 91.43               | 0       | 0          |


Memoria excedida (189556 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet34 --input_shape 128 3 224 224, terminando PID 406283
Memoria excedida (175100 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 406813
Memoria excedida (172512 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 406817
Memoria excedida (169848 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet34 --input_shape 128 3 224 224, terminando PID 406821
Memoria excedida (168772 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 406826
Memoria excedida (181324 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 406830
Memoria excedida (188564 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet34 --input_shape 128 3 224 224, terminando PID 406835
Memoria excedida (190012 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 128 3 224 224, terminando PID 407615
Memoria excedida (188068 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 407623
 
# resnet34 bs 256
 
CUDA is available.
Memoria excedida (161008 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet34 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 407631
Memoria excedida (164912 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet34 --input_shape 256 3 224 224, terminando PID 408613
Memoria excedida (155600 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 408617
Memoria excedida (154588 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 408621
Memoria excedida (154600 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet34 --input_shape 256 3 224 224, terminando PID 408625
Memoria excedida (154336 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 408629
Memoria excedida (154524 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 408633
Memoria excedida (149340 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet34 --input_shape 256 3 224 224, terminando PID 408638
Memoria excedida (152040 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 409424
Memoria excedida (165556 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet34 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 409428
 
# resnet50 bs 1
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  24,7  +2,3 -2,7 |  40.4 / 91.1    +4.2 -4.0 |   38.9 / 71.2    |  97.8      | 80.35                | 95.13               | 0       | 0          |
| TRT_fp32        |  32,9  +3,4 -4,0 |  30.4 / 59.9    +3.5 -3.3 |   29.0 / 58.0    |  108.2     | 80.35                | 95.13               | 59      | 25502912   |
| TRT_fp16        |  101,7  +15,7 -20,5 |   9.8 / 38.4    +1.8 -1.6 |    8.7 / 35.3    |  49.7      | 80.37                | 95.12               | 60      | 25502912   |
| TRT_int8        |  118,7  +23,8 -34,0 |   8.4 / 121.3   +2.1 -1.9 |    7.2 / 73.1    |  25.7      | 79.44                | 95.06               | 58      | 25493504   |
 
# resnet50 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  40,2  +0,2 -0,2 | 796.2 / 805.4   +3.3 -3.3 |  786.3 / 794.3   |  97.8      | 80.36                | 95.13               | 0       | 0          |
| TRT_fp32        |  49,7  +0,2 -0,2 | 643.4 / 664.7   +2.9 -2.9 |  633.8 / 636.6   |  97.7      | 80.36                | 95.13               | 60      | 25502912   |
| TRT_fp16        |  210,9  +12,1 -13,3 | 151.7 / 204.3   +9.3 -9.0 |  139.6 / 157.1   |  49.3      | 80.35                | 95.13               | 87      | 25502912   |
| TRT_int8        |  381,4  +36,0 -42,0 |  83.9 / 145.5   +8.8 -8.3 |   72.6 / 85.7    |  25.2      | 78.63                | 94.98               | 58      | 25493504   |
 
# resnet50 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  40,9  +0,2 -0,2 | 1563.3 / 1584.8  +6.0 -5.9 | 1546.9 / 1556.5  |  97.8      | 80.36                | 95.13               | 0       | 0          |


Memoria excedida (193232 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet50 --input_shape 64 3 224 224, terminando PID 549060
Memoria excedida (176640 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 64 3 224 224, terminando PID 549353
Memoria excedida (169732 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 549357
Memoria excedida (172148 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet50 --input_shape 64 3 224 224, terminando PID 549363
Memoria excedida (188196 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 64 3 224 224, terminando PID 549367
Memoria excedida (152104 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet50 --input_shape 64 3 224 224, terminando PID 549932
Memoria excedida (140724 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 64 3 224 224, terminando PID 550510
Memoria excedida (135652 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 550517
 
# resnet50 bs 128
 
CUDA is available.
Memoria excedida (203112 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet50 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 550526
Memoria excedida (188580 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet50 --input_shape 128 3 224 224, terminando PID 552409
Memoria excedida (175320 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 553126
Memoria excedida (197120 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 553130
Memoria excedida (169860 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet50 --input_shape 128 3 224 224, terminando PID 553134
Memoria excedida (189620 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 553943
Memoria excedida (193592 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 553947
Memoria excedida (151600 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet50 --input_shape 128 3 224 224, terminando PID 553954
 
# resnet50 bs 256
 
CUDA is available.
Memoria excedida (157228 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet50 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 555377
Memoria excedida (150716 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet50 --input_shape 256 3 224 224, terminando PID 556120
Memoria excedida (149904 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 556128
Memoria excedida (148752 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 556132
Memoria excedida (149340 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet50 --input_shape 256 3 224 224, terminando PID 556136
Memoria excedida (138080 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 556140
Memoria excedida (141196 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 556144
Memoria excedida (184008 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet50 --input_shape 256 3 224 224, terminando PID 556149
Memoria excedida (174568 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 556976
Memoria excedida (174708 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet50 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 556983
 
# resnet101 bs 1
 
Memoria excedida (170580 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet101 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 556997
Memoria excedida (152712 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet101 --input_shape 1 3 224 224, terminando PID 557001
Memoria excedida (153912 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 1 3 224 224, terminando PID 557005
Memoria excedida (154836 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 557009
Memoria excedida (154752 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet101 --input_shape 1 3 224 224, terminando PID 557013
Memoria excedida (150592 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 1 3 224 224, terminando PID 557017
Memoria excedida (151932 KB disponibles) por main.py -v --batch_size 1 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 557021
| TRT_int8        |  106,8  +17,3 -23,0 |   9.4 / 69.0    +1.8 -1.7 |    8.3 / 48.1    |  44.4      | 81.13                | 95.64               | 109     | 44433408   |
 
# resnet101 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  24,0  +0,1 -0,1 | 1336.1 / 1352.0  +3.1 -3.1 | 1325.9 / 1332.5  |  170.5     | 81.68                | 95.66               | 0       | 0          |


Memoria excedida (137432 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet101 --input_shape 32 3 224 224, terminando PID 598435
Memoria excedida (141764 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 32 3 224 224, terminando PID 598815
Memoria excedida (150856 KB disponibles) por main.py -v --batch_size 32 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 598819
Memoria excedida (169684 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet101 --input_shape 32 3 224 224, terminando PID 598823
Memoria excedida (196652 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 32 3 224 224, terminando PID 598827
Memoria excedida (203140 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet101 --input_shape 32 3 224 224, terminando PID 599422
Memoria excedida (201832 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 32 3 224 224, terminando PID 600048
 
# resnet101 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  24,3  +0,0 -0,0 | 2633.1 / 2645.3  +4.1 -4.1 | 2617.2 / 2625.1  |  170.5     | 81.68                | 95.66               | 0       | 0          |


Memoria excedida (187556 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet101 --input_shape 64 3 224 224, terminando PID 622424
Memoria excedida (158628 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet101 --input_shape 64 3 224 224, terminando PID 623643
Memoria excedida (159536 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 64 3 224 224, terminando PID 624229
Memoria excedida (169488 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 624233
Memoria excedida (164628 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet101 --input_shape 64 3 224 224, terminando PID 624240
Memoria excedida (162048 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 64 3 224 224, terminando PID 625060
Memoria excedida (135340 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 625065
 
# resnet101 bs 128
 
Memoria excedida (157584 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet101 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 625076
Memoria excedida (191960 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet101 --input_shape 128 3 224 224, terminando PID 625080
Memoria excedida (198124 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 625903
Memoria excedida (204772 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 625907
Memoria excedida (203444 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet101 --input_shape 128 3 224 224, terminando PID 625911
Memoria excedida (199136 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 625915
Memoria excedida (194692 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 625919
Memoria excedida (166320 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet101 --input_shape 128 3 224 224, terminando PID 625924
Memoria excedida (166156 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 128 3 224 224, terminando PID 625929
Memoria excedida (163196 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 625934
 
# resnet101 bs 256
 
CUDA is available.
Memoria excedida (155508 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet101 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 625952
Memoria excedida (142168 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet101 --input_shape 256 3 224 224, terminando PID 626940
Memoria excedida (145864 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 626948
Memoria excedida (145272 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 626952
Memoria excedida (145740 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet101 --input_shape 256 3 224 224, terminando PID 626956
Memoria excedida (144640 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 626960
Memoria excedida (143916 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 626964
Memoria excedida (136108 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet101 --input_shape 256 3 224 224, terminando PID 626969
Memoria excedida (136960 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 626973
Memoria excedida (139896 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet101 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 626977
 
# resnet152 bs 1
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  10,4  +0,2 -0,3 |  96.2 / 161.6   +2.3 -2.3 |   94.8 / 145.9   |  230.5     | 82.35                | 95.92               | 0       | 0          |
| TRT_fp32        |  13,1  +0,3 -0,3 |  76.1 / 107.7   +1.7 -1.7 |   74.9 / 106.2   |  294.6     | 82.35                | 95.92               | 166     | 60040384   |
| TRT_fp16        |  40,5  +4,7 -5,7 |  24.7 / 59.7    +3.2 -3.0 |   23.4 / 58.2    |  116.0     | 82.34                | 95.92               | 161     | 60040384   |
| TRT_int8        |  76,3  +13,3 -18,0 |  13.1 / 56.0    +2.8 -2.5 |   11.9 / 48.6    |  59.9      | 80.19                | 95.78               | 160     | 60030976   |
 
# resnet152 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  16,8  +0,0 -0,0 | 1899.7 / 1907.4  +2.8 -2.8 | 1889.3 / 1895.7  |  230.5     | 82.35                | 95.92               | 0       | 0          |


Memoria excedida (135256 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet152 --input_shape 32 3 224 224, terminando PID 794832
Memoria excedida (179792 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet152 --input_shape 32 3 224 224, terminando PID 795857
Memoria excedida (193720 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 32 3 224 224, terminando PID 796431
Memoria excedida (201796 KB disponibles) por main.py -v --batch_size 32 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 796436
Memoria excedida (167180 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet152 --input_shape 32 3 224 224, terminando PID 796445
 
# resnet152 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  17,0  +0,0 -0,0 | 3755.3 / 3773.8  +4.2 -4.2 | 3739.6 / 3745.3  |  230.5     | 82.35                | 95.92               | 0       | 0          |


Memoria excedida (173892 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet152 --input_shape 64 3 224 224, terminando PID 828457
Memoria excedida (161504 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 64 3 224 224, terminando PID 828932
Memoria excedida (170056 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 828936
Memoria excedida (173412 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet152 --input_shape 64 3 224 224, terminando PID 828940
Memoria excedida (192864 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 64 3 224 224, terminando PID 828944
Memoria excedida (202060 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet152 --input_shape 64 3 224 224, terminando PID 829508
Memoria excedida (198548 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 64 3 224 224, terminando PID 830093
Memoria excedida (193340 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 830097
 
# resnet152 bs 128
 
CUDA is available.
Memoria excedida (181500 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet152 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 830104
Memoria excedida (202060 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet152 --input_shape 128 3 224 224, terminando PID 831248
Memoria excedida (191544 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 832025
Memoria excedida (191096 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 832031
Memoria excedida (185748 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet152 --input_shape 128 3 224 224, terminando PID 832038
Memoria excedida (180388 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 832045
Memoria excedida (177588 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 832049
Memoria excedida (157696 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet152 --input_shape 128 3 224 224, terminando PID 832054
Memoria excedida (143384 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 128 3 224 224, terminando PID 832058
Memoria excedida (142108 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 832062
 
# resnet152 bs 256
 
CUDA is available.
Memoria excedida (130120 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet152 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 832069
Memoria excedida (128004 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet152 --input_shape 256 3 224 224, terminando PID 833120
Memoria excedida (128208 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 833124
Memoria excedida (131740 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 833128
Memoria excedida (128308 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet152 --input_shape 256 3 224 224, terminando PID 833132
Memoria excedida (128152 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 833136
Memoria excedida (170764 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet152 --input_shape 256 3 224 224, terminando PID 833710
Memoria excedida (161204 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 256 3 224 224, terminando PID 834295
Memoria excedida (157864 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet152 -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 834302
 
