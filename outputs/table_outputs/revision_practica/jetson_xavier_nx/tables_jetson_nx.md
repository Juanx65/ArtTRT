# mobilenet bs 1
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  71,3  +17,3 -27,3 |  14.0 / 85.1    +4.5 -3.9 |   13.0 / 66.1    |  13.6      | 72.01                | 90.62               | 0       | 0          |
| TRT_fp32        |  140,0  +39,3 -68,0 |   7.1 / 52.8    +2.8 -2.3 |    6.1 / 32.2    |  14.2      | 72.01                | 90.62               | 84      | 3469760    |
| TRT_fp16        |  175,1  +59,0 -119,0 |   5.7 / 149.0   +2.9 -2.3 |    4.5 / 82.1    |  7.5       | 71.99                | 90.63               | 58      | 3469760    |
| TRT_int8        |  224,6  +81,1 -176,5 |   4.5 / 31.0    +2.5 -2.0 |    3.3 / 24.4    |  4.4       | 71.44                | 90.33               | 57      | 3469760    |
 
# mobilenet bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  139,4  +12,1 -13,9 | 229.5 / 619.8   +21.8 -20.8 |  219.5 / 606.2   |  13.6      | 72.02                | 90.63               | 0       | 0          |
| TRT_fp32        |  275,1  +25,8 -30,1 | 116.3 / 153.3   +12.1 -11.5 |  105.5 / 141.7   |  13.6      | 72.02                | 90.63               | 82      | 3469760    |
| TRT_fp16        |  563,1  +86,7 -112,8 |  56.8 / 98.2    +10.3 -9.5 |   46.0 / 77.6    |  7.1       | 72.04                | 90.62               | 58      | 3469760    |
| TRT_int8        |  898,2  +175,8 -249,3 |  35.6 / 63.8    +8.7 -7.7 |   25.2 / 44.6    |  4.0       | 71.48                | 90.29               | 57      | 3469760    |
 
# mobilenet bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  143,7  +21,1 -27,1 | 445.3 / 1011.1  +76.5 -70.6 |  425.3 / 989.8   |  13.6      | 72.02                | 90.63               | 0       | 0          |


Memoria excedida (170248 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 53020
Memoria excedida (162004 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 64 3 224 224, terminando PID 53200
Memoria excedida (170968 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 53204
Memoria excedida (179632 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 53208
Memoria excedida (190672 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 64 3 224 224, terminando PID 53212
Memoria excedida (200048 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 53216
Memoria excedida (185476 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 64 3 224 224, terminando PID 53221
Memoria excedida (188664 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 64 3 224 224, terminando PID 53523
Memoria excedida (198036 KB disponibles) por main.py -v --batch_size 64 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 53527
 
# mobilenet bs 128
 
CUDA is available.
Memoria excedida (181264 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network mobilenet --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 53534
Memoria excedida (189908 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 53890
Memoria excedida (144492 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 54141
Memoria excedida (141664 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 54345
Memoria excedida (133764 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 54349
Memoria excedida (198176 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 128 3 224 224, terminando PID 54354
Memoria excedida (192004 KB disponibles) por build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape 128 3 224 224, terminando PID 54646
Memoria excedida (180244 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8, terminando PID 54650
 
# mobilenet bs 256
 
CUDA is available.
Memoria excedida (164784 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 54657
Memoria excedida (166812 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 55032
Memoria excedida (179436 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 55036
Memoria excedida (157948 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 55251
Memoria excedida (138340 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 55461
Memoria excedida (122140 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network mobilenet -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 55465
Memoria excedida (116608 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network mobilenet --input_shape 256 3 224 224, terminando PID 55470
 
# resnet18 bs 1
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  55,8  +10,9 -15,5 |  17.9 / 56.2    +4.4 -3.9 |   16.6 / 54.8    |  44.7      | 69.76                | 89.08               | 0       | 0          |
| TRT_fp32        |  83,9  +17,0 -24,4 |  11.9 / 42.2    +3.0 -2.7 |   10.8 / 33.1    |  66.2      | 69.76                | 89.08               | 24      | 11678912   |
| TRT_fp16        |  142,5  +38,9 -66,0 |   7.0 / 30.9    +2.6 -2.2 |    5.9 / 25.3    |  23.1      | 69.75                | 89.08               | 26      | 11678912   |
| TRT_int8        |  215,8  +81,6 -187,8 |   4.6 / 38.4    +2.8 -2.2 |    3.4 / 18.2    |  11.9      | 69.56                | 88.98               | 25      | 11669504   |
 
# resnet18 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  131,2  +5,4 -5,8 | 244.0 / 273.0   +10.5 -10.3 |  234.7 / 248.4   |  44.7      | 69.76                | 89.09               | 0       | 0          |
| TRT_fp32        |  123,8  +4,8 -5,1 | 258.5 / 309.8   +10.4 -10.2 |  249.2 / 264.5   |  45.2      | 69.76                | 89.09               | 29      | 11678912   |
| TRT_fp16        |  461,6  +56,1 -68,7 |  69.3 / 97.2    +9.6 -9.0 |   58.8 / 74.1    |  22.6      | 69.78                | 89.09               | 33      | 11678912   |
| TRT_int8        |  839,8  +141,4 -189,3 |  38.1 / 62.8    +7.7 -7.0 |   27.9 / 41.3    |  11.4      | 69.55                | 88.95               | 25      | 11669504   |
 
# resnet18 bs 64
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  133,3  +2,4 -2,5 | 480.1 / 505.9   +8.9 -8.8 |  461.8 / 470.7   |  44.7      | 69.76                | 89.08               | 0       | 0          |
| TRT_fp32        |  130,0  +2,4 -2,5 | 492.2 / 523.6   +9.3 -9.2 |  473.9 / 501.3   |  44.7      | 69.76                | 89.09               | 24      | 11678912   |
| TRT_fp16        |  485,4  +49,5 -58,5 | 131.8 / 168.5   +15.0 -14.2 |  111.6 / 123.4   |  22.7      | 69.77                | 89.08               | 39      | 11678912   |
| TRT_int8        |  874,4  +137,5 -180,2 |  73.2 / 125.8   +13.7 -12.5 |   52.9 / 67.1    |  11.4      | 69.59                | 88.93               | 25      | 11669504   |
 
# resnet18 bs 128
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  135,4  +1,6 -1,6 | 945.3 / 970.2   +11.4 -11.3 |  910.8 / 921.8   |  44.7      | 69.80                | 89.10               | 0       | 0          |


Memoria excedida (149588 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet18 --input_shape 128 3 224 224, terminando PID 102582
Memoria excedida (136388 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 128 3 224 224, terminando PID 102828
Memoria excedida (138312 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32, terminando PID 102832
Memoria excedida (137748 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet18 --input_shape 128 3 224 224, terminando PID 102836
Memoria excedida (142712 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 128 3 224 224, terminando PID 102840
Memoria excedida (142568 KB disponibles) por main.py -v --batch_size 128 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 102844
Memoria excedida (179000 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet18 --input_shape 128 3 224 224, terminando PID 102849
 
# resnet18 bs 256
 
CUDA is available.
Memoria excedida (198112 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet18 --less --engine weights/best_fp32.engine --model_version Vanilla, terminando PID 103108
Memoria excedida (199280 KB disponibles) por onnx_transform.py --weights weights/best_fp32.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 103327
Memoria excedida (200404 KB disponibles) por build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape 256 3 224 224, terminando PID 103331
Memoria excedida (140156 KB disponibles) por onnx_transform.py --weights weights/best_fp16.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 103541
Memoria excedida (130156 KB disponibles) por build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape 256 3 224 224, terminando PID 103748
Memoria excedida (125984 KB disponibles) por main.py -v --batch_size 256 --dataset datasets/dataset_val/val --network resnet18 -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16, terminando PID 103752
Memoria excedida (185552 KB disponibles) por onnx_transform.py --weights weights/best_int8.pth --pretrained --network resnet18 --input_shape 256 3 224 224, terminando PID 103757
 
# resnet34 bs 1
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  33,1  +4,4 -5,5 |  30.2 / 144.5   +4.6 -4.3 |   28.9 / 120.3   |  83.3      | 73.30                | 91.42               | 0       | 0          |
| TRT_fp32        |  49,8  +6,5 -8,0 |  20.1 / 46.7    +3.0 -2.8 |   19.0 / 42.7    |  127.8     | 73.30                | 91.42               | 40      | 21779648   |
| TRT_fp16        |  120,6  +29,0 -45,5 |   8.3 / 208.1   +2.6 -2.3 |    7.2 / 108.8   |  42.4      | 73.30                | 91.43               | 43      | 21779648   |
| TRT_int8        |  152,4  +37,4 -59,2 |   6.6 / 24.0    +2.1 -1.8 |    5.5 / 19.3    |  21.6      | 73.22                | 91.42               | 41      | 21770240   |
 
# resnet34 bs 32
 
CUDA is available.
No se encontró el número de parametros
No se encontró el número de capas
|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|
|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|
| Vanilla         |  80,0  +1,1 -1,1 | 400.1 / 425.0   +5.5 -5.4 |  391.3 / 415.0   |  83.3      | 73.31                | 91.42               | 0       | 0          |
| TRT_fp32        |  69,6  +0,7 -0,7 | 460.1 / 478.3   +4.6 -4.6 |  451.2 / 458.0   |  83.6      | 73.31                | 91.42               | 45      | 21779648   |
| TRT_fp16        |  276,1  +18,8 -20,9 | 115.9 / 171.0   +8.5 -8.2 |  105.5 / 140.9   |  42.0      | 73.30                | 91.42               | 67      | 21779648   |