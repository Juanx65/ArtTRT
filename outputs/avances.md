# Avances para 21 mar 2024

* Automatizando el proceso para las pruebas en conjunto con juanjo, se descubre que tegrastats solo es capaz de samplear efectivamente a 100ms en lugar de 1ms como dice la documentacion https://docs.nvidia.com/drive/drive-os-5.2.0.0L/drive-os/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html#wwpID0E0EB0HA.

* Se automatizo el proceso para probar la red de juanjo, resultado en `outouts/table_outputs/juanjo_test_all.md`


# Avances para 20 mar 2024

Se genero una red, tal como las que usa juanjo, (lineal y relu) , usando el codigo en `utils/experiments/RELUNet.py`, con CustomNet,
y se calcula el tiempo promedio / max para procesar 10000 batches ( de batch size 1 ), sea nx, M, nu, L = entradas, neuronas x capa, salidas, Capas, dando como resultado:

para todos nx, nu = 2, 1

* L, M = 3, 5

    Vanilla 0.0006388949632644654  |  0.061730384826660156 segundos

    TRT fp32 0.0002678051710128784 |  0.0010783672332763672 segundos

    TRT fp16 0.000269349217414856  |  0.0011029243469238281 segundos

* L, M = 3, 10

    Vanilla 0.0006396333694458008 | 0.053017616271972656 segundos

    TRT fp32 0.00027604928016662595 | 0.0011589527130126953 segundos

    TRT fp16 0.00027497944831848147 | 0.0011036396026611328 segundos

* L, M = 3, 50

    Vanilla 0.0006770261764526367 | 0.051016807556152344 segundos

    TRT fp32 0.00028007709980010985 | 0.001172780990600586 segundos

    TRT fp16 0.0002736010789871216 | 0.0011126995086669922 segundos

* L, M = 3, 100

    Vanilla 0.000680115795135498 | 0.06267952919006348 segundos

    TRT fp32 0.00027640178203582766 | 0.0011110305786132812 segundos

    TRT fp16 0.00027351372241973874 | 0.0011425018310546875 segundos

* L, M = 3, 500

    Vanilla 0.0006533797264099121 | 0.05687451362609863 segundos

    TRT fp32 0.0002992525577545166 | 0.0011496543884277344 segundos

    TRT fp16 0.0003029118299484253 | 0.0011544227600097656 segundos

* L, M = 10, 5

    Vanilla 0.001537083911895752 | 0.05540776252746582 segundos

    TRT fp32 0.00039377779960632326 | 0.0014982223510742188 segundos

    Vanilla 0.0003935312509536743 | 0.0015141963958740234 segundos

* L, M = 10, 10

    Vanilla 0.001563770580291748 | 0.056751251220703125 segundos

    TRT fp32 0.0004172969341278076 | 0.0015053749084472656 segundos

    TRT fp16 0.0003932722568511963 | 0.0016186237335205078 segundos

* L, M = 10, 50

    Vanilla 0.0015504326820373535 | 0.055413007736206055 segundos

    TRT fp32 0.00041126227378845216 | 0.0016429424285888672 segundos

    TRT FP16 0.00041672141551971434 | 0.0016052722930908203 segundos

* L, M = 10, 100

    Vanilla 0.001504798746109009 | 0.06291484832763672 segundos

    TRT fp32 0.00043169891834259034 | 0.001615762710571289 segundos

    TRT fp16 0.000417138147354126 | 0.0016262531280517578 segundos

* L, M = 10, 500

    Vanilla 0.0014842774391174316 | 0.054978370666503906 segundos

    TRT fp32 0.000998554825782776 | 0.0019719600677490234 segundos

    TRT fp16 0.0007437546014785767 | 0.009686708450317383 segundos

# Avances para 23 nov

Se genero una red de 18 capas, todas RELU, usando el codigo en `utils/experiments/RELUNet.py`
y se calcula el tiempo promedio para procesar 10000 batches ( de batch size 1 ) dando como resultado:

Vanilla: 0.0010208755970001222 segundos

TRT fp32: 0.0008841029405593872 segundos

TRT fp16: 0.0008815905809402466 segundos

---

Para comparar capa por capa la red en pytorch (.pth), onnx, y engine, use el codigo en `build_experiment.sh` el cual en la primera iteracion muestra los 3 resumenes como se muestra a continuacion: (  esto para una resnet18 de batch size 1)

## yolov8n-cls

<details><summary> ENGINE fp32 SUMM </summary> 

| Layer (type) | Output Shape |
| ---------------|-----------------|
| Reformat - 1 | 1,3,224,224 |
| CaskConvolution - 2 | 1,16,112,112 |
| NoOp - 3 | 1,16,112,112 |
| PointWiseV2 - 4 | 1,16,112,112 |
| CaskConvolution - 5 | 1,32,56,56 |
| PointWiseV2 - 6 | 1,32,56,56 |
| CaskConvolution - 7 | 1,32,56,56 |
| PointWiseV2 - 8 | 1,32,56,56 |
| CaskConvolution - 9 | 1,16,56,56 |
| PointWiseV2 - 10 | 1,16,56,56 |
| CaskConvolution - 11 | 1,16,56,56 |
| PointWiseV2 - 12 | 1,16,56,56 |
| Reformat - 13 | 1,16,56,56 |
| Reformat - 14 | 1,16,56,56 |
| CaskConvolution - 15 | 1,32,56,56 |
| PointWiseV2 - 16 | 1,32,56,56 |
| CaskConvolution - 17 | 1,64,28,28 |
| NoOp - 18 | 1,64,28,28 |
| PointWiseV2 - 19 | 1,64,28,28 |
| NoOp - 20 | 1,64,28,28 |
| CaskConvolution - 21 | 1,64,28,28 |
| NoOp - 22 | 1,64,28,28 |
| PointWiseV2 - 23 | 1,64,28,28 |
| NoOp - 24 | 1,64,28,28 |
| CaskConvolution - 25 | 1,32,28,28 |
| PointWiseV2 - 26 | 1,32,28,28 |
| CaskConvolution - 27 | 1,32,28,28 |
| PointWiseV2 - 28 | 1,32,28,28 |
| NoOp - 29 | 1,32,28,28 |
| NoOp - 30 | 1,32,28,28 |
| CaskConvolution - 31 | 1,32,28,28 |
| PointWiseV2 - 32 | 1,32,28,28 |
| CaskConvolution - 33 | 1,32,28,28 |
| NoOp - 34 | 1,32,28,28 |
| PointWiseV2 - 35 | 1,32,28,28 |
| Reformat - 36 | 1,32,28,28 |
| Reformat - 37 | 1,32,28,28 |
| Reformat - 38 | 1,32,28,28 |
| NoOp - 39 | 1,128,28,28 |
| CaskGemmConvolution - 40 | 1,64,28,28 |
| NoOp - 41 | 1,64,28,28 |
| PointWiseV2 - 42 | 1,64,28,28 |
| NoOp - 43 | 1,64,28,28 |
| CaskConvolution - 44 | 1,128,14,14 |
| PointWiseV2 - 45 | 1,128,14,14 |
| CaskGemmConvolution - 46 | 1,128,14,14 |
| PointWiseV2 - 47 | 1,128,14,14 |
| Reformat - 48 | 1,128,14,14 |
| CaskConvolution - 49 | 1,64,14,14 |
| PointWiseV2 - 50 | 1,64,14,14 |
| CaskConvolution - 51 | 1,64,14,14 |
| PointWiseV2 - 52 | 1,64,14,14 |
| CaskConvolution - 53 | 1,64,14,14 |
| PointWiseV2 - 54 | 1,64,14,14 |
| CaskConvolution - 55 | 1,64,14,14 |
| PointWiseV2 - 56 | 1,64,14,14 |
| Reformat - 57 | 1,64,14,14 |
| Reformat - 58 | 1,64,14,14 |
| Reformat - 59 | 1,64,14,14 |
| Reformat - 60 | 1,64,14,14 |
| NoOp - 61 | 1,256,14,14 |
| CaskGemmConvolution - 62 | 1,128,14,14 |
| PointWiseV2 - 63 | 1,128,14,14 |
| NoOp - 64 | 1,128,14,14 |
| CaskConvolution - 65 | 1,256,7,7 |
| PointWiseV2 - 66 | 1,256,7,7 |
| NoOp - 67 | 1,256,7,7 |
| CaskGemmConvolution - 68 | 1,256,7,7 |
| NoOp - 69 | 1,256,7,7 |
| PointWiseV2 - 70 | 1,256,7,7 |
| NoOp - 71 | 1,256,7,7 |
| Reformat - 72 | 1,128,7,7 |
| CaskConvolution - 73 | 1,128,7,7 |
| PointWiseV2 - 74 | 1,128,7,7 |
| CaskConvolution - 75 | 1,128,7,7 |
| Reformat - 76 | 1,128,7,7 |
| PointWiseV2 - 77 | 1,128,7,7 |
| Reformat - 78 | 1,128,7,7 |
| Reformat - 79 | 1,128,7,7 |
| CaskGemmConvolution - 80 | 1,256,7,7 |
| NoOp - 81 | 1,256,7,7 |
| PointWiseV2 - 82 | 1,256,7,7 |
| NoOp - 83 | 1,256,7,7 |
| CaskGemmConvolution - 84 | 1,1280,7,7 |
| NoOp - 85 | 1,1280,7,7 |
| PointWiseV2 - 86 | 1,1280,7,7 |
| NoOp - 87 | 1,1280,7,7 |
| CaskPooling - 88 | 1,1280,1,1 |
| NoOp - 89 | 1,1280,1,1 |
| CaskGemmConvolution - 90 | 1,1000,1,1 |
| NoOp - 91 | 1,1000 |
| CaskSoftMaxV2 - 92 | 1,1000 |

</details>

<details><summary> ONNX SUMM  </summary> 

| Layer (type) | Output Shape |
| ---------------|-----------------|
| LayerType.CONVOLUTION - 1 | 1,16,112,112 |
| ActivationType.SIGMOID - 2 | 1,16,112,112 |
| LayerType.ELEMENTWISE - 3 | 1,16,112,112 |
| LayerType.CONVOLUTION - 4 | 1,32,56,56 |
| ActivationType.SIGMOID - 5 | 1,32,56,56 |
| LayerType.ELEMENTWISE - 6 | 1,32,56,56 |
| LayerType.CONVOLUTION - 7 | 1,32,56,56 |
| ActivationType.SIGMOID - 8 | 1,32,56,56 |
| LayerType.ELEMENTWISE - 9 | 1,32,56,56 |
| LayerType.SHAPE - 10 | 4, |
| LayerType.CONSTANT - 11 | 1, |
| LayerType.GATHER - 12 | 1, |
| LayerType.CONSTANT - 13 | 1, |
| LayerType.ELEMENTWISE - 14 | 1, |
| LayerType.CONSTANT - 15 | 1, |
| LayerType.ELEMENTWISE - 16 | 1, |
| LayerType.CONSTANT - 17 | 1, |
| LayerType.ELEMENTWISE - 18 | 1, |
| LayerType.CONSTANT - 19 | 4, |
| LayerType.CONCATENATION - 20 | 5, |
| LayerType.CONSTANT - 21 | 4, |
| LayerType.GATHER - 22 | 4, |
| LayerType.CONSTANT - 23 | 1, |
| LayerType.ELEMENTWISE - 24 | 4, |
| LayerType.CONSTANT - 25 | 1, |
| LayerType.ELEMENTWISE - 26 | 4, |
| LayerType.ELEMENTWISE - 27 | 4, |
| LayerType.ELEMENTWISE - 28 | 4, |
| LayerType.CONSTANT - 29 | 4, |
| LayerType.ELEMENTWISE - 30 | 4, |
| LayerType.ELEMENTWISE - 31 | 4, |
| LayerType.CONSTANT - 32 | 4, |
| LayerType.ELEMENTWISE - 33 | 4, |
| LayerType.SLICE - 34 | 1,16,56,56 |
| LayerType.CONSTANT - 35 | 1, |
| LayerType.ELEMENTWISE - 36 | 1, |
| LayerType.CONSTANT - 37 | 4, |
| LayerType.CONCATENATION - 38 | 5, |
| LayerType.CONSTANT - 39 | 4, |
| LayerType.GATHER - 40 | 4, |
| LayerType.CONSTANT - 41 | 4, |
| LayerType.CONCATENATION - 42 | 5, |
| LayerType.GATHER - 43 | 4, |
| LayerType.CONSTANT - 44 | 1, |
| LayerType.ELEMENTWISE - 45 | 4, |
| LayerType.CONSTANT - 46 | 1, |
| LayerType.ELEMENTWISE - 47 | 4, |
| LayerType.ELEMENTWISE - 48 | 4, |
| LayerType.ELEMENTWISE - 49 | 4, |
| LayerType.CONSTANT - 50 | 1, |
| LayerType.ELEMENTWISE - 51 | 4, |
| LayerType.ELEMENTWISE - 52 | 4, |
| LayerType.CONSTANT - 53 | 1, |
| LayerType.ELEMENTWISE - 54 | 4, |
| LayerType.CONSTANT - 55 | 1, |
| LayerType.ELEMENTWISE - 56 | 4, |
| LayerType.ELEMENTWISE - 57 | 4, |
| LayerType.ELEMENTWISE - 58 | 4, |
| LayerType.CONSTANT - 59 | 4, |
| LayerType.ELEMENTWISE - 60 | 4, |
| LayerType.ELEMENTWISE - 61 | 4, |
| LayerType.ELEMENTWISE - 62 | 4, |
| LayerType.CONSTANT - 63 | 4, |
| LayerType.ELEMENTWISE - 64 | 4, |
| LayerType.SLICE - 65 | 1,16,56,56 |
| LayerType.CONVOLUTION - 66 | 1,16,56,56 |
| ActivationType.SIGMOID - 67 | 1,16,56,56 |
| LayerType.ELEMENTWISE - 68 | 1,16,56,56 |
| LayerType.CONVOLUTION - 69 | 1,16,56,56 |
| ActivationType.SIGMOID - 70 | 1,16,56,56 |
| LayerType.ELEMENTWISE - 71 | 1,16,56,56 |
| LayerType.ELEMENTWISE - 72 | 1,16,56,56 |
| LayerType.CONCATENATION - 73 | 1,48,56,56 |
| LayerType.CONVOLUTION - 74 | 1,32,56,56 |
| ActivationType.SIGMOID - 75 | 1,32,56,56 |
| LayerType.ELEMENTWISE - 76 | 1,32,56,56 |
| LayerType.CONVOLUTION - 77 | 1,64,28,28 |
| ActivationType.SIGMOID - 78 | 1,64,28,28 |
| LayerType.ELEMENTWISE - 79 | 1,64,28,28 |
| LayerType.CONVOLUTION - 80 | 1,64,28,28 |
| ActivationType.SIGMOID - 81 | 1,64,28,28 |
| LayerType.ELEMENTWISE - 82 | 1,64,28,28 |
| LayerType.SHAPE - 83 | 4, |
| LayerType.CONSTANT - 84 | 1, |
| LayerType.GATHER - 85 | 1, |
| LayerType.CONSTANT - 86 | 1, |
| LayerType.ELEMENTWISE - 87 | 1, |
| LayerType.CONSTANT - 88 | 1, |
| LayerType.ELEMENTWISE - 89 | 1, |
| LayerType.CONSTANT - 90 | 1, |
| LayerType.ELEMENTWISE - 91 | 1, |
| LayerType.CONSTANT - 92 | 4, |
| LayerType.CONCATENATION - 93 | 5, |
| LayerType.CONSTANT - 94 | 4, |
| LayerType.GATHER - 95 | 4, |
| LayerType.CONSTANT - 96 | 1, |
| LayerType.ELEMENTWISE - 97 | 4, |
| LayerType.CONSTANT - 98 | 1, |
| LayerType.ELEMENTWISE - 99 | 4, |
| LayerType.ELEMENTWISE - 100 | 4, |
| LayerType.ELEMENTWISE - 101 | 4, |
| LayerType.CONSTANT - 102 | 4, |
| LayerType.ELEMENTWISE - 103 | 4, |
| LayerType.ELEMENTWISE - 104 | 4, |
| LayerType.CONSTANT - 105 | 4, |
| LayerType.ELEMENTWISE - 106 | 4, |
| LayerType.SLICE - 107 | 1,32,28,28 |
| LayerType.CONSTANT - 108 | 1, |
| LayerType.ELEMENTWISE - 109 | 1, |
| LayerType.CONSTANT - 110 | 4, |
| LayerType.CONCATENATION - 111 | 5, |
| LayerType.CONSTANT - 112 | 4, |
| LayerType.GATHER - 113 | 4, |
| LayerType.CONSTANT - 114 | 4, |
| LayerType.CONCATENATION - 115 | 5, |
| LayerType.GATHER - 116 | 4, |
| LayerType.CONSTANT - 117 | 1, |
| LayerType.ELEMENTWISE - 118 | 4, |
| LayerType.CONSTANT - 119 | 1, |
| LayerType.ELEMENTWISE - 120 | 4, |
| LayerType.ELEMENTWISE - 121 | 4, |
| LayerType.ELEMENTWISE - 122 | 4, |
| LayerType.CONSTANT - 123 | 1, |
| LayerType.ELEMENTWISE - 124 | 4, |
| LayerType.ELEMENTWISE - 125 | 4, |
| LayerType.CONSTANT - 126 | 1, |
| LayerType.ELEMENTWISE - 127 | 4, |
| LayerType.CONSTANT - 128 | 1, |
| LayerType.ELEMENTWISE - 129 | 4, |
| LayerType.ELEMENTWISE - 130 | 4, |
| LayerType.ELEMENTWISE - 131 | 4, |
| LayerType.CONSTANT - 132 | 4, |
| LayerType.ELEMENTWISE - 133 | 4, |
| LayerType.ELEMENTWISE - 134 | 4, |
| LayerType.ELEMENTWISE - 135 | 4, |
| LayerType.CONSTANT - 136 | 4, |
| LayerType.ELEMENTWISE - 137 | 4, |
| LayerType.SLICE - 138 | 1,32,28,28 |
| LayerType.CONVOLUTION - 139 | 1,32,28,28 |
| ActivationType.SIGMOID - 140 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 141 | 1,32,28,28 |
| LayerType.CONVOLUTION - 142 | 1,32,28,28 |
| ActivationType.SIGMOID - 143 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 144 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 145 | 1,32,28,28 |
| LayerType.CONVOLUTION - 146 | 1,32,28,28 |
| ActivationType.SIGMOID - 147 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 148 | 1,32,28,28 |
| LayerType.CONVOLUTION - 149 | 1,32,28,28 |
| ActivationType.SIGMOID - 150 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 151 | 1,32,28,28 |
| LayerType.ELEMENTWISE - 152 | 1,32,28,28 |
| LayerType.CONCATENATION - 153 | 1,128,28,28 |
| LayerType.CONVOLUTION - 154 | 1,64,28,28 |
| ActivationType.SIGMOID - 155 | 1,64,28,28 |
| LayerType.ELEMENTWISE - 156 | 1,64,28,28 |
| LayerType.CONVOLUTION - 157 | 1,128,14,14 |
| ActivationType.SIGMOID - 158 | 1,128,14,14 |
| LayerType.ELEMENTWISE - 159 | 1,128,14,14 |
| LayerType.CONVOLUTION - 160 | 1,128,14,14 |
| ActivationType.SIGMOID - 161 | 1,128,14,14 |
| LayerType.ELEMENTWISE - 162 | 1,128,14,14 |
| LayerType.SHAPE - 163 | 4, |
| LayerType.CONSTANT - 164 | 1, |
| LayerType.GATHER - 165 | 1, |
| LayerType.CONSTANT - 166 | 1, |
| LayerType.ELEMENTWISE - 167 | 1, |
| LayerType.CONSTANT - 168 | 1, |
| LayerType.ELEMENTWISE - 169 | 1, |
| LayerType.CONSTANT - 170 | 1, |
| LayerType.ELEMENTWISE - 171 | 1, |
| LayerType.CONSTANT - 172 | 4, |
| LayerType.CONCATENATION - 173 | 5, |
| LayerType.CONSTANT - 174 | 4, |
| LayerType.GATHER - 175 | 4, |
| LayerType.CONSTANT - 176 | 1, |
| LayerType.ELEMENTWISE - 177 | 4, |
| LayerType.CONSTANT - 178 | 1, |
| LayerType.ELEMENTWISE - 179 | 4, |
| LayerType.ELEMENTWISE - 180 | 4, |
| LayerType.ELEMENTWISE - 181 | 4, |
| LayerType.CONSTANT - 182 | 4, |
| LayerType.ELEMENTWISE - 183 | 4, |
| LayerType.ELEMENTWISE - 184 | 4, |
| LayerType.CONSTANT - 185 | 4, |
| LayerType.ELEMENTWISE - 186 | 4, |
| LayerType.SLICE - 187 | 1,64,14,14 |
| LayerType.CONSTANT - 188 | 1, |
| LayerType.ELEMENTWISE - 189 | 1, |
| LayerType.CONSTANT - 190 | 4, |
| LayerType.CONCATENATION - 191 | 5, |
| LayerType.CONSTANT - 192 | 4, |
| LayerType.GATHER - 193 | 4, |
| LayerType.CONSTANT - 194 | 4, |
| LayerType.CONCATENATION - 195 | 5, |
| LayerType.GATHER - 196 | 4, |
| LayerType.CONSTANT - 197 | 1, |
| LayerType.ELEMENTWISE - 198 | 4, |
| LayerType.CONSTANT - 199 | 1, |
| LayerType.ELEMENTWISE - 200 | 4, |
| LayerType.ELEMENTWISE - 201 | 4, |
| LayerType.ELEMENTWISE - 202 | 4, |
| LayerType.CONSTANT - 203 | 1, |
| LayerType.ELEMENTWISE - 204 | 4, |
| LayerType.ELEMENTWISE - 205 | 4, |
| LayerType.CONSTANT - 206 | 1, |
| LayerType.ELEMENTWISE - 207 | 4, |
| LayerType.CONSTANT - 208 | 1, |
| LayerType.ELEMENTWISE - 209 | 4, |
| LayerType.ELEMENTWISE - 210 | 4, |
| LayerType.ELEMENTWISE - 211 | 4, |
| LayerType.CONSTANT - 212 | 4, |
| LayerType.ELEMENTWISE - 213 | 4, |
| LayerType.ELEMENTWISE - 214 | 4, |
| LayerType.ELEMENTWISE - 215 | 4, |
| LayerType.CONSTANT - 216 | 4, |
| LayerType.ELEMENTWISE - 217 | 4, |
| LayerType.SLICE - 218 | 1,64,14,14 |
| LayerType.CONVOLUTION - 219 | 1,64,14,14 |
| ActivationType.SIGMOID - 220 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 221 | 1,64,14,14 |
| LayerType.CONVOLUTION - 222 | 1,64,14,14 |
| ActivationType.SIGMOID - 223 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 224 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 225 | 1,64,14,14 |
| LayerType.CONVOLUTION - 226 | 1,64,14,14 |
| ActivationType.SIGMOID - 227 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 228 | 1,64,14,14 |
| LayerType.CONVOLUTION - 229 | 1,64,14,14 |
| ActivationType.SIGMOID - 230 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 231 | 1,64,14,14 |
| LayerType.ELEMENTWISE - 232 | 1,64,14,14 |
| LayerType.CONCATENATION - 233 | 1,256,14,14 |
| LayerType.CONVOLUTION - 234 | 1,128,14,14 |
| ActivationType.SIGMOID - 235 | 1,128,14,14 |
| LayerType.ELEMENTWISE - 236 | 1,128,14,14 |
| LayerType.CONVOLUTION - 237 | 1,256,7,7 |
| ActivationType.SIGMOID - 238 | 1,256,7,7 |
| LayerType.ELEMENTWISE - 239 | 1,256,7,7 |
| LayerType.CONVOLUTION - 240 | 1,256,7,7 |
| ActivationType.SIGMOID - 241 | 1,256,7,7 |
| LayerType.ELEMENTWISE - 242 | 1,256,7,7 |
| LayerType.SHAPE - 243 | 4, |
| LayerType.CONSTANT - 244 | 1, |
| LayerType.GATHER - 245 | 1, |
| LayerType.CONSTANT - 246 | 1, |
| LayerType.ELEMENTWISE - 247 | 1, |
| LayerType.CONSTANT - 248 | 1, |
| LayerType.ELEMENTWISE - 249 | 1, |
| LayerType.CONSTANT - 250 | 1, |
| LayerType.ELEMENTWISE - 251 | 1, |
| LayerType.CONSTANT - 252 | 4, |
| LayerType.CONCATENATION - 253 | 5, |
| LayerType.CONSTANT - 254 | 4, |
| LayerType.GATHER - 255 | 4, |
| LayerType.CONSTANT - 256 | 1, |
| LayerType.ELEMENTWISE - 257 | 4, |
| LayerType.CONSTANT - 258 | 1, |
| LayerType.ELEMENTWISE - 259 | 4, |
| LayerType.ELEMENTWISE - 260 | 4, |
| LayerType.ELEMENTWISE - 261 | 4, |
| LayerType.CONSTANT - 262 | 4, |
| LayerType.ELEMENTWISE - 263 | 4, |
| LayerType.ELEMENTWISE - 264 | 4, |
| LayerType.CONSTANT - 265 | 4, |
| LayerType.ELEMENTWISE - 266 | 4, |
| LayerType.SLICE - 267 | 1,128,7,7 |
| LayerType.CONSTANT - 268 | 1, |
| LayerType.ELEMENTWISE - 269 | 1, |
| LayerType.CONSTANT - 270 | 4, |
| LayerType.CONCATENATION - 271 | 5, |
| LayerType.CONSTANT - 272 | 4, |
| LayerType.GATHER - 273 | 4, |
| LayerType.CONSTANT - 274 | 4, |
| LayerType.CONCATENATION - 275 | 5, |
| LayerType.GATHER - 276 | 4, |
| LayerType.CONSTANT - 277 | 1, |
| LayerType.ELEMENTWISE - 278 | 4, |
| LayerType.CONSTANT - 279 | 1, |
| LayerType.ELEMENTWISE - 280 | 4, |
| LayerType.ELEMENTWISE - 281 | 4, |
| LayerType.ELEMENTWISE - 282 | 4, |
| LayerType.CONSTANT - 283 | 1, |
| LayerType.ELEMENTWISE - 284 | 4, |
| LayerType.ELEMENTWISE - 285 | 4, |
| LayerType.CONSTANT - 286 | 1, |
| LayerType.ELEMENTWISE - 287 | 4, |
| LayerType.CONSTANT - 288 | 1, |
| LayerType.ELEMENTWISE - 289 | 4, |
| LayerType.ELEMENTWISE - 290 | 4, |
| LayerType.ELEMENTWISE - 291 | 4, |
| LayerType.CONSTANT - 292 | 4, |
| LayerType.ELEMENTWISE - 293 | 4, |
| LayerType.ELEMENTWISE - 294 | 4, |
| LayerType.ELEMENTWISE - 295 | 4, |
| LayerType.CONSTANT - 296 | 4, |
| LayerType.ELEMENTWISE - 297 | 4, |
| LayerType.SLICE - 298 | 1,128,7,7 |
| LayerType.CONVOLUTION - 299 | 1,128,7,7 |
| ActivationType.SIGMOID - 300 | 1,128,7,7 |
| LayerType.ELEMENTWISE - 301 | 1,128,7,7 |
| LayerType.CONVOLUTION - 302 | 1,128,7,7 |
| ActivationType.SIGMOID - 303 | 1,128,7,7 |
| LayerType.ELEMENTWISE - 304 | 1,128,7,7 |
| LayerType.ELEMENTWISE - 305 | 1,128,7,7 |
| LayerType.CONCATENATION - 306 | 1,384,7,7 |
| LayerType.CONVOLUTION - 307 | 1,256,7,7 |
| ActivationType.SIGMOID - 308 | 1,256,7,7 |
| LayerType.ELEMENTWISE - 309 | 1,256,7,7 |
| LayerType.CONVOLUTION - 310 | 1,1280,7,7 |
| ActivationType.SIGMOID - 311 | 1,1280,7,7 |
| LayerType.ELEMENTWISE - 312 | 1,1280,7,7 |
| LayerType.REDUCE - 313 | 1,1280,1,1 |
| LayerType.SHUFFLE - 314 | 1,1280 |
| LayerType.CONSTANT - 315 | 1000,1280 |
| LayerType.MATRIX_MULTIPLY - 316 | 1,1000 |
| LayerType.CONSTANT - 317 | 1000, |
| LayerType.SHUFFLE - 318 | 1,1000 |
| LayerType.ELEMENTWISE - 319 | 1,1000 |
| LayerType.SHUFFLE - 320 | 1,1000 |
| LayerType.SOFTMAX - 321 | 1,1000 |
| LayerType.SHUFFLE - 322 | 1,1000 |

</details>

<details><summary> PTH SUMM  </summary> 

        Layer (type)               Output Shape         Param #
            Conv2d-1         [-1, 16, 112, 112]             448
              SiLU-2         [-1, 16, 112, 112]               0
              SiLU-3         [-1, 16, 112, 112]               0
              SiLU-4         [-1, 16, 112, 112]               0
              SiLU-5         [-1, 16, 112, 112]               0
              SiLU-6         [-1, 16, 112, 112]               0
              SiLU-7         [-1, 16, 112, 112]               0
              SiLU-8         [-1, 16, 112, 112]               0
              SiLU-9         [-1, 16, 112, 112]               0
             SiLU-10         [-1, 16, 112, 112]               0
             SiLU-11         [-1, 16, 112, 112]               0
             SiLU-12         [-1, 16, 112, 112]               0
             SiLU-13         [-1, 16, 112, 112]               0
             SiLU-14         [-1, 16, 112, 112]               0
             SiLU-15         [-1, 16, 112, 112]               0
             SiLU-16         [-1, 16, 112, 112]               0
             SiLU-17         [-1, 16, 112, 112]               0
             SiLU-18         [-1, 16, 112, 112]               0
             SiLU-19         [-1, 16, 112, 112]               0
             SiLU-20         [-1, 16, 112, 112]               0
             SiLU-21         [-1, 16, 112, 112]               0
             SiLU-22         [-1, 16, 112, 112]               0
             SiLU-23         [-1, 16, 112, 112]               0
             SiLU-24         [-1, 16, 112, 112]               0
             SiLU-25         [-1, 16, 112, 112]               0
             SiLU-26         [-1, 16, 112, 112]               0
             SiLU-27         [-1, 16, 112, 112]               0
             Conv-28         [-1, 16, 112, 112]               0
           Conv2d-29           [-1, 32, 56, 56]           4,640
             SiLU-30           [-1, 32, 56, 56]               0
             SiLU-31           [-1, 32, 56, 56]               0
             SiLU-32           [-1, 32, 56, 56]               0
             SiLU-33           [-1, 32, 56, 56]               0
             SiLU-34           [-1, 32, 56, 56]               0
             SiLU-35           [-1, 32, 56, 56]               0
             SiLU-36           [-1, 32, 56, 56]               0
             SiLU-37           [-1, 32, 56, 56]               0
             SiLU-38           [-1, 32, 56, 56]               0
             SiLU-39           [-1, 32, 56, 56]               0
             SiLU-40           [-1, 32, 56, 56]               0
             SiLU-41           [-1, 32, 56, 56]               0
             SiLU-42           [-1, 32, 56, 56]               0
             SiLU-43           [-1, 32, 56, 56]               0
             SiLU-44           [-1, 32, 56, 56]               0
             SiLU-45           [-1, 32, 56, 56]               0
             SiLU-46           [-1, 32, 56, 56]               0
             SiLU-47           [-1, 32, 56, 56]               0
             SiLU-48           [-1, 32, 56, 56]               0
             SiLU-49           [-1, 32, 56, 56]               0
             SiLU-50           [-1, 32, 56, 56]               0
             SiLU-51           [-1, 32, 56, 56]               0
             SiLU-52           [-1, 32, 56, 56]               0
             SiLU-53           [-1, 32, 56, 56]               0
             SiLU-54           [-1, 32, 56, 56]               0
             SiLU-55           [-1, 32, 56, 56]               0
             Conv-56           [-1, 32, 56, 56]               0
           Conv2d-57           [-1, 32, 56, 56]           1,056
             SiLU-58           [-1, 32, 56, 56]               0
             SiLU-59           [-1, 32, 56, 56]               0
             SiLU-60           [-1, 32, 56, 56]               0
             SiLU-61           [-1, 32, 56, 56]               0
             SiLU-62           [-1, 32, 56, 56]               0
             SiLU-63           [-1, 32, 56, 56]               0
             SiLU-64           [-1, 32, 56, 56]               0
             SiLU-65           [-1, 32, 56, 56]               0
             SiLU-66           [-1, 32, 56, 56]               0
             SiLU-67           [-1, 32, 56, 56]               0
             SiLU-68           [-1, 32, 56, 56]               0
             SiLU-69           [-1, 32, 56, 56]               0
             SiLU-70           [-1, 32, 56, 56]               0
             SiLU-71           [-1, 32, 56, 56]               0
             SiLU-72           [-1, 32, 56, 56]               0
             SiLU-73           [-1, 32, 56, 56]               0
             SiLU-74           [-1, 32, 56, 56]               0
             SiLU-75           [-1, 32, 56, 56]               0
             SiLU-76           [-1, 32, 56, 56]               0
             SiLU-77           [-1, 32, 56, 56]               0
             SiLU-78           [-1, 32, 56, 56]               0
             SiLU-79           [-1, 32, 56, 56]               0
             SiLU-80           [-1, 32, 56, 56]               0
             SiLU-81           [-1, 32, 56, 56]               0
             SiLU-82           [-1, 32, 56, 56]               0
             SiLU-83           [-1, 32, 56, 56]               0
             Conv-84           [-1, 32, 56, 56]               0
           Conv2d-85           [-1, 16, 56, 56]           2,320
             SiLU-86           [-1, 16, 56, 56]               0
             SiLU-87           [-1, 16, 56, 56]               0
             SiLU-88           [-1, 16, 56, 56]               0
             SiLU-89           [-1, 16, 56, 56]               0
             SiLU-90           [-1, 16, 56, 56]               0
             SiLU-91           [-1, 16, 56, 56]               0
             SiLU-92           [-1, 16, 56, 56]               0
             SiLU-93           [-1, 16, 56, 56]               0
             SiLU-94           [-1, 16, 56, 56]               0
             SiLU-95           [-1, 16, 56, 56]               0
             SiLU-96           [-1, 16, 56, 56]               0
             SiLU-97           [-1, 16, 56, 56]               0
             SiLU-98           [-1, 16, 56, 56]               0
             SiLU-99           [-1, 16, 56, 56]               0
            SiLU-100           [-1, 16, 56, 56]               0
            SiLU-101           [-1, 16, 56, 56]               0
            SiLU-102           [-1, 16, 56, 56]               0
            SiLU-103           [-1, 16, 56, 56]               0
            SiLU-104           [-1, 16, 56, 56]               0
            SiLU-105           [-1, 16, 56, 56]               0
            SiLU-106           [-1, 16, 56, 56]               0
            SiLU-107           [-1, 16, 56, 56]               0
            SiLU-108           [-1, 16, 56, 56]               0
            SiLU-109           [-1, 16, 56, 56]               0
            SiLU-110           [-1, 16, 56, 56]               0
            SiLU-111           [-1, 16, 56, 56]               0
            Conv-112           [-1, 16, 56, 56]               0
          Conv2d-113           [-1, 16, 56, 56]           2,320
            SiLU-114           [-1, 16, 56, 56]               0
            SiLU-115           [-1, 16, 56, 56]               0
            SiLU-116           [-1, 16, 56, 56]               0
            SiLU-117           [-1, 16, 56, 56]               0
            SiLU-118           [-1, 16, 56, 56]               0
            SiLU-119           [-1, 16, 56, 56]               0
            SiLU-120           [-1, 16, 56, 56]               0
            SiLU-121           [-1, 16, 56, 56]               0
            SiLU-122           [-1, 16, 56, 56]               0
            SiLU-123           [-1, 16, 56, 56]               0
            SiLU-124           [-1, 16, 56, 56]               0
            SiLU-125           [-1, 16, 56, 56]               0
            SiLU-126           [-1, 16, 56, 56]               0
            SiLU-127           [-1, 16, 56, 56]               0
            SiLU-128           [-1, 16, 56, 56]               0
            SiLU-129           [-1, 16, 56, 56]               0
            SiLU-130           [-1, 16, 56, 56]               0
            SiLU-131           [-1, 16, 56, 56]               0
            SiLU-132           [-1, 16, 56, 56]               0
            SiLU-133           [-1, 16, 56, 56]               0
            SiLU-134           [-1, 16, 56, 56]               0
            SiLU-135           [-1, 16, 56, 56]               0
            SiLU-136           [-1, 16, 56, 56]               0
            SiLU-137           [-1, 16, 56, 56]               0
            SiLU-138           [-1, 16, 56, 56]               0
            SiLU-139           [-1, 16, 56, 56]               0
            Conv-140           [-1, 16, 56, 56]               0
      Bottleneck-141           [-1, 16, 56, 56]               0
          Conv2d-142           [-1, 32, 56, 56]           1,568
            SiLU-143           [-1, 32, 56, 56]               0
            SiLU-144           [-1, 32, 56, 56]               0
            SiLU-145           [-1, 32, 56, 56]               0
            SiLU-146           [-1, 32, 56, 56]               0
            SiLU-147           [-1, 32, 56, 56]               0
            SiLU-148           [-1, 32, 56, 56]               0
            SiLU-149           [-1, 32, 56, 56]               0
            SiLU-150           [-1, 32, 56, 56]               0
            SiLU-151           [-1, 32, 56, 56]               0
            SiLU-152           [-1, 32, 56, 56]               0
            SiLU-153           [-1, 32, 56, 56]               0
            SiLU-154           [-1, 32, 56, 56]               0
            SiLU-155           [-1, 32, 56, 56]               0
            SiLU-156           [-1, 32, 56, 56]               0
            SiLU-157           [-1, 32, 56, 56]               0
            SiLU-158           [-1, 32, 56, 56]               0
            SiLU-159           [-1, 32, 56, 56]               0
            SiLU-160           [-1, 32, 56, 56]               0
            SiLU-161           [-1, 32, 56, 56]               0
            SiLU-162           [-1, 32, 56, 56]               0
            SiLU-163           [-1, 32, 56, 56]               0
            SiLU-164           [-1, 32, 56, 56]               0
            SiLU-165           [-1, 32, 56, 56]               0
            SiLU-166           [-1, 32, 56, 56]               0
            SiLU-167           [-1, 32, 56, 56]               0
            SiLU-168           [-1, 32, 56, 56]               0
            Conv-169           [-1, 32, 56, 56]               0
             C2f-170           [-1, 32, 56, 56]               0
          Conv2d-171           [-1, 64, 28, 28]          18,496
            SiLU-172           [-1, 64, 28, 28]               0
            SiLU-173           [-1, 64, 28, 28]               0
            SiLU-174           [-1, 64, 28, 28]               0
            SiLU-175           [-1, 64, 28, 28]               0
            SiLU-176           [-1, 64, 28, 28]               0
            SiLU-177           [-1, 64, 28, 28]               0
            SiLU-178           [-1, 64, 28, 28]               0
            SiLU-179           [-1, 64, 28, 28]               0
            SiLU-180           [-1, 64, 28, 28]               0
            SiLU-181           [-1, 64, 28, 28]               0
            SiLU-182           [-1, 64, 28, 28]               0
            SiLU-183           [-1, 64, 28, 28]               0
            SiLU-184           [-1, 64, 28, 28]               0
            SiLU-185           [-1, 64, 28, 28]               0
            SiLU-186           [-1, 64, 28, 28]               0
            SiLU-187           [-1, 64, 28, 28]               0
            SiLU-188           [-1, 64, 28, 28]               0
            SiLU-189           [-1, 64, 28, 28]               0
            SiLU-190           [-1, 64, 28, 28]               0
            SiLU-191           [-1, 64, 28, 28]               0
            SiLU-192           [-1, 64, 28, 28]               0
            SiLU-193           [-1, 64, 28, 28]               0
            SiLU-194           [-1, 64, 28, 28]               0
            SiLU-195           [-1, 64, 28, 28]               0
            SiLU-196           [-1, 64, 28, 28]               0
            SiLU-197           [-1, 64, 28, 28]               0
            Conv-198           [-1, 64, 28, 28]               0
          Conv2d-199           [-1, 64, 28, 28]           4,160
            SiLU-200           [-1, 64, 28, 28]               0
            SiLU-201           [-1, 64, 28, 28]               0
            SiLU-202           [-1, 64, 28, 28]               0
            SiLU-203           [-1, 64, 28, 28]               0
            SiLU-204           [-1, 64, 28, 28]               0
            SiLU-205           [-1, 64, 28, 28]               0
            SiLU-206           [-1, 64, 28, 28]               0
            SiLU-207           [-1, 64, 28, 28]               0
            SiLU-208           [-1, 64, 28, 28]               0
            SiLU-209           [-1, 64, 28, 28]               0
            SiLU-210           [-1, 64, 28, 28]               0
            SiLU-211           [-1, 64, 28, 28]               0
            SiLU-212           [-1, 64, 28, 28]               0
            SiLU-213           [-1, 64, 28, 28]               0
            SiLU-214           [-1, 64, 28, 28]               0
            SiLU-215           [-1, 64, 28, 28]               0
            SiLU-216           [-1, 64, 28, 28]               0
            SiLU-217           [-1, 64, 28, 28]               0
            SiLU-218           [-1, 64, 28, 28]               0
            SiLU-219           [-1, 64, 28, 28]               0
            SiLU-220           [-1, 64, 28, 28]               0
            SiLU-221           [-1, 64, 28, 28]               0
            SiLU-222           [-1, 64, 28, 28]               0
            SiLU-223           [-1, 64, 28, 28]               0
            SiLU-224           [-1, 64, 28, 28]               0
            SiLU-225           [-1, 64, 28, 28]               0
            Conv-226           [-1, 64, 28, 28]               0
          Conv2d-227           [-1, 32, 28, 28]           9,248
            SiLU-228           [-1, 32, 28, 28]               0
            SiLU-229           [-1, 32, 28, 28]               0
            SiLU-230           [-1, 32, 28, 28]               0
            SiLU-231           [-1, 32, 28, 28]               0
            SiLU-232           [-1, 32, 28, 28]               0
            SiLU-233           [-1, 32, 28, 28]               0
            SiLU-234           [-1, 32, 28, 28]               0
            SiLU-235           [-1, 32, 28, 28]               0
            SiLU-236           [-1, 32, 28, 28]               0
            SiLU-237           [-1, 32, 28, 28]               0
            SiLU-238           [-1, 32, 28, 28]               0
            SiLU-239           [-1, 32, 28, 28]               0
            SiLU-240           [-1, 32, 28, 28]               0
            SiLU-241           [-1, 32, 28, 28]               0
            SiLU-242           [-1, 32, 28, 28]               0
            SiLU-243           [-1, 32, 28, 28]               0
            SiLU-244           [-1, 32, 28, 28]               0
            SiLU-245           [-1, 32, 28, 28]               0
            SiLU-246           [-1, 32, 28, 28]               0
            SiLU-247           [-1, 32, 28, 28]               0
            SiLU-248           [-1, 32, 28, 28]               0
            SiLU-249           [-1, 32, 28, 28]               0
            SiLU-250           [-1, 32, 28, 28]               0
            SiLU-251           [-1, 32, 28, 28]               0
            SiLU-252           [-1, 32, 28, 28]               0
            SiLU-253           [-1, 32, 28, 28]               0
            Conv-254           [-1, 32, 28, 28]               0
          Conv2d-255           [-1, 32, 28, 28]           9,248
            SiLU-256           [-1, 32, 28, 28]               0
            SiLU-257           [-1, 32, 28, 28]               0
            SiLU-258           [-1, 32, 28, 28]               0
            SiLU-259           [-1, 32, 28, 28]               0
            SiLU-260           [-1, 32, 28, 28]               0
            SiLU-261           [-1, 32, 28, 28]               0
            SiLU-262           [-1, 32, 28, 28]               0
            SiLU-263           [-1, 32, 28, 28]               0
            SiLU-264           [-1, 32, 28, 28]               0
            SiLU-265           [-1, 32, 28, 28]               0
            SiLU-266           [-1, 32, 28, 28]               0
            SiLU-267           [-1, 32, 28, 28]               0
            SiLU-268           [-1, 32, 28, 28]               0
            SiLU-269           [-1, 32, 28, 28]               0
            SiLU-270           [-1, 32, 28, 28]               0
            SiLU-271           [-1, 32, 28, 28]               0
            SiLU-272           [-1, 32, 28, 28]               0
            SiLU-273           [-1, 32, 28, 28]               0
            SiLU-274           [-1, 32, 28, 28]               0
            SiLU-275           [-1, 32, 28, 28]               0
            SiLU-276           [-1, 32, 28, 28]               0
            SiLU-277           [-1, 32, 28, 28]               0
            SiLU-278           [-1, 32, 28, 28]               0
            SiLU-279           [-1, 32, 28, 28]               0
            SiLU-280           [-1, 32, 28, 28]               0
            SiLU-281           [-1, 32, 28, 28]               0
            Conv-282           [-1, 32, 28, 28]               0
      Bottleneck-283           [-1, 32, 28, 28]               0
          Conv2d-284           [-1, 32, 28, 28]           9,248
            SiLU-285           [-1, 32, 28, 28]               0
            SiLU-286           [-1, 32, 28, 28]               0
            SiLU-287           [-1, 32, 28, 28]               0
            SiLU-288           [-1, 32, 28, 28]               0
            SiLU-289           [-1, 32, 28, 28]               0
            SiLU-290           [-1, 32, 28, 28]               0
            SiLU-291           [-1, 32, 28, 28]               0
            SiLU-292           [-1, 32, 28, 28]               0
            SiLU-293           [-1, 32, 28, 28]               0
            SiLU-294           [-1, 32, 28, 28]               0
            SiLU-295           [-1, 32, 28, 28]               0
            SiLU-296           [-1, 32, 28, 28]               0
            SiLU-297           [-1, 32, 28, 28]               0
            SiLU-298           [-1, 32, 28, 28]               0
            SiLU-299           [-1, 32, 28, 28]               0
            SiLU-300           [-1, 32, 28, 28]               0
            SiLU-301           [-1, 32, 28, 28]               0
            SiLU-302           [-1, 32, 28, 28]               0
            SiLU-303           [-1, 32, 28, 28]               0
            SiLU-304           [-1, 32, 28, 28]               0
            SiLU-305           [-1, 32, 28, 28]               0
            SiLU-306           [-1, 32, 28, 28]               0
            SiLU-307           [-1, 32, 28, 28]               0
            SiLU-308           [-1, 32, 28, 28]               0
            SiLU-309           [-1, 32, 28, 28]               0
            SiLU-310           [-1, 32, 28, 28]               0
            Conv-311           [-1, 32, 28, 28]               0
          Conv2d-312           [-1, 32, 28, 28]           9,248
            SiLU-313           [-1, 32, 28, 28]               0
            SiLU-314           [-1, 32, 28, 28]               0
            SiLU-315           [-1, 32, 28, 28]               0
            SiLU-316           [-1, 32, 28, 28]               0
            SiLU-317           [-1, 32, 28, 28]               0
            SiLU-318           [-1, 32, 28, 28]               0
            SiLU-319           [-1, 32, 28, 28]               0
            SiLU-320           [-1, 32, 28, 28]               0
            SiLU-321           [-1, 32, 28, 28]               0
            SiLU-322           [-1, 32, 28, 28]               0
            SiLU-323           [-1, 32, 28, 28]               0
            SiLU-324           [-1, 32, 28, 28]               0
            SiLU-325           [-1, 32, 28, 28]               0
            SiLU-326           [-1, 32, 28, 28]               0
            SiLU-327           [-1, 32, 28, 28]               0
            SiLU-328           [-1, 32, 28, 28]               0
            SiLU-329           [-1, 32, 28, 28]               0
            SiLU-330           [-1, 32, 28, 28]               0
            SiLU-331           [-1, 32, 28, 28]               0
            SiLU-332           [-1, 32, 28, 28]               0
            SiLU-333           [-1, 32, 28, 28]               0
            SiLU-334           [-1, 32, 28, 28]               0
            SiLU-335           [-1, 32, 28, 28]               0
            SiLU-336           [-1, 32, 28, 28]               0
            SiLU-337           [-1, 32, 28, 28]               0
            SiLU-338           [-1, 32, 28, 28]               0
            Conv-339           [-1, 32, 28, 28]               0
      Bottleneck-340           [-1, 32, 28, 28]               0
          Conv2d-341           [-1, 64, 28, 28]           8,256
            SiLU-342           [-1, 64, 28, 28]               0
            SiLU-343           [-1, 64, 28, 28]               0
            SiLU-344           [-1, 64, 28, 28]               0
            SiLU-345           [-1, 64, 28, 28]               0
            SiLU-346           [-1, 64, 28, 28]               0
            SiLU-347           [-1, 64, 28, 28]               0
            SiLU-348           [-1, 64, 28, 28]               0
            SiLU-349           [-1, 64, 28, 28]               0
            SiLU-350           [-1, 64, 28, 28]               0
            SiLU-351           [-1, 64, 28, 28]               0
            SiLU-352           [-1, 64, 28, 28]               0
            SiLU-353           [-1, 64, 28, 28]               0
            SiLU-354           [-1, 64, 28, 28]               0
            SiLU-355           [-1, 64, 28, 28]               0
            SiLU-356           [-1, 64, 28, 28]               0
            SiLU-357           [-1, 64, 28, 28]               0
            SiLU-358           [-1, 64, 28, 28]               0
            SiLU-359           [-1, 64, 28, 28]               0
            SiLU-360           [-1, 64, 28, 28]               0
            SiLU-361           [-1, 64, 28, 28]               0
            SiLU-362           [-1, 64, 28, 28]               0
            SiLU-363           [-1, 64, 28, 28]               0
            SiLU-364           [-1, 64, 28, 28]               0
            SiLU-365           [-1, 64, 28, 28]               0
            SiLU-366           [-1, 64, 28, 28]               0
            SiLU-367           [-1, 64, 28, 28]               0
            Conv-368           [-1, 64, 28, 28]               0
             C2f-369           [-1, 64, 28, 28]               0
          Conv2d-370          [-1, 128, 14, 14]          73,856
            SiLU-371          [-1, 128, 14, 14]               0
            SiLU-372          [-1, 128, 14, 14]               0
            SiLU-373          [-1, 128, 14, 14]               0
            SiLU-374          [-1, 128, 14, 14]               0
            SiLU-375          [-1, 128, 14, 14]               0
            SiLU-376          [-1, 128, 14, 14]               0
            SiLU-377          [-1, 128, 14, 14]               0
            SiLU-378          [-1, 128, 14, 14]               0
            SiLU-379          [-1, 128, 14, 14]               0
            SiLU-380          [-1, 128, 14, 14]               0
            SiLU-381          [-1, 128, 14, 14]               0
            SiLU-382          [-1, 128, 14, 14]               0
            SiLU-383          [-1, 128, 14, 14]               0
            SiLU-384          [-1, 128, 14, 14]               0
            SiLU-385          [-1, 128, 14, 14]               0
            SiLU-386          [-1, 128, 14, 14]               0
            SiLU-387          [-1, 128, 14, 14]               0
            SiLU-388          [-1, 128, 14, 14]               0
            SiLU-389          [-1, 128, 14, 14]               0
            SiLU-390          [-1, 128, 14, 14]               0
            SiLU-391          [-1, 128, 14, 14]               0
            SiLU-392          [-1, 128, 14, 14]               0
            SiLU-393          [-1, 128, 14, 14]               0
            SiLU-394          [-1, 128, 14, 14]               0
            SiLU-395          [-1, 128, 14, 14]               0
            SiLU-396          [-1, 128, 14, 14]               0
            Conv-397          [-1, 128, 14, 14]               0
          Conv2d-398          [-1, 128, 14, 14]          16,512
            SiLU-399          [-1, 128, 14, 14]               0
            SiLU-400          [-1, 128, 14, 14]               0
            SiLU-401          [-1, 128, 14, 14]               0
            SiLU-402          [-1, 128, 14, 14]               0
            SiLU-403          [-1, 128, 14, 14]               0
            SiLU-404          [-1, 128, 14, 14]               0
            SiLU-405          [-1, 128, 14, 14]               0
            SiLU-406          [-1, 128, 14, 14]               0
            SiLU-407          [-1, 128, 14, 14]               0
            SiLU-408          [-1, 128, 14, 14]               0
            SiLU-409          [-1, 128, 14, 14]               0
            SiLU-410          [-1, 128, 14, 14]               0
            SiLU-411          [-1, 128, 14, 14]               0
            SiLU-412          [-1, 128, 14, 14]               0
            SiLU-413          [-1, 128, 14, 14]               0
            SiLU-414          [-1, 128, 14, 14]               0
            SiLU-415          [-1, 128, 14, 14]               0
            SiLU-416          [-1, 128, 14, 14]               0
            SiLU-417          [-1, 128, 14, 14]               0
            SiLU-418          [-1, 128, 14, 14]               0
            SiLU-419          [-1, 128, 14, 14]               0
            SiLU-420          [-1, 128, 14, 14]               0
            SiLU-421          [-1, 128, 14, 14]               0
            SiLU-422          [-1, 128, 14, 14]               0
            SiLU-423          [-1, 128, 14, 14]               0
            SiLU-424          [-1, 128, 14, 14]               0
            Conv-425          [-1, 128, 14, 14]               0
          Conv2d-426           [-1, 64, 14, 14]          36,928
            SiLU-427           [-1, 64, 14, 14]               0
            SiLU-428           [-1, 64, 14, 14]               0
            SiLU-429           [-1, 64, 14, 14]               0
            SiLU-430           [-1, 64, 14, 14]               0
            SiLU-431           [-1, 64, 14, 14]               0
            SiLU-432           [-1, 64, 14, 14]               0
            SiLU-433           [-1, 64, 14, 14]               0
            SiLU-434           [-1, 64, 14, 14]               0
            SiLU-435           [-1, 64, 14, 14]               0
            SiLU-436           [-1, 64, 14, 14]               0
            SiLU-437           [-1, 64, 14, 14]               0
            SiLU-438           [-1, 64, 14, 14]               0
            SiLU-439           [-1, 64, 14, 14]               0
            SiLU-440           [-1, 64, 14, 14]               0
            SiLU-441           [-1, 64, 14, 14]               0
            SiLU-442           [-1, 64, 14, 14]               0
            SiLU-443           [-1, 64, 14, 14]               0
            SiLU-444           [-1, 64, 14, 14]               0
            SiLU-445           [-1, 64, 14, 14]               0
            SiLU-446           [-1, 64, 14, 14]               0
            SiLU-447           [-1, 64, 14, 14]               0
            SiLU-448           [-1, 64, 14, 14]               0
            SiLU-449           [-1, 64, 14, 14]               0
            SiLU-450           [-1, 64, 14, 14]               0
            SiLU-451           [-1, 64, 14, 14]               0
            SiLU-452           [-1, 64, 14, 14]               0
            Conv-453           [-1, 64, 14, 14]               0
          Conv2d-454           [-1, 64, 14, 14]          36,928
            SiLU-455           [-1, 64, 14, 14]               0
            SiLU-456           [-1, 64, 14, 14]               0
            SiLU-457           [-1, 64, 14, 14]               0
            SiLU-458           [-1, 64, 14, 14]               0
            SiLU-459           [-1, 64, 14, 14]               0
            SiLU-460           [-1, 64, 14, 14]               0
            SiLU-461           [-1, 64, 14, 14]               0
            SiLU-462           [-1, 64, 14, 14]               0
            SiLU-463           [-1, 64, 14, 14]               0
            SiLU-464           [-1, 64, 14, 14]               0
            SiLU-465           [-1, 64, 14, 14]               0
            SiLU-466           [-1, 64, 14, 14]               0
            SiLU-467           [-1, 64, 14, 14]               0
            SiLU-468           [-1, 64, 14, 14]               0
            SiLU-469           [-1, 64, 14, 14]               0
            SiLU-470           [-1, 64, 14, 14]               0
            SiLU-471           [-1, 64, 14, 14]               0
            SiLU-472           [-1, 64, 14, 14]               0
            SiLU-473           [-1, 64, 14, 14]               0
            SiLU-474           [-1, 64, 14, 14]               0
            SiLU-475           [-1, 64, 14, 14]               0
            SiLU-476           [-1, 64, 14, 14]               0
            SiLU-477           [-1, 64, 14, 14]               0
            SiLU-478           [-1, 64, 14, 14]               0
            SiLU-479           [-1, 64, 14, 14]               0
            SiLU-480           [-1, 64, 14, 14]               0
            Conv-481           [-1, 64, 14, 14]               0
      Bottleneck-482           [-1, 64, 14, 14]               0
          Conv2d-483           [-1, 64, 14, 14]          36,928
            SiLU-484           [-1, 64, 14, 14]               0
            SiLU-485           [-1, 64, 14, 14]               0
            SiLU-486           [-1, 64, 14, 14]               0
            SiLU-487           [-1, 64, 14, 14]               0
            SiLU-488           [-1, 64, 14, 14]               0
            SiLU-489           [-1, 64, 14, 14]               0
            SiLU-490           [-1, 64, 14, 14]               0
            SiLU-491           [-1, 64, 14, 14]               0
            SiLU-492           [-1, 64, 14, 14]               0
            SiLU-493           [-1, 64, 14, 14]               0
            SiLU-494           [-1, 64, 14, 14]               0
            SiLU-495           [-1, 64, 14, 14]               0
            SiLU-496           [-1, 64, 14, 14]               0
            SiLU-497           [-1, 64, 14, 14]               0
            SiLU-498           [-1, 64, 14, 14]               0
            SiLU-499           [-1, 64, 14, 14]               0
            SiLU-500           [-1, 64, 14, 14]               0
            SiLU-501           [-1, 64, 14, 14]               0
            SiLU-502           [-1, 64, 14, 14]               0
            SiLU-503           [-1, 64, 14, 14]               0
            SiLU-504           [-1, 64, 14, 14]               0
            SiLU-505           [-1, 64, 14, 14]               0
            SiLU-506           [-1, 64, 14, 14]               0
            SiLU-507           [-1, 64, 14, 14]               0
            SiLU-508           [-1, 64, 14, 14]               0
            SiLU-509           [-1, 64, 14, 14]               0
            Conv-510           [-1, 64, 14, 14]               0
          Conv2d-511           [-1, 64, 14, 14]          36,928
            SiLU-512           [-1, 64, 14, 14]               0
            SiLU-513           [-1, 64, 14, 14]               0
            SiLU-514           [-1, 64, 14, 14]               0
            SiLU-515           [-1, 64, 14, 14]               0
            SiLU-516           [-1, 64, 14, 14]               0
            SiLU-517           [-1, 64, 14, 14]               0
            SiLU-518           [-1, 64, 14, 14]               0
            SiLU-519           [-1, 64, 14, 14]               0
            SiLU-520           [-1, 64, 14, 14]               0
            SiLU-521           [-1, 64, 14, 14]               0
            SiLU-522           [-1, 64, 14, 14]               0
            SiLU-523           [-1, 64, 14, 14]               0
            SiLU-524           [-1, 64, 14, 14]               0
            SiLU-525           [-1, 64, 14, 14]               0
            SiLU-526           [-1, 64, 14, 14]               0
            SiLU-527           [-1, 64, 14, 14]               0
            SiLU-528           [-1, 64, 14, 14]               0
            SiLU-529           [-1, 64, 14, 14]               0
            SiLU-530           [-1, 64, 14, 14]               0
            SiLU-531           [-1, 64, 14, 14]               0
            SiLU-532           [-1, 64, 14, 14]               0
            SiLU-533           [-1, 64, 14, 14]               0
            SiLU-534           [-1, 64, 14, 14]               0
            SiLU-535           [-1, 64, 14, 14]               0
            SiLU-536           [-1, 64, 14, 14]               0
            SiLU-537           [-1, 64, 14, 14]               0
            Conv-538           [-1, 64, 14, 14]               0
      Bottleneck-539           [-1, 64, 14, 14]               0
          Conv2d-540          [-1, 128, 14, 14]          32,896
            SiLU-541          [-1, 128, 14, 14]               0
            SiLU-542          [-1, 128, 14, 14]               0
            SiLU-543          [-1, 128, 14, 14]               0
            SiLU-544          [-1, 128, 14, 14]               0
            SiLU-545          [-1, 128, 14, 14]               0
            SiLU-546          [-1, 128, 14, 14]               0
            SiLU-547          [-1, 128, 14, 14]               0
            SiLU-548          [-1, 128, 14, 14]               0
            SiLU-549          [-1, 128, 14, 14]               0
            SiLU-550          [-1, 128, 14, 14]               0
            SiLU-551          [-1, 128, 14, 14]               0
            SiLU-552          [-1, 128, 14, 14]               0
            SiLU-553          [-1, 128, 14, 14]               0
            SiLU-554          [-1, 128, 14, 14]               0
            SiLU-555          [-1, 128, 14, 14]               0
            SiLU-556          [-1, 128, 14, 14]               0
            SiLU-557          [-1, 128, 14, 14]               0
            SiLU-558          [-1, 128, 14, 14]               0
            SiLU-559          [-1, 128, 14, 14]               0
            SiLU-560          [-1, 128, 14, 14]               0
            SiLU-561          [-1, 128, 14, 14]               0
            SiLU-562          [-1, 128, 14, 14]               0
            SiLU-563          [-1, 128, 14, 14]               0
            SiLU-564          [-1, 128, 14, 14]               0
            SiLU-565          [-1, 128, 14, 14]               0
            SiLU-566          [-1, 128, 14, 14]               0
            Conv-567          [-1, 128, 14, 14]               0
             C2f-568          [-1, 128, 14, 14]               0
          Conv2d-569            [-1, 256, 7, 7]         295,168
            SiLU-570            [-1, 256, 7, 7]               0
            SiLU-571            [-1, 256, 7, 7]               0
            SiLU-572            [-1, 256, 7, 7]               0
            SiLU-573            [-1, 256, 7, 7]               0
            SiLU-574            [-1, 256, 7, 7]               0
            SiLU-575            [-1, 256, 7, 7]               0
            SiLU-576            [-1, 256, 7, 7]               0
            SiLU-577            [-1, 256, 7, 7]               0
            SiLU-578            [-1, 256, 7, 7]               0
            SiLU-579            [-1, 256, 7, 7]               0
            SiLU-580            [-1, 256, 7, 7]               0
            SiLU-581            [-1, 256, 7, 7]               0
            SiLU-582            [-1, 256, 7, 7]               0
            SiLU-583            [-1, 256, 7, 7]               0
            SiLU-584            [-1, 256, 7, 7]               0
            SiLU-585            [-1, 256, 7, 7]               0
            SiLU-586            [-1, 256, 7, 7]               0
            SiLU-587            [-1, 256, 7, 7]               0
            SiLU-588            [-1, 256, 7, 7]               0
            SiLU-589            [-1, 256, 7, 7]               0
            SiLU-590            [-1, 256, 7, 7]               0
            SiLU-591            [-1, 256, 7, 7]               0
            SiLU-592            [-1, 256, 7, 7]               0
            SiLU-593            [-1, 256, 7, 7]               0
            SiLU-594            [-1, 256, 7, 7]               0
            SiLU-595            [-1, 256, 7, 7]               0
            Conv-596            [-1, 256, 7, 7]               0
          Conv2d-597            [-1, 256, 7, 7]          65,792
            SiLU-598            [-1, 256, 7, 7]               0
            SiLU-599            [-1, 256, 7, 7]               0
            SiLU-600            [-1, 256, 7, 7]               0
            SiLU-601            [-1, 256, 7, 7]               0
            SiLU-602            [-1, 256, 7, 7]               0
            SiLU-603            [-1, 256, 7, 7]               0
            SiLU-604            [-1, 256, 7, 7]               0
            SiLU-605            [-1, 256, 7, 7]               0
            SiLU-606            [-1, 256, 7, 7]               0
            SiLU-607            [-1, 256, 7, 7]               0
            SiLU-608            [-1, 256, 7, 7]               0
            SiLU-609            [-1, 256, 7, 7]               0
            SiLU-610            [-1, 256, 7, 7]               0
            SiLU-611            [-1, 256, 7, 7]               0
            SiLU-612            [-1, 256, 7, 7]               0
            SiLU-613            [-1, 256, 7, 7]               0
            SiLU-614            [-1, 256, 7, 7]               0
            SiLU-615            [-1, 256, 7, 7]               0
            SiLU-616            [-1, 256, 7, 7]               0
            SiLU-617            [-1, 256, 7, 7]               0
            SiLU-618            [-1, 256, 7, 7]               0
            SiLU-619            [-1, 256, 7, 7]               0
            SiLU-620            [-1, 256, 7, 7]               0
            SiLU-621            [-1, 256, 7, 7]               0
            SiLU-622            [-1, 256, 7, 7]               0
            SiLU-623            [-1, 256, 7, 7]               0
            Conv-624            [-1, 256, 7, 7]               0
          Conv2d-625            [-1, 128, 7, 7]         147,584
            SiLU-626            [-1, 128, 7, 7]               0
            SiLU-627            [-1, 128, 7, 7]               0
            SiLU-628            [-1, 128, 7, 7]               0
            SiLU-629            [-1, 128, 7, 7]               0
            SiLU-630            [-1, 128, 7, 7]               0
            SiLU-631            [-1, 128, 7, 7]               0
            SiLU-632            [-1, 128, 7, 7]               0
            SiLU-633            [-1, 128, 7, 7]               0
            SiLU-634            [-1, 128, 7, 7]               0
            SiLU-635            [-1, 128, 7, 7]               0
            SiLU-636            [-1, 128, 7, 7]               0
            SiLU-637            [-1, 128, 7, 7]               0
            SiLU-638            [-1, 128, 7, 7]               0
            SiLU-639            [-1, 128, 7, 7]               0
            SiLU-640            [-1, 128, 7, 7]               0
            SiLU-641            [-1, 128, 7, 7]               0
            SiLU-642            [-1, 128, 7, 7]               0
            SiLU-643            [-1, 128, 7, 7]               0
            SiLU-644            [-1, 128, 7, 7]               0
            SiLU-645            [-1, 128, 7, 7]               0
            SiLU-646            [-1, 128, 7, 7]               0
            SiLU-647            [-1, 128, 7, 7]               0
            SiLU-648            [-1, 128, 7, 7]               0
            SiLU-649            [-1, 128, 7, 7]               0
            SiLU-650            [-1, 128, 7, 7]               0
            SiLU-651            [-1, 128, 7, 7]               0
            Conv-652            [-1, 128, 7, 7]               0
          Conv2d-653            [-1, 128, 7, 7]         147,584
            SiLU-654            [-1, 128, 7, 7]               0
            SiLU-655            [-1, 128, 7, 7]               0
            SiLU-656            [-1, 128, 7, 7]               0
            SiLU-657            [-1, 128, 7, 7]               0
            SiLU-658            [-1, 128, 7, 7]               0
            SiLU-659            [-1, 128, 7, 7]               0
            SiLU-660            [-1, 128, 7, 7]               0
            SiLU-661            [-1, 128, 7, 7]               0
            SiLU-662            [-1, 128, 7, 7]               0
            SiLU-663            [-1, 128, 7, 7]               0
            SiLU-664            [-1, 128, 7, 7]               0
            SiLU-665            [-1, 128, 7, 7]               0
            SiLU-666            [-1, 128, 7, 7]               0
            SiLU-667            [-1, 128, 7, 7]               0
            SiLU-668            [-1, 128, 7, 7]               0
            SiLU-669            [-1, 128, 7, 7]               0
            SiLU-670            [-1, 128, 7, 7]               0
            SiLU-671            [-1, 128, 7, 7]               0
            SiLU-672            [-1, 128, 7, 7]               0
            SiLU-673            [-1, 128, 7, 7]               0
            SiLU-674            [-1, 128, 7, 7]               0
            SiLU-675            [-1, 128, 7, 7]               0
            SiLU-676            [-1, 128, 7, 7]               0
            SiLU-677            [-1, 128, 7, 7]               0
            SiLU-678            [-1, 128, 7, 7]               0
            SiLU-679            [-1, 128, 7, 7]               0
            Conv-680            [-1, 128, 7, 7]               0
      Bottleneck-681            [-1, 128, 7, 7]               0
          Conv2d-682            [-1, 256, 7, 7]          98,560
            SiLU-683            [-1, 256, 7, 7]               0
            SiLU-684            [-1, 256, 7, 7]               0
            SiLU-685            [-1, 256, 7, 7]               0
            SiLU-686            [-1, 256, 7, 7]               0
            SiLU-687            [-1, 256, 7, 7]               0
            SiLU-688            [-1, 256, 7, 7]               0
            SiLU-689            [-1, 256, 7, 7]               0
            SiLU-690            [-1, 256, 7, 7]               0
            SiLU-691            [-1, 256, 7, 7]               0
            SiLU-692            [-1, 256, 7, 7]               0
            SiLU-693            [-1, 256, 7, 7]               0
            SiLU-694            [-1, 256, 7, 7]               0
            SiLU-695            [-1, 256, 7, 7]               0
            SiLU-696            [-1, 256, 7, 7]               0
            SiLU-697            [-1, 256, 7, 7]               0
            SiLU-698            [-1, 256, 7, 7]               0
            SiLU-699            [-1, 256, 7, 7]               0
            SiLU-700            [-1, 256, 7, 7]               0
            SiLU-701            [-1, 256, 7, 7]               0
            SiLU-702            [-1, 256, 7, 7]               0
            SiLU-703            [-1, 256, 7, 7]               0
            SiLU-704            [-1, 256, 7, 7]               0
            SiLU-705            [-1, 256, 7, 7]               0
            SiLU-706            [-1, 256, 7, 7]               0
            SiLU-707            [-1, 256, 7, 7]               0
            SiLU-708            [-1, 256, 7, 7]               0
            Conv-709            [-1, 256, 7, 7]               0
             C2f-710            [-1, 256, 7, 7]               0
          Conv2d-711           [-1, 1280, 7, 7]         328,960
            SiLU-712           [-1, 1280, 7, 7]               0
            SiLU-713           [-1, 1280, 7, 7]               0
            SiLU-714           [-1, 1280, 7, 7]               0
            SiLU-715           [-1, 1280, 7, 7]               0
            SiLU-716           [-1, 1280, 7, 7]               0
            SiLU-717           [-1, 1280, 7, 7]               0
            SiLU-718           [-1, 1280, 7, 7]               0
            SiLU-719           [-1, 1280, 7, 7]               0
            SiLU-720           [-1, 1280, 7, 7]               0
            SiLU-721           [-1, 1280, 7, 7]               0
            SiLU-722           [-1, 1280, 7, 7]               0
            SiLU-723           [-1, 1280, 7, 7]               0
            SiLU-724           [-1, 1280, 7, 7]               0
            SiLU-725           [-1, 1280, 7, 7]               0
            SiLU-726           [-1, 1280, 7, 7]               0
            SiLU-727           [-1, 1280, 7, 7]               0
            SiLU-728           [-1, 1280, 7, 7]               0
            SiLU-729           [-1, 1280, 7, 7]               0
            SiLU-730           [-1, 1280, 7, 7]               0
            SiLU-731           [-1, 1280, 7, 7]               0
            SiLU-732           [-1, 1280, 7, 7]               0
            SiLU-733           [-1, 1280, 7, 7]               0
            SiLU-734           [-1, 1280, 7, 7]               0
            SiLU-735           [-1, 1280, 7, 7]               0
            SiLU-736           [-1, 1280, 7, 7]               0
            SiLU-737           [-1, 1280, 7, 7]               0
            Conv-738           [-1, 1280, 7, 7]               0
AdaptiveAvgPool2d-739           [-1, 1280, 1, 1]               0
         Dropout-740                 [-1, 1280]               0
          Linear-741                 [-1, 1000]       1,281,000
        Classify-742                 [-1, 1000]               0

Total params: 2,715,880

</details>

--- 


## resnet18

<details><summary> ENGINE fp32 SUMM </summary> 

| Layer (type) | Output Shape |
| ---------------|-----------------|
| Reformat - 1 | 1,3,224,224 |
| CaskConvolution - 2 | 1,64,112,112 |
| CaskPooling - 3 | 1,64,56,56 |
| Reformat - 4 | 1,64,56,56 |
| CaskConvolution - 5 | 1,64,56,56 |
| CaskConvolution - 6 | 1,64,56,56 |
| CaskConvolution - 7 | 1,64,56,56 |
| CaskConvolution - 8 | 1,64,56,56 |
| CaskConvolution - 9 | 1,128,28,28 |
| CaskConvolution - 10 | 1,128,28,28 |
| CaskConvolution - 11 | 1,128,28,28 |
| CaskConvolution - 12 | 1,128,28,28 |
| CaskConvolution - 13 | 1,128,28,28 |
| CaskConvolution - 14 | 1,256,14,14 |
| CaskConvolution - 15 | 1,256,14,14 |
| CaskConvolution - 16 | 1,256,14,14 |
| CaskConvolution - 17 | 1,256,14,14 |
| CaskConvolution - 18 | 1,256,14,14 |
| CaskConvolution - 19 | 1,512,7,7 |
| CaskConvolution - 20 | 1,512,7,7 |
| CaskConvolution - 21 | 1,512,7,7 |
| CaskConvolution - 22 | 1,512,7,7 |
| CaskConvolution - 23 | 1,512,7,7 |
| CaskPooling - 24 | 1,512,1,1 |
| CaskGemmConvolution - 25 | 1,1000,1,1 |
| NoOp - 26 | 1,1000 |

</details>

<details><summary> ONNX SUMM  </summary> 

| Layer (type) | Output Shape |
| ---------------|-----------------|
| LayerType.CONVOLUTION - 1 | 1,64,112,112 |
| ActivationType.RELU - 2 | 1,64,112,112 |
| PoolingType.MAX - 3 | 1,64,56,56 |
| LayerType.CONVOLUTION - 4 | 1,64,56,56 |
| ActivationType.RELU - 5 | 1,64,56,56 |
| LayerType.CONVOLUTION - 6 | 1,64,56,56 |
| LayerType.ELEMENTWISE - 7 | 1,64,56,56 |
| ActivationType.RELU - 8 | 1,64,56,56 |
| LayerType.CONVOLUTION - 9 | 1,64,56,56 |
| ActivationType.RELU - 10 | 1,64,56,56 |
| LayerType.CONVOLUTION - 11 | 1,64,56,56 |
| LayerType.ELEMENTWISE - 12 | 1,64,56,56 |
| ActivationType.RELU - 13 | 1,64,56,56 |
| LayerType.CONVOLUTION - 14 | 1,128,28,28 |
| ActivationType.RELU - 15 | 1,128,28,28 |
| LayerType.CONVOLUTION - 16 | 1,128,28,28 |
| LayerType.CONVOLUTION - 17 | 1,128,28,28 |
| LayerType.ELEMENTWISE - 18 | 1,128,28,28 |
| ActivationType.RELU - 19 | 1,128,28,28 |
| LayerType.CONVOLUTION - 20 | 1,128,28,28 |
| ActivationType.RELU - 21 | 1,128,28,28 |
| LayerType.CONVOLUTION - 22 | 1,128,28,28 |
| LayerType.ELEMENTWISE - 23 | 1,128,28,28 |
| ActivationType.RELU - 24 | 1,128,28,28 |
| LayerType.CONVOLUTION - 25 | 1,256,14,14 |
| ActivationType.RELU - 26 | 1,256,14,14 |
| LayerType.CONVOLUTION - 27 | 1,256,14,14 |
| LayerType.CONVOLUTION - 28 | 1,256,14,14 |
| LayerType.ELEMENTWISE - 29 | 1,256,14,14 |
| ActivationType.RELU - 30 | 1,256,14,14 |
| LayerType.CONVOLUTION - 31 | 1,256,14,14 |
| ActivationType.RELU - 32 | 1,256,14,14 |
| LayerType.CONVOLUTION - 33 | 1,256,14,14 |
| LayerType.ELEMENTWISE - 34 | 1,256,14,14 |
| ActivationType.RELU - 35 | 1,256,14,14 |
| LayerType.CONVOLUTION - 36 | 1,512,7,7 |
| ActivationType.RELU - 37 | 1,512,7,7 |
| LayerType.CONVOLUTION - 38 | 1,512,7,7 |
| LayerType.CONVOLUTION - 39 | 1,512,7,7 |
| LayerType.ELEMENTWISE - 40 | 1,512,7,7 |
| ActivationType.RELU - 41 | 1,512,7,7 |
| LayerType.CONVOLUTION - 42 | 1,512,7,7 |
| ActivationType.RELU - 43 | 1,512,7,7 |
| LayerType.CONVOLUTION - 44 | 1,512,7,7 |
| LayerType.ELEMENTWISE - 45 | 1,512,7,7 |
| ActivationType.RELU - 46 | 1,512,7,7 |
| LayerType.REDUCE - 47 | 1,512,1,1 |
| LayerType.SHUFFLE - 48 | 1,512 |
| LayerType.CONSTANT - 49 | 1000,512 |
| LayerType.MATRIX_MULTIPLY - 50 | 1,1000 |
| LayerType.CONSTANT - 51 | 1000, |
| LayerType.SHUFFLE - 52 | 1,1000 |
| LayerType.ELEMENTWISE - 53 | 1,1000 |

</details>

<details><summary> PTH  </summary> 

        Layer (type)               Output Shape         Param #
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          36,864
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
       BasicBlock-11           [-1, 64, 56, 56]               0
           Conv2d-12           [-1, 64, 56, 56]          36,864
      BatchNorm2d-13           [-1, 64, 56, 56]             128
             ReLU-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 64, 56, 56]          36,864
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
       BasicBlock-18           [-1, 64, 56, 56]               0
           Conv2d-19          [-1, 128, 28, 28]          73,728
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
           Conv2d-22          [-1, 128, 28, 28]         147,456
      BatchNorm2d-23          [-1, 128, 28, 28]             256
           Conv2d-24          [-1, 128, 28, 28]           8,192
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
       BasicBlock-27          [-1, 128, 28, 28]               0
           Conv2d-28          [-1, 128, 28, 28]         147,456
      BatchNorm2d-29          [-1, 128, 28, 28]             256
             ReLU-30          [-1, 128, 28, 28]               0
           Conv2d-31          [-1, 128, 28, 28]         147,456
      BatchNorm2d-32          [-1, 128, 28, 28]             256
             ReLU-33          [-1, 128, 28, 28]               0
       BasicBlock-34          [-1, 128, 28, 28]               0
           Conv2d-35          [-1, 256, 14, 14]         294,912
      BatchNorm2d-36          [-1, 256, 14, 14]             512
             ReLU-37          [-1, 256, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]         589,824
      BatchNorm2d-39          [-1, 256, 14, 14]             512
           Conv2d-40          [-1, 256, 14, 14]          32,768
      BatchNorm2d-41          [-1, 256, 14, 14]             512
             ReLU-42          [-1, 256, 14, 14]               0
       BasicBlock-43          [-1, 256, 14, 14]               0
           Conv2d-44          [-1, 256, 14, 14]         589,824
      BatchNorm2d-45          [-1, 256, 14, 14]             512
             ReLU-46          [-1, 256, 14, 14]               0
           Conv2d-47          [-1, 256, 14, 14]         589,824
      BatchNorm2d-48          [-1, 256, 14, 14]             512
             ReLU-49          [-1, 256, 14, 14]               0
       BasicBlock-50          [-1, 256, 14, 14]               0
           Conv2d-51            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-52            [-1, 512, 7, 7]           1,024
             ReLU-53            [-1, 512, 7, 7]               0
           Conv2d-54            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-55            [-1, 512, 7, 7]           1,024
           Conv2d-56            [-1, 512, 7, 7]         131,072
      BatchNorm2d-57            [-1, 512, 7, 7]           1,024
             ReLU-58            [-1, 512, 7, 7]               0
       BasicBlock-59            [-1, 512, 7, 7]               0
           Conv2d-60            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-61            [-1, 512, 7, 7]           1,024
             ReLU-62            [-1, 512, 7, 7]               0
           Conv2d-63            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-64            [-1, 512, 7, 7]           1,024
             ReLU-65            [-1, 512, 7, 7]               0
       BasicBlock-66            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                 [-1, 1000]         513,000

</details>

--- 

Para demostrar que al construir el engine no seimpre se genera la misma estructura ( cantidad de capas ) Corri el experimento que se ecuentra en `build_experiment.sh`
con el resultado siguiente tras generar 4 engine de resnet152 para un batch size de 32:

1er
number of layers:  191
number of parameters:  60040384
2do
number of layers:  221
number of parameters:  60040384
3ero
number of layers:  197
number of parameters:  60040384
4to
number of layers:  185
number of parameters:  60040384

--- 

Al comparar `nvidia-smi` con `pythorch profiler` obtenemos que:

* la utilizacion de la memoria de nvida-smi se ve afectada x el uso base del sistema, siendo en mi caso de unos 540 mb adicionales.

* En el caso de Vanilla concuerda con eso, mientras que en los de TRT no veo relacion.

* La utilizacion de SM se ve afectada en unos 20 puntos, aunque no puedo determinar la causa, ya que segun nvtop, en idle se tiene 0%

comparaciones: 

| MODEL    | SM % (profiler)  |   Memory mb (profiler)  |SM % (nvidia-smi)  |   Memory mb (nvidia-smi)  |
|----------|------------------|-------------------------|-------------------|---------------------------|
| Vanilla  | none             | 1138                    | 80                | 1791                      |
| TRT fp32 | 50               | 40                      | 60                | 1201                      |
| TRT fp16 | 27               | 40                      | 44                | 931                       |
| TRT int8 | 14               | 40                      | 35                | 760                       |

obs: valores bien aprox

Notas: 

se genero el shell script `gen_tables.sh` que corre diferentes instanciaciones de `main_all.sh` que a su vez corre instancias de `build_all.sh`, el cual genera los `.engine` asi como el `main.py` para generar la tabla.

Meti las tablas generadas en `outputs/table_outputs/` para dejar más limpio todo.

Hay notas utiles de como resolver problemas q tuve con las instalaciones en el [proyecto](https://github.com/users/Juanx65/projects/2) del repositorio, q voy anotando a diario.

---

# Avances 10 nov

Siguiendo la [documentacion](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector), escribi el script en  `param_counter.py` que suma la totalidad de pesos por capa para entregar el total de parametros de el engine.

luego, comprobando que la cantidad de parametros de pytorch y onnx son las mismas usando [la herramienta onnx-opcounter](https://github.com/gmalivenko/onnx-opcounter); por ejemplo, la cantidad de parametros de [yolov8n-cls](https://docs.ultralytics.com/models/yolov8/#supported-modes) es de 2.7M, mientras que [ onnx-opcounter](https://github.com/gmalivenko/onnx-opcounter) da como resultado 2715880.

Finalmente, usando `param_counter.py` tenemos el resultado de 2711472 parametros para el yolov8n.engine.
 
--- 

Notas: 

Segun la informacion disponible en enlaces como [este](https://stackoverflow.com/questions/70097798/number-of-parameters-and-flops-in-onnx-and-tensorrt-model#:~:text=1,onnx%20model%20here), al optimizar con tesnorRT no cambia el numero de parametros ni de FLOPS amenos de que existan ramas redindantes en la red [almenos para los flops segun esto](https://forums.developer.nvidia.com/t/number-of-operations-in-a-tensorrt-model/77687).

De todas formas [aca](https://stackoverflow.com/questions/70097798/number-of-parameters-and-flops-in-onnx-and-tensorrt-model#:~:text=1,onnx%20model%20here) solo se demuestra que la cantidad de parametros es igual de pytorch a onnx usando esta [herramienta: onnx-opcounter](https://github.com/gmalivenko/onnx-opcounter), la cual solo permite extraer la cantidad de parametros de redes en el formato onnx, no asi de un engine.

Se demostro entonces que la cantidad de parametros de pytorch a onnx a tensorrt son iguales. se suguiere usar texec, herramienta que debido a que incluye el builder para construir el engine no planeo usar.

# Avances 1 nov

det pose con yolov8m-pose
Speed: 1.9ms preprocess, 10.4ms inference, 1.2ms postprocess per image at shape (1, 3, 640, 384)

det pose con fp32
Speed: 2.2ms preprocess, 12.3ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 640)

det pose con fp16
Speed: 2.1ms preprocess, 4.2ms inference, 1.6ms postprocess per image at shape (1, 3, 640, 640)

# Avances 26/oct

* Comparación de tiempos y precisiones entre "sync" y "no sync" (tablas).

    En donde:
    - No se encuentran cambios en la precisión de los modelos.
    - El tiempo de los modelos TRT no cambia significativamente.
    - La latencia del modelo Vanilla ahora sigue la tendencia esperada (ser mayor que la de TRT).

    Cabe notar que:
    - El cambio en el tamaño se debe a que escribí una función que revisara automáticamente el tamaño de los archivos, los cuales antes anotaba revisando según lo visto en "Folders". No lo actualicé en las tablas no-sync, pero deberían ser iguales.
    - Los tiempos a veces varían más, dependiendo de cada ejecución de la validación.

<details><summary>  ResNet152 </summary> 

### Batch Size 1

|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|
|-----------------|----------------|------------------|-----------|----------------------|---------------------|
| Vanilla         |    8.1 / 16.3  |  6.0 / 15.9      |241.7      |82.34                 |95.92                |
| TRT fp32-static |    5.5 / 10.0  |  5.1 / 9.5       |243.3      |82.34                 |95.92                |
| TRT fp16-static |    2.2 / 8.6   |  1.8 / 8.1       |122.6      |82.32                 |95.90                |
| TRT int8-static |    1.5 / 4.3   |  1.2 / 3.9       |65.5       |79.99                 |95.74                |


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

* Programe un script que permita evaluar tanto los pesos preentrenados Vanilla para la segmentación de salmones, como el engine optimizado TRT en fp16 y fp32.

    Cabe notar que:
    - Hasta el momento, no ha sido posible generar los engines optimizados utilizando mi propio flujo de trabajo. En su lugar, optimicé la red usando las herramientas disponibles en Ultralytics. Estas herramientas permiten usar fp32, fp16, batch size estático y dinámico. Sin embargo, el uso del batch size dinámico es más limitado, ya que no me permite generar en fp16. Aún estoy revisando sus parámetros (flags).

## batch size 1 (map sobre mask)

|  Model      |Latency (ms)| size (MB) | mAP50 (%) |mAP50-95 (%) |
|-------------|------------|-----------|-----------|-------------|
| Vanilla     | 23.4       | 54.8      | 51.7      | 25.2        |
| TRT fp32    | 14.6       | 157.2     | 50.7      | 25.1        |
| TRT fp16    | 4.9        | 58.6      | 51.2      | 25.2        |
