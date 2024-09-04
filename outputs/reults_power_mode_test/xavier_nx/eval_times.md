# Jetson Xavier NX

Power mode 5 (default): 10w 4 cores
Power mode 8: 202 6 cores

## Resnet 152 bs 1 build on mode 8 run on mode 8

| Test Type | Total Eval Time (ms) | Avg Eval Time (ms) | Max Eval Time (ms) |
|-----------|----------------------|--------------------|--------------------|
| Vanilla   | 92296.89             | 7691.41            | 8169.47            |
| FP32      | 8408.14              | 700.68             | 918.29             |
| FP16      | 2211.48              | 184.29             | 266.07             |
| INT8      | 1480.25              | 123.35             | 174.14             |

## Resnet 152 bs 1 build on mode 5 run on mode 5

| Test Type | Total Eval Time (ms) | Avg Eval Time (ms) | Max Eval Time (ms) |
|-----------|----------------------|--------------------|--------------------|
| Vanilla   | 81622.76             | 6801.90            | 7503.56            |
| FP32      | 6285.06              | 523.76             | 940.83             |
| FP16      | 2735.28              | 227.94             | 359.75             |
| INT8      | 1760.78              | 146.73             | 218.22             |

