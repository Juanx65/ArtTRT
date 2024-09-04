# Jetson Orin Nano Power Mode Test

Power mode 0: 15w

Power mode 1: 7w

## resnet152 bs 1 build on mode 0 run on mode 0

| Test Type | Total Eval Time (ms) | Avg Eval Time (ms) | Max Eval Time (ms) |
|-----------|----------------------|--------------------|--------------------|
| Vanilla   | 55645.90             | 4637.16            | 5023.08            |
| FP32      | 1558.01              | 129.83             | 204.90             |
| FP16      | 3273.47              | 272.79             | 319.65             |
| INT8      | 958.08               | 79.84              | 121.92             |


## resnet152 bs 1 build on mode 1 run on mode 1

| Test Type | Total Eval Time (ms) | Avg Eval Time (ms) | Max Eval Time (ms) |
|-----------|----------------------|--------------------|--------------------|
| Vanilla   | 45696.73             | 3808.06            | 4206.33            |
| FP32      | 1703.01              | 141.92             | 219.21             |
| FP16      | 3348.36              | 279.03             | 329.02             |
| INT8      | 965.52               | 80.46              | 123.61             |

