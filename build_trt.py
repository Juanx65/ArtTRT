import argparse
from utils.engine import EngineBuilder

BATCH_SIZE = 256

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='weights/best.onnx',
                        help='Weights file')
    parser.add_argument('--input_shape',
                        nargs='+',
                        type=int,
                        default=[BATCH_SIZE,3, 224,224],
                        help='Model input shape, el primer valor es el batch_size, 128)]')
    parser.add_argument('--fp32',
                        action='store_true',
                        help='Build model with fp32 mode')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Build model with fp16 mode')
    parser.add_argument('--int8',
                        action='store_true',
                        help='Build model with int8 mode')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT builder device')
    parser.add_argument('--seg',
                        action='store_true',
                        help='Build seg model by onnx')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    builder = EngineBuilder(args.weights, args.device)
    builder.seg = args.seg
    builder.build(fp32=args.fp32, fp16=args.fp16, int8=args.int8, input_shape=args.input_shape)

if __name__ == '__main__':
    args = parse_args()
    main(args)