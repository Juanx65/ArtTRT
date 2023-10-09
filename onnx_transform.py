import torch
import onnx
import os
from io import BytesIO

import argparse

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def main(opt):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = opt.weights
    weights_path = os.path.join(current_directory,weights_path)

    if opt.pretrained:
        if opt.network == "mobilenet":
            model = torch.hub.load('pytorch/vision:v0.15.2', "mobilenet_v2", weights=f'MobileNet_V2_Weights.DEFAULT')
        elif "resnet" in opt.network:
            model = torch.hub.load('pytorch/vision:v0.15.2', opt.network, weights=f'ResNet{opt.network[6:]}_Weights.DEFAULT')
        elif "yolo" in opt.network:
            from ultralytics import YOLO
            YOLOv8 = YOLO("weights/yolov8x-cls.pt")
            model = YOLOv8.model.fuse()
        else:
            print("Red no reconocida.")
    else:
        model = torch.load(weights_path)

    model.to(device)
    model.eval()
    
    fake_input = torch.randn([opt.batch_size,3, 224, 224]).to(device)
    dynamic_axes = {'images': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}}  # Indica ejes dinámicos

    for _ in range(2):
        model(fake_input)
    save_path = weights_path.replace('.pth', '.onnx')

    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=11,
            input_names=['images'],
            output_names=['outputs'],
            dynamic_axes=dynamic_axes)  # Añade los ejes dinámicos
        f.seek(0)
        onnx_model = onnx.load(f)

    # Guardar el modelo ONNX en un archivo .onnx
    onnx.save(onnx_model, save_path)

    print("La conversión a ONNX se ha completado exitosamente. El modelo se ha guardado en:", save_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size to train')
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='path to the pth weight file')
    parser.add_argument('-p','--pretrained', action='store_true',help='transform a pretrained model from torch.hub.load')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)