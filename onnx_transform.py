import torch
import onnx
import os
from io import BytesIO

from utils.models.CustomNets import JuanjoNet #juanjo experimento

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
            #from ultralytics import YOLO
            #YOLOv8 = YOLO(opt.weights)
            from ultralytics.nn.autobackend import AutoBackend
            YOLOv8 = AutoBackend(opt.weights, device=device) #, dnn=False, fp16=False)
            model = YOLOv8.model.fuse()
        else:
            print("Red no reconocida.")
    else:
        #model = ReLUNet() #para probar el experimento nms
        model = torch.load(weights_path)

    model.to(device)
    model.eval()
    
    # Se inicializa dynamic_axes como None por defecto
    dynamic_axes = None
    # Solo define dynamic_axes si opt.batch_size es -1 / es decir, si es batch size dinamico
    if opt.input_shape[0] == -1:
        fake_input = torch.randn( [1,opt.input_shape[1],opt.input_shape[2],opt.input_shape[3]]).to(device)
        dynamic_axes = {'images': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}}
    else:
        fake_input = torch.randn(opt.input_shape).to(device)

    for _ in range(2):
        model(fake_input)
    if opt.network == "yolo":
        save_path = weights_path.replace('.pt', '.onnx')
    else:
        save_path = weights_path.replace('.pth', '.onnx')

    with BytesIO() as f:
        # Añade dynamic_axes solo si está definido
        export_args = {
            "model": model,
            "args": fake_input,
            "f": f,
            "opset_version": 11,
            "input_names": ['images'],
            "output_names": ['outputs']#['output0',"a","b","c","d",'output1']
        }
        if dynamic_axes:
            export_args["dynamic_axes"] = dynamic_axes
    
        torch.onnx.export(**export_args)
        f.seek(0)
        onnx_model = onnx.load(f)

    # Guardar el modelo ONNX en un archivo .onnx
    onnx.save(onnx_model, save_path)

    print("La conversión a ONNX se ha completado exitosamente. El modelo se ha guardado en:", save_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='path to the pth weight file')
    parser.add_argument('-p','--pretrained', action='store_true',help='transform a pretrained model from torch.hub.load')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    parser.add_argument('--input_shape',
                        nargs='+',
                        type=int,
                        default=[-1,3, 224,224],
                        help='Model input shape, el primer valor es el batch_size, -1 (dinamico))]')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)