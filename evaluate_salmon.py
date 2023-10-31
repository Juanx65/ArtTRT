from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
import os
import argparse
import torch
from PIL import Image

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def main(args:argparse.Namespace) -> None:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_directory,args.imgs)
    
    model = YOLO(args.weights, task='segment')
    #model =  AutoBackend( args.weights, device=device)#, dnn=False, fp16=True).to(device)

    model.val(data=data_path, verbose=True, device=device, imgsz=640, batch=1) # evaluar
    
    #results = model.eval()#"dataset_salmon/images/val/Img1.jpeg")
    #print("results: ", results)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Engine file',default='weights/best_salmon.pt')
    parser.add_argument('--imgs', type=str, help='Images file', default='datasets/dataset_salmon/data.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)