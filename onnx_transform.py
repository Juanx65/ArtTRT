import torch
import onnx
import os
from io import BytesIO

BATCH_SIZE = 256

current_directory = os.path.dirname(os.path.abspath(__file__))
weights_path = 'weights/best.pth'
weights_path = os.path.join(current_directory,weights_path)

print("weights path: ", weights_path)

model = torch.load(weights_path)
model.to('cuda:0')
model.eval()
fake_input = torch.randn([BATCH_SIZE,3, 224, 224]).to('cuda:0')
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
        output_names=['outputs'])
    f.seek(0)
    onnx_model = onnx.load(f)

# Guardar el modelo ONNX en un archivo .onnx
onnx.save(onnx_model, save_path)

print("La conversión a ONNX se ha completado exitosamente. El modelo se ha guardado en:", save_path)