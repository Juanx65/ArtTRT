import onnx
import numpy as np
from onnx import numpy_helper

def modificar_pesos(modelo, nombres_capas, nuevo_valor):
    inicializadores = {init.name: init for init in modelo.graph.initializer}

    #Crear un diccionario para los nombres de los nodos
    nodos = {node.name: node for node in modelo.graph.node}

    for nombre_capa in nombres_capas:
        if nombre_capa in nodos:
            nodo = nodos[nombre_capa]
            # Obtener los nombres de los tensores de entrada del nodo
            nombres_pesos = nodo.input
            
            for nombre_peso in nombres_pesos:
                if nombre_peso in inicializadores:
                    peso = inicializadores[nombre_peso]
                    array_peso = numpy_helper.to_array(peso)

                    # Modificar los pesos a cero
                    #array_peso_modificado = np.zeros_like(array_peso)
                    array_peso_modificable = np.copy(array_peso)
                    # Modificar el primer peso a cero
                    array_peso_modificable.flat[0] = 0.0

                    # Convertir el array modificado de vuelta a un tensor ONNX
                    peso_modificado = numpy_helper.from_array(array_peso_modificable, name=nombre_peso)

                    # Reemplazar el inicializador original por el modificado en el grafo del modelo
                    for i, init in enumerate(modelo.graph.initializer):
                        if init.name == nombre_peso:
                            modelo.graph.initializer[i].CopyFrom(peso_modificado)
                            break
                    print(f'Pesos modificados para el nodo: {nombre_capa} (inicializador: {nombre_peso})')
                else:
                    print(f'Inicializador no encontrado para el tensor: {nombre_peso}')
        else:
            print(f'Nodo no encontrado para la capa: {nombre_capa}')

# Cargar el modelo ONNX
model_path = '../weights/best.onnx'
model = onnx.load(model_path)

# Nombres de las capas cuyas pesos quieres modificar
nombres_capas = ['/layer1/layer1.0/conv1/Conv','/layer1/layer1.0/conv2/Conv','/layer1/layer1.0/conv3/Conv']
modificar_pesos(model, nombres_capas, nuevo_valor=0.0)

# Guardar el modelo modificado
onnx.save(model, '../weights/best_new.onnx')

# Verificar el modelo
onnx.checker.check_model(model)
