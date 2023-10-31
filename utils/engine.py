import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import glob
import logging
import sys
import random

try:
    from processing import preprocess_imagenet as preprocessing
except ImportError:
    try:
        from utils.processing import preprocess_imagenet as preprocessing
    except ImportError:
        print("No se pudo importar el módulo de procesamiento.")

import onnx
import tensorrt as trt
import torch

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
#logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1
CHANNEL = 3
HEIGHT = 224
WIDTH = 224

CACHE_FOLDER = "outputs/cache/"

class EngineBuilder:
    seg = False

    def __init__(
            self,
            checkpoint: Union[str, Path],
            device: Optional[Union[str, int, torch.device]] = None) -> None:
        
        checkpoint = Path(checkpoint) if isinstance(checkpoint,str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix in ('.onnx')

        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device
    def __build_engine(self,
                       fp32: bool = True,
                       fp16: bool = False,
                       int8: bool = False,
                       input_shape: Union[List, Tuple] = (1,3,128, 32),
                       with_profiling: bool = True) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(self.device).total_memory

        if(input_shape[0] == -1): # solo si se hara un engine para batch size dinamico
            # para trabajar con batch size dinamico
            profile = builder.create_optimization_profile()

            # dimensions for dynamic input "images" defined in the onnx_transform script
            min_in_dims = trt.Dims4(1,input_shape[1],input_shape[2],input_shape[3])
            max_in_dims = trt.Dims4(256,input_shape[1],input_shape[2],input_shape[3])

            profile.set_shape("images", min_in_dims, max_in_dims, max_in_dims)
            config.add_optimization_profile(profile) # Agrega el perfil de optimización a la configuración
            #continua el codigo como antes

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)

        self.logger = logger
        self.builder = builder
        self.network = network

        self.build_from_onnx()

        if ~fp32:
            if fp16 and self.builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8 and self.builder.platform_has_fast_int8:
                ## Carga de los datos
                calibration_file = get_calibration_files(calibration_data="datasets/img_preprocess/")
                Int8_calibrator = ImagenetCalibrator(calibration_files=calibration_file,
                                                     batch_size=input_shape[0],
                                                     input_shape=(input_shape[1],input_shape[2],input_shape[3]),
                                                     preprocess_func=preprocessing)
   
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = Int8_calibrator
    
        self.weight = self.checkpoint.with_suffix('.engine')

        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        with self.builder.build_engine(self.network, config) as engine:
            self.weight.write_bytes(engine.serialize())
        self.logger.log(
            trt.Logger.WARNING, f'Build tensorrt engine finish.\n'
            f'Save in {str(self.weight.absolute())}')

    def build(self,
              fp32: bool = True,
              fp16: bool = False,
              int8: bool = False,
              input_shape: Union[List, Tuple] = (1, 3, 128, 32),
              with_profiling=True) -> None:
        self.__build_engine(fp32, fp16, int8, input_shape, with_profiling)

    def build_from_onnx(self):
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.checkpoint))

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(
                f'failed to load ONNX file: {str(self.checkpoint)}')
        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        for inp in inputs:
            self.logger.log(
                trt.Logger.WARNING,
                f'input "{inp.name}" with shape: {inp.shape} '
                f'dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(
                trt.Logger.WARNING,
                f'output "{out.name}" with shape: {out.shape} '
                f'dtype: {out.dtype}')

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()
        num_bindings = model.num_bindings
        names = [model.get_binding_name(i) for i in range(num_bindings)]

        self.bindings: List[int] = [0] * num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]

        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.idynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))

        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.odynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape,
                                     dtype=self.out_info[i].dtype,
                                     device=self.device)
            else:
                output = self.output_tensor[i]
            self.bindings[j] = output.data_ptr()
            outputs.append(output)

        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs[i]
                     for i in self.idx) if len(outputs) > 1 else outputs[0]

class TRTProfilerV1(trt.IProfiler):

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.total_runtime = 0.0
        self.recorder = defaultdict(float)

    def report_layer_time(self, layer_name: str, ms: float):
        self.total_runtime += ms * 1000
        self.recorder[layer_name] += ms * 1000

    def report(self):
        f = '\t%40s\t\t\t\t%10.4f'
        print('\t%40s\t\t\t\t%10s' % ('layername', 'cost(us)'))
        for name, cost in sorted(self.recorder.items(), key=lambda x: -x[1]):
            print(
                f %
                (name if len(name) < 40 else name[:35] + ' ' + '*' * 4, cost))
        print(f'\nTotal Inference Time: {self.total_runtime:.4f}(us)')

def get_calibration_files(calibration_data, max_calibration_size=None, allowed_extensions=("JPEG",".jpeg", ".jpg", ".png",".tiff")):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.
    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    max_calibration_size: int
        Max number of files to use for calibration. If calibration_data contains more than this number,
        a random sample of size max_calibration_size will be returned instead. If None, all samples will be used.
    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """

    logger.info("Collecting calibration files from: {:}".format(calibration_data))
    calibration_files = [path for path in glob.iglob(os.path.join(calibration_data, "**"), recursive=True)
                         if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
    logger.info("Number of Calibration Files found: {:}".format(len(calibration_files)))

    if len(calibration_files) == 0:
        raise Exception("ERROR: Calibration data path [{:}] contains no files!".format(calibration_data))

    if max_calibration_size:
        if len(calibration_files) > max_calibration_size:
            logger.warning("Capping number of calibration images to max_calibration_size: {:}".format(max_calibration_size))
            random.seed(42)  # Set seed for reproducibility
            calibration_files = random.sample(calibration_files, max_calibration_size)

    return calibration_files

class ImagenetCalibrator(trt.IInt8EntropyCalibrator2):
    """
        https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html

        INT8 Calibrator Class for Imagenet-based Image Classification Models.
        Parameters
        ----------
        calibration_files: List[str]
            List of image filenames to use for INT8 Calibration
        batch_size: int
            Number of images to pass through in one batch during calibration
        input_shape: Tuple[int]
            Tuple of integers defining the shape of input to the model (Default: (3, 224, 224))
        cache_file: str
            Name of file to read/write calibration cache from/to.
        preprocess_func: function -> numpy.ndarray
            Pre-processing function to run on calibration data. This should match the pre-processing
            done at inference time. In general, this function should return a numpy array of
            shape `input_shape`.
    """
    def __init__(self, calibration_files=[], batch_size=BATCH_SIZE, 
                 input_shape=(CHANNEL, HEIGHT, WIDTH),
                 cache_file=CACHE_FOLDER+"calibration.cache", preprocess_func=None):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        
        # Inicializa un tensor en la GPU usando torch
        self.device_input = torch.zeros(self.batch_size, *self.input_shape, dtype=torch.float32, device='cuda')

        self.files = calibration_files
        if len(self.files) % self.batch_size != 0:
            logger.info("Padding # calibration files to be a multiple of batch_size {:}".format(self.batch_size))
            self.files += calibration_files[(len(calibration_files) % self.batch_size):self.batch_size]

        self.batches = self.load_batches()

        if preprocess_func is None:
            logger.error("No preprocess_func defined! Please provide one to the constructor.")
            sys.exit(1)
        else:
            self.preprocess_func = preprocess_func
        
        # Verificar si el archivo calibration.cache existe, si no, créalo
        if not os.path.exists(self.cache_file):
            # Crea el directorio si no existe
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                pass
            logger.info(f"'{self.cache_file}' has been created!")

    def load_batches(self):
        for index in range(0, len(self.files), self.batch_size):
            for offset in range(self.batch_size):
                image = Image.open(self.files[index + offset])
                self.batch[offset] = self.preprocess_func(image, *self.input_shape)
            logger.info("Calibration images pre-processed: {:}/{:}".format(index+self.batch_size, len(self.files)))
            yield self.batch

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.batches)
            # Copia los datos desde la CPU (numpy) a la GPU (torch)
            self.device_input.copy_(torch.tensor(batch, device='cuda'))
            return [int(self.device_input.data_ptr())]  # Devuelve el puntero en la memoria de GPU
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
