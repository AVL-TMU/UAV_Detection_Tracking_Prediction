import time
import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
from PIL import Image
import rfdetr.datasets.transforms as T

# Default configuration
DETECTION_THRESHOLD = 0.5
CUSTOM_RESOLUTION = 560  # must match TRT engine

# Transforms for detection
det_transforms = T.Compose([
    T.SquareResize([CUSTOM_RESOLUTION]),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

class Detector:
    def __init__(self, engine_path):
        """Initialize the detector with a TensorRT engine"""
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate TRT buffers
        self.input_tensors, self.output_tensors = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            count = int(np.prod(shape))
            host = cuda.pagelocked_empty(count, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            if mode == trt.TensorIOMode.INPUT:
                self.input_tensors.append((host, dev))
            else:
                self.output_tensors.append((name, host, dev, shape))
    
    def load_engine(self, path):
        """Load TensorRT engine from file"""
        with open(path, 'rb') as f:
            return trt.Runtime(self.TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    def detect(self, frame, threshold=None):
        """Run detection on a frame"""
        if threshold is None:
            threshold = DETECTION_THRESHOLD
            
        h0, w0 = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CUSTOM_RESOLUTION, CUSTOM_RESOLUTION))
        pil = Image.fromarray(img)
        tensor, _ = det_transforms(pil, None)
        arr = tensor.unsqueeze(0).numpy().astype(np.float32).ravel()

        host, dev = self.input_tensors[0]
        np.copyto(host, arr)
        cuda.memcpy_htod(dev, host)

        t0 = time.time()
        bindings = [int(dev)] + [int(d) for _,_,d,_ in self.output_tensors]
        self.context.execute_v2(bindings)
        t1 = time.time()

        outs = {}
        for name, host_mem, dev_mem, shape in self.output_tensors:
            cuda.memcpy_dtoh(host_mem, dev_mem)
            outs[name] = host_mem.reshape(shape)

        dets = torch.from_numpy(outs['dets'][0]).float()
        logits = torch.from_numpy(outs['labels'][0]).float()
        scores = torch.sigmoid(logits)
        maxs, _ = scores.max(dim=1)
        sel = torch.nonzero(maxs >= threshold, as_tuple=False).squeeze(1)

        if sel.numel() == 0:
            return np.zeros((0,5), dtype=float), (t1 - t0)

        results = []
        for i in sel.tolist():
            cx, cy, w_norm, h_norm = dets[i].tolist()
            score = float(maxs[i])
            x1 = (cx - w_norm/2) * w0
            y1 = (cy - h_norm/2) * h0
            x2 = (cx + w_norm/2) * w0
            y2 = (cy + h_norm/2) * h0
            results.append([x1, y1, x2, y2, score])

        return np.array(results, dtype=float).reshape(-1,5), (t1 - t0) 