import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "/home/hiwonder/pug/src/pug_alif/shoe_detection/shoe_yolov8n_single_fp16.engine"

def report(stage):
    with open("/proc/self/status") as f:
        lines = f.readlines()
    keys = ("VmRSS:", "RssAnon:", "RssFile:")
    print("\n" + stage, flush=True)
    for line in lines:
        if line.startswith(keys):
            print(line.strip(), flush=True)

report("After imports and CUDA initialization")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine_data = f.read()

report("After reading engine file")

engine = runtime.deserialize_cuda_engine(engine_data)
if engine is None:
    raise RuntimeError("TensorRT engine deserialization failed")

report("After engine deserialization")

context = engine.create_execution_context()
if context is None:
    raise RuntimeError("TensorRT execution context creation failed")

report("After execution context creation")

input_bytes = 1 * 3 * 416 * 416 * 4
output_bytes = 1 * 5 * 3549 * 4

host_input = cuda.pagelocked_empty(input_bytes // 4, dtype="float32")
host_output = cuda.pagelocked_empty(output_bytes // 4, dtype="float32")
device_input = cuda.mem_alloc(input_bytes)
device_output = cuda.mem_alloc(output_bytes)

report("After input/output buffer allocation")
print("\nDiagnostic complete. No camera or inference was started.")
