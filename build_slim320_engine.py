import tensorrt as trt

ONNX_PATH = "version-slim-320.onnx"
ENGINE_PATH = "version-slim-320_fp16.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)

flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, "rb") as model:
    if not parser.parse(model.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
config.set_flag(trt.BuilderFlag.FP16)

print("Building FP16 TensorRT engine...")

engine = builder.build_engine(network, config)

if engine is None:
    raise RuntimeError("TensorRT engine build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())

print("Engine saved:", ENGINE_PATH)
