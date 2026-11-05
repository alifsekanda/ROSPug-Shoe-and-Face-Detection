#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time

from sensor_msgs.msg import Image


ENGINE_PATH = "/home/hiwonder/pug/src/pug_alif/shoe_detection/version-slim-320_fp16.engine"

CAMERA_TOPIC = "/csi_camera/image_rect_color"
OUTPUT_TOPIC = "/slim320_face_detector/image"

INPUT_WIDTH = 320
INPUT_HEIGHT = 240

CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.30
INFERENCE_INTERVAL = 0.5

CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2

MIN_BOXES = [
    [10, 16, 24],
    [32, 48],
    [64, 96],
    [128, 192, 256]
]

STRIDES = [8, 16, 32, 64]


def generate_priors():
    priors = []

    feature_maps = [
        [30, 40],
        [15, 20],
        [8, 10],
        [4, 5]
    ]

    for k, feature_map in enumerate(feature_maps):
        for i in range(feature_map[0]):
            for j in range(feature_map[1]):
                for min_box in MIN_BOXES[k]:
                    cx = (j + 0.5) * STRIDES[k] / INPUT_WIDTH
                    cy = (i + 0.5) * STRIDES[k] / INPUT_HEIGHT
                    w = min_box / INPUT_WIDTH
                    h = min_box / INPUT_HEIGHT

                    priors.append([cx, cy, w, h])

    priors = np.array(priors, dtype=np.float32)

    assert len(priors) == 4420, "Incorrect prior count"

    return priors


PRIORS = generate_priors()


class TensorRTFaceDetector:

    def __init__(self):
        rospy.init_node("slim320_tensorrt_test")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(ENGINE_PATH, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()

        self.bindings = [0] * self.engine.num_bindings
        self.host_buffers = {}
        self.device_buffers = {}
        self.binding_info = {}

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)

            self.host_buffers[name] = host
            self.device_buffers[name] = device

            self.binding_info[name] = {
                "index": i,
                "shape": shape,
                "dtype": dtype
            }

            self.bindings[i] = int(device)

            if self.engine.binding_is_input(i):
                self.input_name = name
            else:
                print("Output:", name, shape)

        if "boxes" not in self.binding_info or "scores" not in self.binding_info:
            raise RuntimeError("Expected boxes and scores outputs not found")

        input_info = self.binding_info[self.input_name]

        print("Input:", self.input_name, input_info["shape"])

        if input_info["shape"] != (1, 3, 240, 320):
            raise RuntimeError(
                "Unexpected engine input shape: {}".format(input_info["shape"])
            )

        dummy = np.zeros(
            (1, 3, 240, 320),
            dtype=input_info["dtype"]
        )

        self.run_engine(dummy)

        self.publisher = rospy.Publisher(
            OUTPUT_TOPIC,
            Image,
            queue_size=1
        )

        self.last_inference = 0
        self.last_report = time.time()

        rospy.loginfo("Slim-320 TensorRT FP16 detector started")

        self.run()

    def run_engine(self, blob):
        input_host = self.host_buffers[self.input_name]

        # 1. Host-to-device copy
        start_h2d = time.time()

        np.copyto(input_host, blob.ravel())

        cuda.memcpy_htod_async(
            self.device_buffers[self.input_name],
            input_host,
            self.stream
        )

        self.stream.synchronize()
        h2d_ms = (time.time() - start_h2d) * 1000

        # 2. TensorRT inference
        start_inference = time.time()

        success = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        self.stream.synchronize()
        inference_ms = (time.time() - start_inference) * 1000

        if not success:
            raise RuntimeError("TensorRT execution failed")

        # 3. Device-to-host copy
        start_d2h = time.time()

        for name in ["boxes", "scores"]:
            cuda.memcpy_dtoh_async(
                self.host_buffers[name],
                self.device_buffers[name],
                self.stream
            )

        self.stream.synchronize()
        d2h_ms = (time.time() - start_d2h) * 1000

        outputs = {}

        for name in ["boxes", "scores"]:
            info = self.binding_info[name]

            outputs[name] = self.host_buffers[name].reshape(
                info["shape"]
            ).copy()

        # Print timing breakdown every 5 seconds
        if not hasattr(self, "last_timing_report"):
            self.last_timing_report = 0

        now = time.time()

        if now - self.last_timing_report >= 5:
            rospy.loginfo(
                "H2D: %.2f ms | TensorRT: %.2f ms | D2H: %.2f ms",
                h2d_ms,
                inference_ms,
                d2h_ms
            )
            self.last_timing_report = now

        return outputs["boxes"], outputs["scores"]

    def infer(self, image):
        resized = cv2.resize(
            image,
            (INPUT_WIDTH, INPUT_HEIGHT)
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(
            rgb,
            scalefactor=1.0 / 128.0,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            mean=(127, 127, 127),
            swapRB=False
        )

        boxes, scores = self.run_engine(blob)

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, 2)

        decoded_boxes = []

        for i in range(len(boxes)):
            prior = PRIORS[i]

            cx = prior[0] + boxes[i][0] * CENTER_VARIANCE * prior[2]
            cy = prior[1] + boxes[i][1] * CENTER_VARIANCE * prior[3]

            w = prior[2] * np.exp(boxes[i][2] * SIZE_VARIANCE)
            h = prior[3] * np.exp(boxes[i][3] * SIZE_VARIANCE)

            x1 = (cx - w / 2) * INPUT_WIDTH
            y1 = (cy - h / 2) * INPUT_HEIGHT
            x2 = (cx + w / 2) * INPUT_WIDTH
            y2 = (cy + h / 2) * INPUT_HEIGHT

            decoded_boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

        decoded_boxes = np.array(
            decoded_boxes,
            dtype=np.float32
        )

        face_scores = scores[:, 1]

        indices = np.where(
            face_scores >= CONFIDENCE_THRESHOLD
        )[0]

        if len(indices) == 0:
            return resized, 0

        selected_boxes = decoded_boxes[indices]
        selected_scores = face_scores[indices]

        nms_indices = cv2.dnn.NMSBoxes(
            selected_boxes.tolist(),
            selected_scores.tolist(),
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        count = 0

        if len(nms_indices) > 0:
            for idx in np.array(nms_indices).flatten():
                x, y, w, h = selected_boxes[idx].astype(int)

                x = max(0, x)
                y = max(0, y)
                w = min(w, INPUT_WIDTH - x)
                h = min(h, INPUT_HEIGHT - y)

                if w <= 0 or h <= 0:
                    continue

                confidence = selected_scores[idx]

                cv2.rectangle(
                    resized,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    resized,
                    "{:.2f}".format(confidence),
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                count += 1

        return resized, count

    def process_image(self, msg):
        image = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(msg.height, msg.width, -1)

        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        elif msg.encoding != "bgr8":
            rospy.logwarn_throttle(
                5,
                "Unsupported encoding: " + msg.encoding
            )
            return

        start = time.time()

        result, faces = self.infer(image)

        inference_ms = (time.time() - start) * 1000

        output = Image()
        output.header = msg.header
        output.height = result.shape[0]
        output.width = result.shape[1]
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = result.tobytes()

        self.publisher.publish(output)

        now = time.time()

        if now - self.last_report >= 5:
            ram = self.get_ram_mb()

            rospy.loginfo(
                "Faces: %d | Inference: %.1f ms | RAM: %.1f MB",
                faces,
                inference_ms,
                ram
            )

            self.last_report = now

    def run(self):
        while not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(
                    CAMERA_TOPIC,
                    Image,
                    timeout=2.0
                )

                now = time.time()

                if now - self.last_inference < INFERENCE_INTERVAL:
                    continue

                self.last_inference = now

                self.process_image(msg)

            except rospy.ROSException:
                continue

            except Exception as e:
                rospy.logerr_throttle(
                    5,
                    "Detection error: %s",
                    str(e)
                )

    @staticmethod
    def get_ram_mb():
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0

        return 0


if __name__ == "__main__":
    cuda.init()

    cuda_context = cuda.Device(0).make_context()

    try:
        TensorRTFaceDetector()

    except rospy.ROSInterruptException:
        pass

    finally:
        cuda_context.pop()
        cuda_context.detach()
