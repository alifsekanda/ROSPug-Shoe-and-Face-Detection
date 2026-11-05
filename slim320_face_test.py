#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os

from sensor_msgs.msg import Image


MODEL_PATH = "/home/hiwonder/pug/src/pug_alif/shoe_detection/version-slim-320.onnx"
CAMERA_TOPIC = "/csi_camera/image_rect_color"
OUTPUT_TOPIC = "/slim320_face_detector/image"

INPUT_WIDTH = 320
INPUT_HEIGHT = 240
CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.30
INFERENCE_INTERVAL = 0.5

# Model configuration
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

    print("Generated priors:", len(priors))

    return priors


PRIORS = generate_priors()


class Slim320FaceDetector:

    def __init__(self):
        rospy.init_node("slim320_face_test")

        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Warm-up
        dummy = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        self.infer(dummy)

        self.publisher = rospy.Publisher(
            OUTPUT_TOPIC, Image, queue_size=1
        )

        self.last_inference = 0
        self.last_report = time.time()

        self.subscriber = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("Slim-320 face detector started")

        rospy.spin()

    def infer(self, image):
        resized = cv2.resize(
            image, (INPUT_WIDTH, INPUT_HEIGHT)
        )

        # OpenCV camera image is BGR.
        # Model expects RGB input with mean 127 and scale 1/128.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(
            rgb,
            scalefactor=1.0 / 128.0,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            mean=(127, 127, 127),
            swapRB=False
        )

        self.net.setInput(blob)
        boxes, scores = self.net.forward(["boxes", "scores"])

        boxes = np.array(boxes).reshape(-1, 4)
        scores = np.array(scores).reshape(-1, 2)

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

            decoded_boxes.append([x1, y1, x2 - x1, y2 - y1])

        decoded_boxes = np.array(decoded_boxes, dtype=np.float32)

        face_scores = scores[:, 1]
        indices = np.where(face_scores >= CONFIDENCE_THRESHOLD)[0]

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

    def image_callback(self, msg):
        now = time.time()

        if now - self.last_inference < INFERENCE_INTERVAL:
            return

        self.last_inference = now

        try:
            height = msg.height
            width = msg.width

            image = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape(height, width, -1)

            if msg.encoding == "rgb8":
                image = cv2.cvtColor(
                    image, cv2.COLOR_RGB2BGR
                )
            elif msg.encoding != "bgr8":
                rospy.logwarn_throttle(
                    5, "Unsupported image encoding: " + msg.encoding
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

            if now - self.last_report >= 5:
                ram = self.get_ram_mb()

                rospy.loginfo(
                    "Faces: %d | Inference: %.1f ms | RAM: %.1f MB",
                    faces,
                    inference_ms,
                    ram
                )

                self.last_report = now

        except Exception as e:
            rospy.logerr_throttle(5, "Detection error: %s", str(e))

    @staticmethod
    def get_ram_mb():
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        return 0


if __name__ == "__main__":
    try:
        Slim320FaceDetector()
    except rospy.ROSInterruptException:
        pass
