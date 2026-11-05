#!/usr/bin/env python3

import os
import time
import threading
import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from math import ceil


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "version-RFB-320.onnx"
)

CAMERA_TOPIC = "/csi_camera/image_rect_color"
OUTPUT_TOPIC = "/rfb_face_detector/image"

INPUT_WIDTH = 320
INPUT_HEIGHT = 240

CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.30
INFERENCE_INTERVAL = 0.5


MIN_BOXES = [
    [10.0, 16.0, 24.0],
    [32.0, 48.0],
    [64.0, 96.0],
    [128.0, 192.0, 256.0]
]

STRIDES = [8.0, 16.0, 32.0, 64.0]

CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2


def generate_priors():
    priors = []

    for index, stride in enumerate(STRIDES):
        feature_width = int(ceil(INPUT_WIDTH / stride))
        feature_height = int(ceil(INPUT_HEIGHT / stride))

        for j in range(feature_height):
            for i in range(feature_width):
                x_center = (i + 0.5) / (INPUT_WIDTH / stride)
                y_center = (j + 0.5) / (INPUT_HEIGHT / stride)

                for box_size in MIN_BOXES[index]:
                    priors.append([
                        x_center,
                        y_center,
                        box_size / INPUT_WIDTH,
                        box_size / INPUT_HEIGHT
                    ])

    return np.clip(np.array(priors, dtype=np.float32), 0.0, 1.0)


class RFBFaceDetector:

    def __init__(self):
        rospy.init_node("rfb_face_detector")

        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError("Model not found: " + MODEL_PATH)

        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.priors = generate_priors()

        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_inference = 0

        self.image_pub = rospy.Publisher(
            OUTPUT_TOPIC, Image, queue_size=1
        )

        self.sub = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**22
        )

        rospy.loginfo("RFB-320 face detector started")
        rospy.loginfo("Model: %s", MODEL_PATH)
        rospy.loginfo("Camera: %s", CAMERA_TOPIC)
        rospy.loginfo("Output: %s", OUTPUT_TOPIC)
        rospy.loginfo("Prior boxes: %d", len(self.priors))

    def image_callback(self, msg):
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding not in ("rgb8", "bgr8"):
                rospy.logerr_throttle(
                    5, "Unsupported camera encoding: %s", msg.encoding
                )
                return

            image = raw.reshape(msg.height, msg.step)

            image = image[:, :msg.width * 3]
            image = image.reshape(msg.height, msg.width, 3)

            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                frame = image.copy()

            with self.lock:
                self.latest_frame = frame

        except Exception as error:
            rospy.logerr_throttle(5, "Camera conversion error: %s", error)

    def detect(self, frame):
        original_height, original_width = frame.shape[:2]

        resized = cv2.resize(
            frame, (INPUT_WIDTH, INPUT_HEIGHT)
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(
            rgb,
            scalefactor=1.0 / 128.0,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            mean=(127, 127, 127),
            swapRB=False
        )

        self.net.setInput(blob)

        start = time.time()
        raw_boxes, raw_scores = self.net.forward(
            ["boxes", "scores"]
        )
        inference_ms = (time.time() - start) * 1000.0

        boxes = np.asarray(raw_boxes).reshape(-1, 4)
        scores = np.asarray(raw_scores).reshape(-1, 2)

        priors = self.priors

        # Decode predicted box coordinates.
        centers = (
            boxes[:, :2] * CENTER_VARIANCE * priors[:, 2:]
            + priors[:, :2]
        )

        sizes = (
            np.exp(boxes[:, 2:] * SIZE_VARIANCE)
            * priors[:, 2:]
        )

        decoded = np.concatenate([
            centers - sizes / 2,
            centers + sizes / 2
        ], axis=1)

        face_scores = scores[:, 1]

        candidates = np.where(
            face_scores > CONFIDENCE_THRESHOLD
        )[0]

        if len(candidates) == 0:
            return frame, inference_ms, 0

        candidate_boxes = []
        candidate_scores = []

        for index in candidates:
            x1, y1, x2, y2 = decoded[index]

            x1 = int(max(0, min(original_width, x1 * original_width)))
            y1 = int(max(0, min(original_height, y1 * original_height)))
            x2 = int(max(0, min(original_width, x2 * original_width)))
            y2 = int(max(0, min(original_height, y2 * original_height)))

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            candidate_boxes.append([x1, y1, width, height])
            candidate_scores.append(float(face_scores[index]))

        if not candidate_boxes:
            return frame, inference_ms, 0

        selected = cv2.dnn.NMSBoxes(
            candidate_boxes,
            candidate_scores,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        face_count = 0

        if len(selected) > 0:
            for selected_index in np.array(selected).flatten():
                x, y, width, height = candidate_boxes[selected_index]
                confidence = candidate_scores[selected_index]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face: {:.1f}%".format(confidence * 100),
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                face_count += 1

        return frame, inference_ms, face_count

    def get_ram_mb(self):
        try:
            with open("/proc/self/status", "r") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass

        return 0.0

    def run(self):
        last_report = time.time()

        while not rospy.is_shutdown():

            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None

            if frame is None:
                rospy.sleep(0.1)
                continue

            now = time.time()

            if now - self.last_inference < INFERENCE_INTERVAL:
                continue

            self.last_inference = now

            try:
                annotated, inference_ms, face_count = self.detect(frame)

                output_msg = Image()
                output_msg.header.stamp = rospy.Time.now()
                output_msg.header.frame_id = "csi_camera_link"
                output_msg.height = annotated.shape[0]
                output_msg.width = annotated.shape[1]
                output_msg.encoding = "bgr8"
                output_msg.is_bigendian = 0
                output_msg.step = annotated.shape[1] * 3
                output_msg.data = annotated.tobytes()

                self.image_pub.publish(output_msg)

                if time.time() - last_report >= 5:
                    rospy.loginfo(
                        "Faces: %d | Inference: %.1f ms | RAM: %.1f MB",
                        face_count,
                        inference_ms,
                        self.get_ram_mb()
                    )
                    last_report = time.time()

            except Exception as error:
                rospy.logerr("Detection error: %s", error)


if __name__ == "__main__":
    try:
        detector = RFBFaceDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("RFB detector failed: %s", error)
