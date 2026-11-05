
#!/usr/bin/env python3
# encoding: utf-8

import os
import time
import threading
import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from pug_app.src.yunet import YuNet


# Original Hiwonder YuNet model
MODEL_PATH = (
    "/home/hiwonder/pug/src/pug_app/models/"
    "face_detection_yunet_2023mar.onnx"
)

CAMERA_TOPIC = "/csi_camera/image_rect_color"
OUTPUT_TOPIC = "/yunet_face_detector/image"

# Same processing resolution as face_detect_custom.py
INPUT_WIDTH = 640
INPUT_HEIGHT = 360

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# CPU backend, matching the original Hiwonder implementation
BACKEND_ID = cv2.dnn.DNN_BACKEND_OPENCV
TARGET_ID = cv2.dnn.DNN_TARGET_CPU

INFERENCE_INTERVAL = 0.5


class YuNetFaceDetector:

    def __init__(self):

        rospy.init_node("yunet_face_detector")

        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError("Model not found: " + MODEL_PATH)

        # Load the original Hiwonder YuNet implementation
        self.detector = YuNet(
            modelPath=MODEL_PATH,
            inputSize=(INPUT_WIDTH, INPUT_HEIGHT),
            confThreshold=CONFIDENCE_THRESHOLD,
            nmsThreshold=NMS_THRESHOLD,
            topK=TOP_K,
            backendId=BACKEND_ID,
            targetId=TARGET_ID
        )

        # Warm-up inference, matching face_detect_custom.py
        self.detector.infer(
            np.zeros(
                (INPUT_HEIGHT, INPUT_WIDTH, 3),
                dtype=np.uint8
            )
        )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_inference = 0

        self.image_pub = rospy.Publisher(
            OUTPUT_TOPIC,
            Image,
            queue_size=1
        )

        self.sub = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**22
        )

        rospy.loginfo("Hiwonder YuNet face detector started")
        rospy.loginfo("Model: %s", MODEL_PATH)
        rospy.loginfo("Camera: %s", CAMERA_TOPIC)
        rospy.loginfo("Output: %s", OUTPUT_TOPIC)
        rospy.loginfo("Input size: %dx%d", INPUT_WIDTH, INPUT_HEIGHT)
        rospy.loginfo("Backend: OpenCV CPU")

    def image_callback(self, msg):

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding not in ("rgb8", "bgr8"):
                rospy.logerr_throttle(
                    5,
                    "Unsupported camera encoding: %s",
                    msg.encoding
                )
                return

            image = raw.reshape(msg.height, msg.step)
            image = image[:, :msg.width * 3]
            image = image.reshape(msg.height, msg.width, 3)

            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )
            else:
                frame = image.copy()

            with self.lock:
                self.latest_frame = frame

        except Exception as error:
            rospy.logerr_throttle(
                5,
                "Camera conversion error: %s",
                error
            )

    def detect(self, frame):

        resized = cv2.resize(
            frame,
            (INPUT_WIDTH, INPUT_HEIGHT)
        )

        start = time.time()

        results = self.detector.infer(resized)

        inference_ms = (time.time() - start) * 1000.0

        face_count = 0

        if results is not None:

            for det in results:

                bbox = det[0:4].astype(np.int32)

                x, y, width, height = bbox

                confidence = float(det[-1])

                cv2.rectangle(
                    resized,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    resized,
                    "Face: {:.1f}%".format(confidence * 100),
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                face_count += 1

        return resized, inference_ms, face_count

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

                rospy.logerr(
                    "Detection error: %s",
                    error
                )


if __name__ == "__main__":

    try:

        detector = YuNetFaceDetector()
        detector.run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:

        rospy.logfatal(
            "YuNet detector failed: %s",
            error
        )