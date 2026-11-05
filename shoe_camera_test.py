#!/usr/bin/env python3
import time
import cv2
import numpy as np
import rospy
import onnxruntime as ort

from sensor_msgs.msg import Image

MODEL_PATH = (
    "/home/hiwonder/pug/src/pug_alif/"
    "shoe_detection/shoe_yolov8n.onnx"
)
CAMERA_TOPIC = "/csi_camera/image_rect_color"
CLASS_NAMES = ["shoe", "shoe_other"]

IMG_SIZE = 416
CONF_THRESHOLD = 0.40
INFERENCE_INTERVAL = 0.5


class ShoeCameraTest:
    def __init__(self):
        rospy.init_node("shoe_camera_test", anonymous=False)

        self.last_inference = 0.0
        self.frame_count = 0
        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        rospy.loginfo("YOLO model loaded")
        rospy.loginfo("Subscribing to %s", CAMERA_TOPIC)

        self.sub = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_inference < INFERENCE_INTERVAL:
            return

        self.last_inference = now
        self.frame_count += 1

        try:
            # Decode rgb8 ROS image data directly; bypass cv_bridge.
            if msg.encoding != "rgb8":
                rospy.logerr_throttle(
                    5.0, "Expected rgb8 image, received %s", msg.encoding
                )
                return

            frame_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.step // 3, 3
            )[:, :msg.width, :]
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]

            # Resize to the model's required input size.
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

            output = self.session.run(
                None, {self.input_name: tensor}
            )[0][0]

            # YOLOv8 output: [4 box coordinates + 2 class scores, N].
            predictions = output.T
            boxes = predictions[:, :4]
            class_scores = predictions[:, 4:]

            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[
                np.arange(len(class_scores)), class_ids
            ]

            selected = np.where(confidences >= CONF_THRESHOLD)[0]

            detections = []
            for idx in selected:
                cx, cy, bw, bh = boxes[idx]

                # Convert model-input coordinates to original-frame
                # coordinates. This test uses direct resizing.
                scale_x = width / float(IMG_SIZE)
                scale_y = height / float(IMG_SIZE)

                x1 = int((cx - bw / 2) * scale_x)
                y1 = int((cy - bh / 2) * scale_y)
                x2 = int((cx + bw / 2) * scale_x)
                y2 = int((cy + bh / 2) * scale_y)

                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": float(confidences[idx]),
                    "class_id": int(class_ids[idx])
                })

            # Suppress overlapping boxes separately per class.
            final_detections = []
            for class_id in range(len(CLASS_NAMES)):
                class_dets = [
                    d for d in detections
                    if d["class_id"] == class_id
                ]

                if not class_dets:
                    continue

                indices = cv2.dnn.NMSBoxes(
                    [d["box"] for d in class_dets],
                    [d["confidence"] for d in class_dets],
                    CONF_THRESHOLD,
                    0.45
                )

                if len(indices) > 0:
                    for index in np.array(indices).reshape(-1):
                        final_detections.append(class_dets[int(index)])

            if final_detections:
                details = []
                for det in final_detections:
                    name = CLASS_NAMES[det["class_id"]]
                    details.append(
                        "{} {:.2f}".format(name, det["confidence"])
                    )

                rospy.loginfo(
                    "Frame %d (%dx%d): %s",
                    self.frame_count, width, height,
                    ", ".join(details)
                )
            else:
                rospy.loginfo(
                    "Frame %d (%dx%d): no detections above %.2f",
                    self.frame_count, width, height, CONF_THRESHOLD
                )

        except Exception as exc:
            rospy.logerr("Camera inference error: %s", str(exc))


if __name__ == "__main__":
    try:
        ShoeCameraTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
