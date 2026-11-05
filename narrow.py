import os
import cv2
import time
import rospy
import queue
import signal
import threading
import numpy as np

from threading import RLock
from math import ceil

from pug_app import common
from pug_app.common import Heart

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from pug_control.srv import SetActionName
from std_srvs.srv import Trigger, SetBool, TriggerRequest


# ============================================================
# RFB-320 MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/home/hiwonder/pug/src/pug_alif/"
    "shoe_detection/version-RFB-320.onnx"
)

INPUT_WIDTH = 320
INPUT_HEIGHT = 240

CONFIDENCE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.30

# Detect only the middle 50% horizontally
ROI_WIDTH_RATIO = 0.50

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

    return np.clip(
        np.array(priors, dtype=np.float32),
        0.0,
        1.0
    )


# ============================================================
# FACE DETECTION NODE
# ============================================================

class FaceDetectNode:

    def __init__(self, name):

        rospy.init_node(name, log_level=rospy.INFO)

        self.name = name

        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError("RFB model not found: " + MODEL_PATH)

        # Load RFB-320 ONNX model
        self.model = cv2.dnn.readNetFromONNX(MODEL_PATH)

        self.priors = generate_priors()

        rospy.loginfo("RFB-320 model loaded")
        rospy.loginfo("Model: %s", MODEL_PATH)
        rospy.loginfo("Prior boxes: %d", len(self.priors))
        rospy.loginfo("Detection area: Middle 50%% horizontally")

        self.image_queue = queue.Queue(maxsize=2)

        self.start = False
        self.image_sub = None

        self.lock = RLock()

        self.running = True
        self.detect_face = False
        self.face_count = 0

        self.start_greet = False
        self.action_finish = True
        self.enter = False

        signal.signal(signal.SIGINT, self.shutdown)

        # ROS publishers
        self.image_pub = rospy.Publisher(
            '~image_result',
            Image,
            queue_size=1
        )

        self.face_found_pub = rospy.Publisher(
            '~face_found',
            Bool,
            queue_size=1,
            latch=True
        )

        self.face_found_pub.publish(Bool(data=False))

        # ROS services
        rospy.Service(
            '~enter',
            Trigger,
            self.enter_func
        )

        rospy.Service(
            '~exit',
            Trigger,
            self.exit_func
        )

        rospy.Service(
            '~set_running',
            SetBool,
            self.set_running
        )

        # Heartbeat
        self.heart = Heart(
            self.name + '/heartbeat',
            5,
            lambda _: self.exit_func(None)
        )

        # Action group service
        self.run_action_group_srv = rospy.ServiceProxy(
            '/pug_control/run_action_group',
            SetActionName
        )

        # Movement thread
        threading.Thread(
            target=self.move,
            daemon=True
        ).start()

        rospy.loginfo("RFB-320 face detection node started")

        self.run()

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self, signum, frame):

        self.running = False

    # ========================================================
    # ENTER FACE DETECTION
    # ========================================================

    def enter_func(self, msg):

        rospy.loginfo("enter face detect")

        with self.lock:

            self.run_action_group_srv('sit')

            time.sleep(1)

            self.start_greet = False

            try:

                if self.image_sub is not None:

                    self.image_sub.unregister()
                    self.image_sub = None

            except Exception as error:

                rospy.logerr(str(error))

            self.image_sub = rospy.Subscriber(
                '/csi_camera/image_rect_color',
                Image,
                self.image_callback
            )

            self.enter = True

        return [True, 'enter']

    # ========================================================
    # EXIT FACE DETECTION
    # ========================================================

    def exit_func(self, msg):

        rospy.loginfo("exit face detect")

        with self.lock:

            self.start = False

            try:

                if self.image_sub is not None:

                    self.image_sub.unregister()
                    self.image_sub = None

            except Exception as error:

                rospy.logerr(str(error))

            if isinstance(msg, TriggerRequest):

                self.heart.stop()

                self.run_action_group_srv('up')

            self.enter = False

            self.face_count = 0
            self.start_greet = False

            self.face_found_pub.publish(
                Bool(data=False)
            )

        return [True, 'exit']

    # ========================================================
    # START / STOP FACE DETECTION
    # ========================================================

    def set_running(self, msg):

        if msg.data:

            rospy.loginfo("start running face detect")

            with self.lock:

                self.start = True
                self.face_count = 0

        else:

            rospy.loginfo("stop running face detect")

            with self.lock:

                self.start = False
                self.start_greet = False
                self.face_count = 0

        return [True, 'set_running']

    # ========================================================
    # ROBOT MOVEMENT
    # ========================================================

    def move(self):

        while self.running:

            if self.start:

                if self.start_greet:

                    self.start_greet = False

                    self.action_finish = False

                    self.run_action_group_srv('shake_hands1')

                    time.sleep(1)

                    self.action_finish = True

                else:

                    time.sleep(0.01)

            else:

                time.sleep(0.1)

    # ========================================================
    # RFB-320 FACE DETECTION
    # ========================================================

    def detect(self, frame):

        original_height, original_width = frame.shape[:2]

        # Calculate centre 50% detection region
        roi_width = int(original_width * ROI_WIDTH_RATIO)

        x_start = (original_width - roi_width) // 2
        x_end = x_start + roi_width

        # Crop horizontally, keeping full height
        roi = frame[:, x_start:x_end]

        resized = cv2.resize(
            roi,
            (INPUT_WIDTH, INPUT_HEIGHT)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        blob = cv2.dnn.blobFromImage(
            rgb,
            scalefactor=1.0 / 128.0,
            size=(INPUT_WIDTH, INPUT_HEIGHT),
            mean=(127, 127, 127),
            swapRB=False
        )

        self.model.setInput(blob)

        raw_boxes, raw_scores = self.model.forward(
            ["boxes", "scores"]
        )

        boxes = np.asarray(raw_boxes).reshape(-1, 4)
        scores = np.asarray(raw_scores).reshape(-1, 2)

        priors = self.priors

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

        candidate_boxes = []
        candidate_scores = []

        for index in candidates:

            x1, y1, x2, y2 = decoded[index]

            # Map ROI coordinates back to original frame
            x1 = int(x1 * roi_width) + x_start
            y1 = int(y1 * original_height)

            x2 = int(x2 * roi_width) + x_start
            y2 = int(y2 * original_height)

            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))

            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            candidate_boxes.append([
                x1,
                y1,
                width,
                height
            ])

            candidate_scores.append(
                float(face_scores[index])
            )

        face_count = 0

        if candidate_boxes:

            selected = cv2.dnn.NMSBoxes(
                candidate_boxes,
                candidate_scores,
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD
            )

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

        return frame, face_count

    # ========================================================
    # MAIN FACE DETECTION LOOP
    # ========================================================

    def run(self):

        while self.running:

            if self.enter:

                try:

                    image = self.image_queue.get(timeout=0.5)

                except queue.Empty:

                    continue

                if self.start:

                    self.detect_face = False

                    image, face_count = self.detect(image)

                    self.detect_face = face_count > 0

                    self.face_found_pub.publish(
                        Bool(data=self.detect_face)
                    )

                    if self.detect_face:

                        if self.action_finish:

                            self.face_count += 1

                            if self.face_count > 30:

                                self.start_greet = True

                        else:

                            self.face_count = 0

                    else:

                        self.face_count = 0

                self.image_pub.publish(
                    common.cv2_image2ros(image, self.name)
                )

            else:

                time.sleep(0.1)

        rospy.signal_shutdown('shutdown')

    # ========================================================
    # CAMERA CALLBACK
    # ========================================================

    def image_callback(self, ros_image):

        try:

            image = np.ndarray(
                shape=(ros_image.height, ros_image.width, 3),
                dtype=np.uint8,
                buffer=ros_image.data
            )

            cv2_img = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            if self.image_queue.full():

                self.image_queue.get_nowait()

            self.image_queue.put_nowait(cv2_img)

        except Exception as error:

            rospy.logerr_throttle(
                5,
                "Image callback error: %s",
                error
            )


if __name__ == '__main__':

    try:

        FaceDetectNode('face_detect_custom')

    except rospy.ROSInterruptException:

        pass

    except Exception as error:

        rospy.logfatal("RFB face detector failed: %s", error)

