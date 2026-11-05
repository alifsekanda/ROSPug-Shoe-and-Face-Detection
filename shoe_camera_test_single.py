
#!/usr/bin/env python3
import time
import threading
import cv2
import numpy as np
import rospy
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, SetBool
from std_msgs.msg import Bool

ENGINE_PATH = (
    "/home/hiwonder/pug/src/pug_alif/"
    "shoe_detection/shoe_yolov8n_single_fp16.engine"
)
CAMERA_TOPIC = "/csi_camera/image_rect_color"

IMG_SIZE = 416
CONF_THRESHOLD = 0.70
INFERENCE_INTERVAL = 0.5
NMS_THRESHOLD = 0.45
REQUIRED_SHOE_FRAMES = 5
FACE_TIMEOUT_SECONDS = 10.0


class ShoeCameraTestSingle:
    def __init__(self):
        rospy.init_node("shoe_camera_test_single", anonymous=False)
        self.annotated_pub = rospy.Publisher(
            "/shoe_detection/annotated_image", Image, queue_size=1
        )

        self.last_inference = 0.0
        self.frame_count = 0
        self.consecutive_shoe_frames = 0

        self.inference_running = False
        self.inference_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.latest_detections = []
        self.face_mode_triggered = False
        self.face_found = False
        self.face_timeout_timer = None
        self.state_lock = threading.Lock()

        self.face_sub = rospy.Subscriber(
            '/face_detect_custom/face_found', Bool,
            self.face_found_callback, queue_size=1
        )

        # TensorRT engine and CUDA buffers.
        self.cuda_context = pycuda.autoinit.context
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.trt_logger)

        with open(ENGINE_PATH, "rb") as engine_file:
            self.engine = self.trt_runtime.deserialize_cuda_engine(
                engine_file.read()
            )

        if self.engine is None:
            raise RuntimeError("Could not load TensorRT engine")

        self.trt_context = self.engine.create_execution_context()
        if self.trt_context is None:
            raise RuntimeError("Could not create TensorRT execution context")

        self.input_index = self.engine.get_binding_index("images")
        self.output_index = self.engine.get_binding_index("output0")

        self.input_shape = tuple(
            self.engine.get_binding_shape(self.input_index)
        )
        self.output_shape = tuple(
            self.engine.get_binding_shape(self.output_index)
        )

        if self.input_shape != (1, 3, IMG_SIZE, IMG_SIZE):
            raise RuntimeError(
                "Unexpected TensorRT input shape: %s" % (self.input_shape,)
            )

        if self.output_shape != (1, 5, 3549):
            raise RuntimeError(
                "Unexpected TensorRT output shape: %s" % (self.output_shape,)
            )

        self.host_input = cuda.pagelocked_empty(
            trt.volume(self.input_shape), dtype=np.float32
        )
        self.host_output = cuda.pagelocked_empty(
            trt.volume(self.output_shape), dtype=np.float32
        )

        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.device_input)
        self.bindings[self.output_index] = int(self.device_output)

        rospy.loginfo("Single-class YOLO TensorRT engine loaded")
        rospy.loginfo("Engine: %s", ENGINE_PATH)
        rospy.loginfo("Input shape: %s", self.input_shape)
        rospy.loginfo("Output shape: %s", self.output_shape)
        rospy.loginfo("Subscribing to %s", CAMERA_TOPIC)

        self.sub = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

    def face_found_callback(self, msg):
        timer_to_cancel = None

        with self.state_lock:
            previously_found = self.face_found
            self.face_found = bool(msg.data)

            if msg.data:
                timer_to_cancel = self.face_timeout_timer
                self.face_timeout_timer = None

            elif (
                previously_found
                and self.face_mode_triggered
                and self.face_timeout_timer is None
            ):
                self.face_timeout_timer = threading.Timer(
                    FACE_TIMEOUT_SECONDS, self.face_timeout_callback
                )
                self.face_timeout_timer.daemon = True
                self.face_timeout_timer.start()
                rospy.loginfo(
                    "Face lost; starting %.0f-second timeout",
                    FACE_TIMEOUT_SECONDS
                )

        if timer_to_cancel is not None:
            timer_to_cancel.cancel()
            rospy.loginfo("Face detected; timeout cancelled")

    def face_timeout_callback(self):
        with self.state_lock:
            if not self.face_mode_triggered or self.face_found:
                return
            self.face_timeout_timer = None

        rospy.logwarn("No face detected within 10 seconds; exiting face mode")

        try:
            rospy.wait_for_service("/face_detect_custom/exit", timeout=5.0)
            exit_face = rospy.ServiceProxy(
                "/face_detect_custom/exit", Trigger
            )
            response = exit_face()
            rospy.loginfo(
                "Face exit response: success=%s, message=%s",
                response.success, response.message
            )
        except Exception as exc:
            rospy.logerr("Could not exit custom face detector: %s", str(exc))

        with self.state_lock:
            self.face_mode_triggered = False
            self.face_found = False
            self.consecutive_shoe_frames = 0

        rospy.loginfo("Reset complete; waiting for the next shoe")

    def trigger_face_detection(self):
        with self.state_lock:
            if self.face_mode_triggered:
                return
            self.face_found = False

        try:
            rospy.loginfo("Shoe detected for 5 consecutive frames")

            rospy.wait_for_service(
                "/face_detect_custom/enter", timeout=5.0
            )
            rospy.wait_for_service(
                "/face_detect_custom/set_running", timeout=5.0
            )

            enter_face = rospy.ServiceProxy(
                "/face_detect_custom/enter", Trigger
            )
            set_running = rospy.ServiceProxy(
                "/face_detect_custom/set_running", SetBool
            )

            enter_response = enter_face()
            rospy.loginfo(
                "Custom face enter response: success=%s, message=%s",
                enter_response.success, enter_response.message
            )
            if not enter_response.success:
                rospy.logerr("Custom face detector enter failed")
                return

            running_response = set_running(True)
            rospy.loginfo(
                "Custom face running response: success=%s, message=%s",
                running_response.success, running_response.message
            )
            if not running_response.success:
                rospy.logerr("Could not activate custom face detector")
                try:
                    rospy.ServiceProxy(
                        "/face_detect_custom/exit", Trigger
                    )()
                except Exception:
                    pass
                return

            with self.state_lock:
                self.face_mode_triggered = True
                self.face_found = False
                self.face_timeout_timer = threading.Timer(
                    FACE_TIMEOUT_SECONDS, self.face_timeout_callback
                )
                self.face_timeout_timer.daemon = True
                self.face_timeout_timer.start()

            rospy.loginfo(
                "Custom face detection activated; waiting up to %.0f seconds",
                FACE_TIMEOUT_SECONDS
            )

        except Exception as exc:
            rospy.logerr("Custom face detection trigger failed: %s", str(exc))

    def image_callback(self, msg):
        # Display every camera frame, independently of YOLO inference.
        try:
            if msg.encoding != "rgb8":
                rospy.logerr_throttle(
                    5.0, "Expected rgb8 image, received %s", msg.encoding
                )
                return

            frame_rgb = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape(
                msg.height, msg.step // 3, 3
            )[:, :msg.width, :]

            # Draw straight on an RGB copy; green is the same in RGB and BGR.
            frame = frame_rgb.copy()
            height, width = frame.shape[:2]

            with self.detection_lock:
                detections = list(self.latest_detections)

            for detection in detections:
                x, y, w, h = detection["box"]
                confidence = detection["confidence"]
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                cv2.putText(
                    frame, "shoe {:.2f}".format(confidence),
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            annotated_msg = Image()
            annotated_msg.header = msg.header
            annotated_msg.encoding = "rgb8"
            annotated_msg.height = height
            annotated_msg.width = width
            annotated_msg.step = width * 3
            annotated_msg.data = frame.tobytes()
            self.annotated_pub.publish(annotated_msg)

            now = time.time()
            if now - self.last_inference >= INFERENCE_INTERVAL:
                with self.inference_lock:
                    if (
                        not self.inference_running
                        and now - self.last_inference >= INFERENCE_INTERVAL
                    ):
                        self.inference_running = True
                        self.last_inference = now
                        threading.Thread(
                            target=self._inference_worker,
                            args=(msg,),
                            daemon=True
                        ).start()

        except Exception as exc:
            rospy.logerr_throttle(
                5.0, "Camera display error: %s", str(exc)
            )

    def _inference_worker(self, msg):
        context_pushed = False
        try:
            # CUDA contexts must be made current on this worker thread.
            self.cuda_context.push()
            context_pushed = True
            self.process_inference(msg)
        except Exception as exc:
            rospy.logerr("TensorRT worker error: %s", str(exc))
        finally:
            if context_pushed:
                try:
                    cuda.Context.pop()
                except Exception as exc:
                    rospy.logerr("CUDA context cleanup error: %s", str(exc))

            with self.inference_lock:
                self.inference_running = False

    def process_inference(self, msg):
        self.frame_count += 1

        try:
            if msg.encoding != "rgb8":
                rospy.logerr_throttle(
                    5.0,
                    "Expected rgb8 image, received %s",
                    msg.encoding
                )
                return

            frame_rgb = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape(
                msg.height, msg.step // 3, 3
            )[:, :msg.width, :]

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame.shape[:2]

            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

            # Copy the preprocessed tensor to the GPU, run TensorRT,
            # then copy predictions back for the existing postprocessing.
            np.copyto(self.host_input, tensor.ravel())
            cuda.memcpy_htod(self.device_input, self.host_input)

            success = self.trt_context.execute_v2(self.bindings)
            if not success:
                raise RuntimeError("TensorRT execute_v2 failed")

            cuda.memcpy_dtoh(self.host_output, self.device_output)
            output = self.host_output.reshape(self.output_shape)

            if output.shape != (1, 5, 3549):
                rospy.logerr(
                    "Unexpected model output shape: %s",
                    str(output.shape)
                )
                return

            predictions = output[0].T
            boxes = predictions[:, :4]
            confidences = predictions[:, 4]

            selected = np.where(
                confidences >= CONF_THRESHOLD
            )[0]

            scale_x = width / float(IMG_SIZE)
            scale_y = height / float(IMG_SIZE)

            candidate_boxes = []
            candidate_scores = []

            for idx in selected:
                cx, cy, bw, bh = boxes[idx]

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

                candidate_boxes.append(
                    [x1, y1, x2 - x1, y2 - y1]
                )
                candidate_scores.append(float(confidences[idx]))

            final_detections = []

            if candidate_boxes:
                indices = cv2.dnn.NMSBoxes(
                    candidate_boxes,
                    candidate_scores,
                    CONF_THRESHOLD,
                    NMS_THRESHOLD
                )

                if len(indices) > 0:
                    for i in np.array(indices).reshape(-1):
                        final_detections.append({
                            "confidence": candidate_scores[int(i)],
                            "box": candidate_boxes[int(i)]
                        })

            # Save the newest detections for the live display callback.
            with self.detection_lock:
                self.latest_detections = [
                    {
                        "confidence": detection["confidence"],
                        "box": list(detection["box"])
                    }
                    for detection in final_detections
                ]

            if final_detections:
                self.consecutive_shoe_frames += 1

                rospy.loginfo(
                    "Frame %d: SHOE DETECTED (%d/%d)",
                    self.frame_count,
                    self.consecutive_shoe_frames,
                    REQUIRED_SHOE_FRAMES
                )

                if (
                    self.consecutive_shoe_frames >= REQUIRED_SHOE_FRAMES
                    and not self.face_mode_triggered
                ):
                    self.trigger_face_detection()
            else:
                if self.consecutive_shoe_frames > 0:
                    rospy.loginfo("Shoe lost; resetting counter")

                self.consecutive_shoe_frames = 0

                rospy.loginfo(
                    "Frame %d (%dx%d): no detections above %.2f",
                    self.frame_count,
                    width,
                    height,
                    CONF_THRESHOLD
                )

        except Exception as exc:
            rospy.logerr("Camera inference error: %s", str(exc))


if __name__ == "__main__":
    try:
        ShoeCameraTestSingle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
