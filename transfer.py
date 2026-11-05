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