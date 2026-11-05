#!/bin/bash

# Terminal 1: Restart nvargus, kill old node, and launch camera
gnome-terminal -- tab --title="Camera Launch" -- bash -c "
echo 'Restarting nvargus-daemon...'
sudo systemctl restart nvargus-daemon
echo 'Killing existing csi_camera node...'
rosnode kill /csi_camera
echo 'Launching csi_camera...'
roslaunch pug_peripherals csi_camera.launch
exec bash
"

# Terminal 2: Run narrow.py
gnome-terminal -- tab --title="Narrow Script" -- bash -c "
echo 'Running narrow.py...'
python3 narrow.py
exec bash
"

# Terminal 3: Run ONNX camera test script
gnome-terminal -- tab --title="ONNX Test Script" -- bash -c "
echo 'Running shoe_camera_test_single_onnx.py...'
python3 shoe_camera_test_single_onnx.py
exec bash
"

echo "All terminal windows launched successfully!"
