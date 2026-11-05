#!/bin/bash

# Start a new detached tmux session named "rospug_session"
tmux new-session -d -s rospug_session

# Pane 0 (Top): Camera Launch & nvargus restart
tmux send-keys -t rospug_session:0 "sudo systemctl restart nvargus-daemon && rosnode kill /csi_camera || true && roslaunch pug_peripherals csi_camera.launch" C-m

# Split window horizontally for Pane 1 (Bottom Left)
tmux split-window -v
tmux send-keys -t rospug_session:0.1 "python3 narrow.py" C-m

# Split bottom window vertically for Pane 2 (Bottom Right)
tmux split-window -h
tmux send-keys -t rospug_session:0.2 "python3 shoe_camera_test_single_onnx.py" C-m

# Attach to the tmux session so you can see all three running live
tmux attach-session -t rospug_session
