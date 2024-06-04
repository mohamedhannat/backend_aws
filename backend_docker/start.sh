#!/bin/bash
#run webrtc server in background
python3 app.py &
python3 -u webrtc_server.py 2>&1 | tee -a aiortc.log &
##run stream process server
python3 -u stream_process_server.py 2>&1 | tee -a app.log 

