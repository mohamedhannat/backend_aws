import jsonpickle
import numpy as np
import cv2
import os, sys
import logging
import yaml
import datetime
import time as tm
import socketio
import base64
from engineio.payload import Payload
from aiohttp import web
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

#ROOT = os.path.dirname(__file__)
#sys.path.append("src")


Payload.max_decode_packets = 50
sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

log = logging.getLogger('werkzeug')
log.disabled = True

@sio.on('connect')
def handle_connect(sid, environ):
    print('Client connected')

@sio.on('disconnect')
def handle_disconnect(sid):
    print('Client disconnected')



# route http posts to this method
@sio.on('send_img')
def send_img(sid, data):
    global streamprocess

    decoded_data=base64.b64decode(data['img'])
    # convert string of image data to uint8
    nparr = np.frombuffer(decoded_data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # read frame
    try:
        streamprocess.read_video_frame(img, data['ts'])#, frame_type='video')
    except Exception as e:
        print("exception"+repr(e))


async def state(request):
    response = {'message': 'OK'
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return web.Response(text=response_pickled)


async def stop_stream_process(request):
    global streamprocess
    streamprocess.stop()
    response_pickled = jsonpickle.encode({ "success": True, "message": "Server is shutting down..." })
    return web.Response(text=response_pickled)


async def check_signal():
    global streamprocess
    while(True):
        state = streamprocess.get_state()
        if(state!=""):
            await sio.emit('recv_signal',  {
                        'signal': state 
                        }
                    )
        await asyncio.sleep(0.05)

async def start_background_tasks(application):
    application['signal']= asyncio.create_task(check_signal())


async def cleanup_background_tasks(application):
    application['signal'].cancel()


if __name__ == "__main__":

    try:
        global streamprocess
        t1= str(datetime.datetime.now())
        from streamprocess import StreamProcess
        t_start= tm.time()
        streamprocess = StreamProcess()
        streamprocess.run()
        print("time needed to start streamprocess is:", tm.time()-t_start)
        
        app.router.add_get("/state", state)       
        app.router.add_get("/stop", stop_stream_process)
        
        app.on_startup.append(start_background_tasks)
        app.on_cleanup.append(cleanup_background_tasks)
       
        
        web.run_app(app, host='0.0.0.0', port=5000)        


    except KeyboardInterrupt:
        pass



