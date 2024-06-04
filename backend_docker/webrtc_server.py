import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from json import JSONDecodeError
import cv2
import aiohttp_cors
import requests
from aiohttp import web
from av import VideoFrame
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
import sys
#import torch
import time
import datetime
#import librosa
from collections import deque
import socketio
import base64

ROOT = os.path.dirname(__file__)

sys.path.append(os.path.join(ROOT, "../utils"))

logger = logging.getLogger("pc")
pcs = set()

our_channel= None
video_track= None
connection_alive = False
video_relay = MediaRelay()
sio = socketio.AsyncClient()


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    global video_track
    global connection_alive 

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    
    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    
    @pc.on("datachannel")
    def on_datachannel(channel):
        global our_channel
        our_channel= channel
        @channel.on("message")
        async def on_message(message):
            if isinstance(message, str):
                if(message.startswith("ping")):
                    channel.send("pong" + message[4:])
                else:
                    pass

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global video_track
        global connection_alive
        log_info("Track %s received", track.kind)
                
        if track.kind == "audio":
            pass
        elif track.kind == "video":
            video_track = video_relay.subscribe(track)
            connection_alive = True

        @track.on("ended")
        async def on_ended():
            global video_track
            global connection_alive
            log_info("Track %s ended", track.kind)
            connection_alive= False
            video_track = None
            await pc.close()
            pcs.discard(pc)

    # handle offer
    await pc.setRemoteDescription(offer)
    
    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def healthcheck(request):
    return web.Response(
        content_type="application/text",
        text='healthy',
    )


@sio.on('recv_signal')
async def recv_signal(data):
    await signals_queue.put(data)


async def fe_signals_sender():
    global our_channel
    global connection_alive
    global video_track

    while(True):
        try:
            while(video_track is None or not connection_alive):
                await asyncio.sleep(0.5)
            
            if(our_channel is not None):
                signal_to_send= await signals_queue.get()
                signal_to_send= signal_to_send['signal']
                if(signal_to_send!=""):
                    our_channel.send(signal_to_send)
                    await our_channel._RTCDataChannel__transport._data_channel_flush()
                    await our_channel._RTCDataChannel__transport._transmit()
            else:
                await asyncio.sleep(1)
        
        except asyncio.CancelledError as ex:
            exit(1)
        except Exception as e:
            logger.info('fe signal sender Exception')

async def read_video_data():
    global video_track
    global connection_alive

    
    logger.info('starting read video data')
    rows=None
    cols=None

    while(True):
        try:
            while(video_track is None or not connection_alive):
                await asyncio.sleep(0.5)
            
            frame = await video_track.recv()
            ts = time.time() 
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            if(rows is None and cols is None):
                rows, cols, _ = img.shape
            
            img=cv2.resize(img, (cols, rows))
            _, img_encoded = cv2.imencode('.jpg', img)
            
            # send image data throught websocket
            data = {'img': base64.b64encode(img_encoded),
                    'ts': ts
                    }
            await sio.emit('send_img', data)
       
        except asyncio.CancelledError as ex:
            exit(1)
        except Exception as e:
            logger.info('read video data: Exception')



async def start_background_tasks(application):
    global signals_queue
    signals_queue = asyncio.Queue()

    await sio.connect('http://0.0.0.0:5000')
    application['fe_signal_sender']= asyncio.create_task(fe_signals_sender())
    application['read_video_data']= asyncio.ensure_future(read_video_data())
    #application['read_video_data']= asyncio.ensure_future(read_video_data())



async def cleanup_background_tasks(application):
    logger.info('Cleaning up...')
    application['fe_signal_sender'].cancel()
    application['read_video_data'].cancel()
    


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    await sio.disconnect()




if __name__ == "__main__":
    
    global app
    
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    #app.router.add_get("/", index)
    #app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    
    
    app.router.add_get("/healthcheck", healthcheck)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    # core config
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    for route in list(app.router.routes()):
        cors.add(route)

    # wait until streamprocess server start 
    while(True):
        try:
            response = requests.get('http://0.0.0.0:5000/state', verify=False)
            break
        except:
            time.sleep(0.5)

    
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
