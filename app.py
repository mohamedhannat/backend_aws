import os
import logging
import json
import base64
import cv2
from pathlib import Path
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']

    # Decode image
    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        app.logger.error(f"Error decoding image: {e}")
        return jsonify(error="Error decoding image"), 400

    if frame is None or frame.size == 0:
        app.logger.error("Empty frame received")
        return jsonify(error="Empty frame"), 400

    # Process the frame (dummy processing example)
    detection_geometry = process_sift_frame(frame)

    return jsonify(success=bool(detection_geometry), detectionGeometry=detection_geometry) if detection_geometry else jsonify(success=False)

def process_sift_frame(frame):
    # Dummy processing logic
    return {'x': 10, 'y': 10, 'width': 100, 'height': 100}

@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:

        data = request.json
        logger.info(f"Starting Annotation with data: {data}")

        annotations = data['annotations']
        dataset_folder = data['datasetFolder']
        train_percent = data['trainPercent']
        val_percent = data['valPercent']
        test_percent = data['testPercent']
        tags = data['tags']

        base_dir = os.path.join(dataset_folder)
        os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'valid', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

        train, val, test = [], [], []

        for item in annotations:
            label = item['label']
            if label not in tags:
                continue

            rand = os.urandom(1)[0] / 255.0
            if rand < float(train_percent) / 100:
                train.append(item)
            elif rand < (float(train_percent) + float(val_percent)) / 100:
                val.append(item)
            else:
                test.append(item)

        def save_annotations(data, type):
            for anno in data:
                image_id = anno['imageId']
                label = anno['label']
                try:
                    label_index = tags.index(label)
                except ValueError:
                    continue
                x_center = anno['x_center'] / 100
                y_center = anno['y_center'] / 100
                bbox_width = anno['bbox_width'] / 100
                bbox_height = anno['bbox_height'] / 100
                image_data = anno['imageData']

                image_filename = os.path.basename(image_id) + '.png'
                label_filename = os.path.basename(image_id) + '.txt'

                label_file = os.path.join(base_dir, type, 'labels', label_filename)
                with open(label_file, 'a') as f:
                    annotation_str = f"{label_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                    f.write(annotation_str)

                image_file = os.path.join(base_dir, type, 'images', image_filename)
                with open(image_file, 'wb') as f:
                    try:
                        if image_data.startswith('data:image'):
                            f.write(base64.b64decode(image_data.split(',')[1]))
                    except Exception as e:
                        continue

        save_annotations(train, 'train')
        save_annotations(val, 'valid')
        save_annotations(test, 'test')

        data_yaml = f"""
        train: {os.path.join(base_dir, 'train', 'images')}
        val: {os.path.join(base_dir, 'valid', 'images')}
        nc: {len(tags)}
        names: {json.dumps(tags)}
        """

        data_yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml)

        return jsonify(message='Annotations saved and training data prepared successfully.', data_yaml_path=data_yaml_path)
    except Exception as e:
        return jsonify(error=f'Failed to parse request data. {str(e)}'), 500
if __name__ == "__main__":
    # threading.Thread(target=generate_frames, args=("runs/train/exp3/weights/best.pt", 0, (640, 640), 0.25, 0.45, 1000, "", None, False, 3, False, False, False, False)).start()
    app.run(host="0.0.0.0", port=5000)
