import json
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import threading
from map_reduce import mapper, reducer
from pyspark import SparkContext, SparkConf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

EMOTIONS_RESULTS = "emotion_results.json"
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_faces_emotions'

# Initialize Spark context
conf = SparkConf().setAppName("FaceEmotionAnalysis")
sc = SparkContext(conf=conf)

# Create the uploads and output_faces_emotions directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Endpoint to upload a video file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'})

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected video'})

    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

    response = jsonify({'message': 'File uploaded and processed successfully', 'video_path': video_path})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response 
    

@app.route('/extract_emotions', methods=['POST'])
def extract():
    data = request.json
    if 'video_path' not in data:
        return jsonify({'error': 'Video path not provided'})

    video_path = data['video_path']

    face_emotions_rdd = sc.parallelize(mapper(video_path, OUTPUT_FOLDER))

    reducer(face_emotions_rdd, OUTPUT_FOLDER)
    
    response = jsonify({'message': 'Emotions extracted successfully'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

    
# Endpoint to fetch results
@app.route('/results', methods=['GET'])
def fetch_results():
    try:
        with open(EMOTIONS_RESULTS, "r") as f:
            emotions_data = json.load(f)
        return jsonify(emotions_data)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_video', methods=['GET'])
def get_video():
    face_id = request.args.get('face_id')
    if face_id:
        # Assuming the video file path follows a specific format
        video_path = f"{face_id}_video.mp4"
        return send_file(video_path, mimetype='video/mp4')
    else:
        return 'Face ID not provided', 400  # Bad Request

# Function to run Flask app in a separate thread
def run_flask_app():
    app.run(host='0.0.0.0', port=5000)

# Start Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

# Wait for the Flask app to start
while not app.running:
    pass

# Stop Flask app when Spark job is finished
flask_thread.join