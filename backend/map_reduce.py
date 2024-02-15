import base64
import json
import os
import cv2
import numpy as np
import face_recognition
import tensorflow as tf

MAX_FRAMES = 20
EMOTION_DETECTION_MODEL = 'emotion_detection_model_vgg.h5'
EMOTION_MODEL = tf.keras.models.load_model(EMOTION_DETECTION_MODEL)
EMOTIONS_RESULTS = "emotion_results.json"
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_faces_emotions'

def mapper(video_path, output_folder):
    face_emotions = {}
    known_faces = {}  # Dictionary to store known faces and their IDs

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    # Process each frame in the video
    frame_count = 0
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Detect faces in the frame
        detections = face_recognition.face_locations(frame)

        # Process each detected face
        for i, detection in enumerate(detections):
            y1, x2, y2, x1 = detection

            # Extract face image
            face_image = frame[y1:y2, x1:x2]

            # Convert face image to RGB
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Extract face embeddings
            face_embedding = face_recognition.face_encodings(rgb_face)

            if len(face_embedding) > 0:
                # Use the first face embedding
                face_embedding = face_embedding[0]

                # Check if the face is known
                face_id = None
                for known_id, known_embedding in known_faces.items():
                    distance = np.linalg.norm(face_embedding - known_embedding)
                    if distance < 0.6:  # Adjust threshold as needed
                        face_id = known_id
                        break

                if face_id is None:
                    # Assign a new ID to the face
                    face_id = f'face_{len(known_faces) + 1}'
                    known_faces[face_id] = face_embedding

                # Preprocess face image for emotion detection
                resized_face = cv2.resize(rgb_face, (48, 48))
                normalized_face = resized_face.astype('float32') / 255.0
                input_face = np.expand_dims(normalized_face, axis=0)

                # Perform emotion detection
                emotion_probs = EMOTION_MODEL.predict(input_face)[0]
                emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
                emotion_str = ', '.join(f"{label}: {prob*100:.2f}%" for label, prob in zip(emotion_labels, emotion_probs))
                
                # Save face image
                face_filename = f'frame_{frame_count:04d}_{face_id}.jpg'
                cv2.imwrite(os.path.join(output_folder, face_filename), face_image)
                
                # Save emotion probabilities
                with open(os.path.join(output_folder, f'frame_{frame_count:04d}_{face_id}_emotions.txt'), 'w') as f:
                    f.write(emotion_str)

                # Aggregate emotion probabilities for each face
                if face_id not in face_emotions:
                    face_emotions[face_id] = emotion_probs
                else:
                    face_emotions[face_id] = np.maximum(face_emotions[face_id], emotion_probs)

    cap.release()

    return face_emotions.items()

def get_data_url(path):
    # Read the image file
    with open(path, 'rb') as f:
        image_data = f.read()
    
    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Construct the data URL
    data_url = f'data:image/jpeg;base64,{base64_image}'
    
    # Return the data URL as a response
    return data_url

def reducer(face_emotions_rdd, output_folder):
# Aggregate emotions based on face_id
    aggregated_emotions_rdd = face_emotions_rdd.reduceByKey(lambda x, y: np.maximum(x, y))

    # Collect all the results
    result = aggregated_emotions_rdd.collect()

    # Initialize a list to store JSON objects
    json_data = []

    # Format the data and append to the list
    for face_id, emotions in result:
        emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
        emotion_data = {label: f"{prob*100:.2f}%" for label, prob in zip(emotion_labels, emotions)}
        json_data.append({"face_id": face_id, "emotions": emotion_data, "image_path": get_data_url(OUTPUT_FOLDER + "/frame_0020_" + face_id + ".jpg")})

    # Write the JSON data to the file
    with open(EMOTIONS_RESULTS, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    
    # Create videos for each face
    for face_id, _ in result:
        create_video(output_folder, face_id)

    return list(result)

# Function to create a video from frames
def create_video(frames_folder, face_id):
    # Get all frames for the given face ID
    face_frames = [f for f in os.listdir(frames_folder) if f.endswith("_" + face_id + ".jpg")]
    if not face_frames:
        print(f"No frames found for face ID: {face_id}")
        return
    face_frames.sort(key=lambda x: int(x.split('_')[1]))

    # Initialize video writer
    frame_width, frame_height = cv2.imread(os.path.join(frames_folder, face_frames[0])).shape[1], cv2.imread(os.path.join(frames_folder, face_frames[0])).shape[0]
    out = cv2.VideoWriter(f'{face_id}_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # Write frames to video
    for frame in face_frames:
        frame_path = os.path.join(frames_folder, frame)
        img = cv2.imread(frame_path)
        out.write(img)

    # Release video writer
    out.release()