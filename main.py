from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import face_recognition
import numpy as np
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)
run_with_ngrok(app)

def fetch_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(img)

@app.route('/face-embedding', methods=['GET'])
def face_embedding():
    image_url = request.args.get('url')
    if not image_url:
        return jsonify({"error": "Missing image URL parameter"}), 400
    try:
        img_np = fetch_image(image_url)
        face_locations = face_recognition.face_locations(img_np, model="hog")

        if len(face_locations) == 0:
            return jsonify({"message": "No face detected"}), 200

        embeddings = face_recognition.face_encodings(img_np, face_locations)

        # Return only the first face embedding for simplicity
        embedding_list = embeddings[0].tolist()

        return jsonify({"face_embedding": embedding_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
