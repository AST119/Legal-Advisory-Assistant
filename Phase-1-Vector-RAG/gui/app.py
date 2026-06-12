from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
API_URL = os.getenv("FASTAPI_URL")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    files = {'file': (file.filename, file.read())}
    response = requests.post(f"{API_URL}/upload", files=files)
    return jsonify(response.json())

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("question")
    response = requests.post(f"{API_URL}/chat", json={"question": user_input})
    return jsonify(response.json())



if __name__ == '__main__':
    app.run(port=5000, debug=True)