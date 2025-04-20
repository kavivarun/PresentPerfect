from flask import Flask, request
from flask_socketio import SocketIO, emit
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow frontend origin

@app.route('/api/analyze', methods=['POST'])
def analyze():
    video = request.files['video']
    # Save or process file here
    print("logg")
    # Simulate async processing with progress updates
    def process_video():
        for i in range(1, 101, 10):
            time.sleep(0.5)  # simulate delay
            socketio.emit('processing-update', {'progress': i, 'message': f'Processed {i}%'})
        socketio.emit('processing-complete', {'message': 'Processing done!'})

    socketio.start_background_task(target=process_video)
    return {'status': 'processing started'}

if __name__ == '__main__':
    print("running")
    socketio.run(app, host='0.0.0.0', port=4000)