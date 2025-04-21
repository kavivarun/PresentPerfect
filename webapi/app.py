import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import os
import tempfile

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    video = request.files['video']

    if not video:
        return {'error': 'No video uploaded'}, 400

    # Save to temp file first to avoid closed file issues
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, video.filename)
    print(temp_path)
    video.save(temp_path)
    file_size = os.path.getsize(temp_path)

    # Log details
    print(f"[INFO] Uploaded video: {video.filename}")
    print(f"[INFO] Content-Type: {video.content_type}")
    print(f"[INFO] File Size: {file_size / 1024:.2f} KB")
    # e.g. video.save(...)
    def process_video():
        for i in range(1, 101, 10):

            socketio.emit(
                'processing-update',
                {'progress': i, 'message': f'Processed {i}'}
            )
            time.sleep(1)
        socketio.emit(
            'processing-complete',
            {'message': 'Processing done!', 'progress':f'100'}
        )

    socketio.start_background_task(process_video)
    return {'status': 'processing started'}

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=True)
