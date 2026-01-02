"""
Video Background Remover Web Application
Flask-based app that removes backgrounds from videos using rembg and FFmpeg.
"""

import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from io import BytesIO
import time # Added for tracking processing time
import json # Added for parsing ffprobe output
import psutil # Added for resource monitoring

# Add cuDNN to PATH for GPU support
cudnn_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9",
    r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1",
]
for cudnn_path in cudnn_paths:
    if Path(cudnn_path).exists():
        os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')
        print(f"Added cuDNN path: {cudnn_path}")
        break

from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

import threading

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['FRAMES_FOLDER'] = Path(__file__).parent / 'frames'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'output'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'results'

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'webm'}
ALLOWED_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/webm'}

# Global dictionary to store processing progress and sessions
processing_progress = {}
sessions = {}

# Create required directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['FRAMES_FOLDER'],
               app.config['OUTPUT_FOLDER'], app.config['RESULTS_FOLDER']]:
    folder.mkdir(parents=True, exist_ok=True)


def get_session(model_name):
    """Get or create rembg session and return (session, device_name)."""
    global sessions
    if model_name not in sessions:
        device = "GPU"
        try:
            print(f"Loading model {model_name} with GPU/Default providers...")
            session = new_session(model_name)
            print(f"Successfully loaded {model_name} on GPU.")
        except Exception as e:
            print(f"Error loading model {model_name} on GPU: {e}")
            device = "CPU"
            try:
                print(f"Falling back to CPUExecutionProvider for {model_name}...")
                session = new_session(model_name, providers=['CPUExecutionProvider'])
                print(f"Successfully loaded {model_name} on CPU.")
            except Exception as e2:
                print(f"Fatal error loading {model_name}: {e2}")
                session = new_session("u2net_human_seg", providers=['CPUExecutionProvider'])
                model_name = "u2net_human_seg"
                device = "CPU (Fallback)"
        
        sessions[model_name] = {'session': session, 'device': device}
                
    return sessions[model_name]['session'], sessions[model_name]['device']



def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_info(video_path):
    """Get video FPS, frame count, width, height, and duration using FFprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,duration',
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        duration = float(stream.get('duration', 0))
        
        # Calculate FPS
        fps_val = stream.get('avg_frame_rate', '30/1')
        if '/' in fps_val:
            num, den = map(int, fps_val.split('/'))
            fps = num / den if den != 0 else 30
        else:
            fps = float(fps_val)

        # Get frame count (approximate from duration and fps)
        frame_count = int(duration * fps) if duration and fps else 0
        
        return fps, frame_count, width, height, duration
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 30, 0, 0, 0, 0  # Default values


def extract_frames(video_path, frames_dir, job_id):
    """Extract frames from video using FFmpeg."""
    global processing_progress
    processing_progress[job_id]['status'] = 'フレーム分解中'
    
    # Get video info
    fps, frame_count, width, height, duration = get_video_info(video_path)
    
    # Store metadata
    processing_progress[job_id]['metadata'] = {
        'width': width,
        'height': height,
        'duration': round(duration, 2),
        'fps': round(fps, 2)
    }

    # Extract frames
    output_pattern = str(frames_dir / 'frame_%06d.png')
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vf', f'fps={min(fps, 30)}',  # Cap at 30 fps
        output_pattern
    ]
    
    # Update status just before running
    processing_progress[job_id]['status'] = 'フレーム分解中 (実行中...)'
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {process.stderr}")
    
    # Count extracted frames
    frames = sorted(frames_dir.glob('frame_*.png'))
    processing_progress[job_id]['total'] = len(frames)
    
    return frames, fps


def remove_background(frames, output_dir, job_id, model_name="u2net_human_seg"):
    """Remove background from each frame using rembg."""
    global processing_progress
    
    if not REMBG_AVAILABLE:
        raise Exception("rembg is not installed.")
    
    processing_progress[job_id]['status'] = 'AIモデルをロード中...'
    
    # Create or get rembg session for better performance
    session, device = get_session(model_name)
    processing_progress[job_id]['device'] = device
    
    processing_progress[job_id]['status'] = '背景透過処理中'
    
    processed_frames = []
    total = len(frames)
    
    # Try to import torch for CUDA cache clearing
    try:
        import torch
    except ImportError:
        torch = None
        
    import gc

    start_processing_time = time.time()
    processing_progress[job_id]['start_time'] = start_processing_time
    
    for i, frame_path in enumerate(frames):
        # Read frame
        with open(frame_path, 'rb') as f:
            input_data = f.read()
        
        # Remove background with error handling
        try:
            output_data = remove(input_data, session=session)
        except Exception as e:
            if "bad allocation" in str(e):
                print(f"[{job_id}] Memory error at frame {i+1}. Retrying...", flush=True)
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                try:
                    output_data = remove(input_data, session=session)
                except Exception as retry_e:
                    raise Exception(f"GPU memory error: {retry_e}")
            else:
                raise e
        
        # Save processed frame
        output_path = output_dir / frame_path.name
        with open(output_path, 'wb') as f:
            f.write(output_data)
        
        processed_frames.append(output_path)
        
        # System resource monitoring
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu_vram = 0
        
        # Get VRAM via nvidia-smi (works without torch)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    used_vram = float(parts[0].strip())
                    total_vram = float(parts[1].strip())
                    gpu_vram = round((used_vram / total_vram) * 100, 1)
        except:
            pass

        # Update progress and print to terminal
        processing_progress[job_id]['current'] = i + 1
        processing_progress[job_id]['resources'] = {
            'cpu': cpu,
            'ram': ram,
            'gpu_vram': gpu_vram
        }
        
        # Frame processing accounts for 98% of total progress
        frames_progress = int((i + 1) / total * 98)
        processing_progress[job_id]['progress'] = frames_progress
        
        # Check for cancellation
        if processing_progress[job_id].get('cancel'):
            print(f"[{job_id}] Processing cancelled by user.", flush=True)
            raise Exception("キャンセルされました")
        
        status_msg = f'背景透過処理中 ({i + 1}/{total})'
        processing_progress[job_id]['status'] = status_msg
        
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"[{job_id}] {status_msg}", flush=True)

        # Explicit cleanup to prevent VRAM leak
        del input_data
        del output_data
        if (i + 1) % 10 == 0:
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    
    return processed_frames


def create_webm(frames_dir, output_path, fps, job_id):
    """Create WebM video with alpha channel."""
    global processing_progress
    processing_progress[job_id]['status'] = 'WebM生成中'
    
    input_pattern = str(frames_dir / 'frame_%06d.png')
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-b:v', '2M',
        '-auto-alt-ref', '0',
        str(output_path)
    ]
    
    processing_progress[job_id]['status'] = 'WebM生成中 (実行中...)'
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise Exception(f"WebM creation failed: {process.stderr}")
    
    return output_path


def create_gif(frames_dir, output_path, fps, job_id):
    """Create animated GIF with transparency."""
    global processing_progress
    processing_progress[job_id]['status'] = 'GIF生成中'
    
    input_pattern = str(frames_dir / 'frame_%06d.png')
    palette_path = frames_dir.parent / f'{job_id}_palette.png'
    
    # Generate palette
    cmd_palette = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-vf', 'palettegen=reserve_transparent=1',
        str(palette_path)
    ]
    
    process = subprocess.run(cmd_palette, capture_output=True, text=True)
    if process.returncode != 0:
        raise Exception(f"Palette generation failed: {process.stderr}")
    
    # Create GIF using palette
    cmd_gif = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-i', str(palette_path),
        '-lavfi', 'paletteuse=alpha_threshold=128',
        '-gifflags', '-offsetting',
        str(output_path)
    ]
    
    processing_progress[job_id]['status'] = 'GIF生成中 (実行中...)'
    
    process = subprocess.run(cmd_gif, capture_output=True, text=True)
    
    # Clean up palette
    if palette_path.exists():
        palette_path.unlink()
    
    if process.returncode != 0:
        raise Exception(f"GIF creation failed: {process.stderr}")
    
    return output_path


def cleanup_job_files(job_id):
    """Clean up temporary files for a job."""
    # Clean frames
    frames_dir = app.config['FRAMES_FOLDER'] / job_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    
    # Clean output
    output_dir = app.config['OUTPUT_FOLDER'] / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Clean upload
    upload_dir = app.config['UPLOAD_FOLDER']
    for f in upload_dir.glob(f'{job_id}_*'):
        f.unlink()


@app.route('/')
def index():
    """Render main page."""
    ffmpeg_available = check_ffmpeg()
    return render_template('index.html', ffmpeg_available=ffmpeg_available, rembg_available=REMBG_AVAILABLE)


@app.route('/upload', methods=['POST'])
def upload():
    """Handle video upload and start processing."""
    if not check_ffmpeg():
        return jsonify({'error': 'FFmpegがインストールされていません。'}), 500
    
    if not REMBG_AVAILABLE:
        return jsonify({'error': 'rembgがインストールされていません。'}), 500
    
    if 'video' not in request.files:
        return jsonify({'error': '動画ファイルが選択されていません。'}), 400
    
    file = request.files['video']
    model = request.form.get('model', 'u2net_human_seg')
    
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません。'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'対応していないファイル形式です。対応形式: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    uploaded_path = app.config['UPLOAD_FOLDER'] / f'{job_id}_{filename}'
    file.save(str(uploaded_path))
    
    # Store model selection for this job
    processing_progress[job_id] = {
        'status': '待機中',
        'progress': 0,
        'model': model
    }
    
    # Create job directories
    frames_dir = app.config['FRAMES_FOLDER'] / job_id
    output_dir = app.config['OUTPUT_FOLDER'] / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return jsonify({
        'job_id': job_id,
        'filename': filename,
        'message': '処理を開始します...'
    })


def run_processing_task(job_id, video_path, frames_dir, output_dir, model_name):
    """Background task to process the video."""
    global processing_progress
    start_time = time.time() # Capture start time for total_time
    try:
        # 1. Extract frames
        processing_progress[job_id]['status'] = 'FFmpeg初期化中...'
        processing_progress[job_id]['progress'] = 1
        
        frames, fps = extract_frames(video_path, frames_dir, job_id)
        
        if not frames:
            raise Exception('フレームを抽出できませんでした。')
        
        # 2. Remove backgrounds (98% total)
        processing_progress[job_id]['status'] = 'GPU/モデル準備中...'
        processing_progress[job_id]['progress'] = 2
        
        remove_background(frames, output_dir, job_id, model_name=model_name)
        
        # 3. Create output files (remaining 2%)
        webm_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.webm'
        gif_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        
        processing_progress[job_id]['status'] = 'WebM生成中'
        processing_progress[job_id]['progress'] = 99
        create_webm(output_dir, webm_path, fps, job_id)
        
        processing_progress[job_id]['status'] = 'GIF生成中'
        processing_progress[job_id]['progress'] = 99.5
        create_gif(output_dir, gif_path, fps, job_id)
        
        processing_progress[job_id]['status'] = '完了'
        processing_progress[job_id]['progress'] = 100
        processing_progress[job_id]['total_time'] = round(time.time() - start_time, 1) # Added total processing time
        processing_progress[job_id]['webm_url'] = f'/download/{job_id}/webm'
        processing_progress[job_id]['gif_url'] = f'/download/{job_id}/gif'
        processing_progress[job_id]['zip_url'] = f'/download/{job_id}/zip'
        
    except Exception as e:
        print(f"Processing error: {e}")
        processing_progress[job_id] = {'status': 'エラー', 'progress': 0, 'error': str(e)}


@app.route('/process/<job_id>', methods=['POST'])
def process_video(job_id):
    """Start video processing in a background thread."""
    if job_id not in processing_progress:
        return jsonify({'error': '無効なジョブIDです。'}), 404
        
    model_name = processing_progress[job_id].get('model', 'u2net_human_seg')
    
    # Find uploaded file
    upload_files = list(app.config['UPLOAD_FOLDER'].glob(f'{job_id}_*'))
    if not upload_files:
        return jsonify({'error': 'アップロードされたファイルが見つかりません。'}), 404
    
    video_path = upload_files[0]
    frames_dir = app.config['FRAMES_FOLDER'] / job_id
    output_dir = app.config['OUTPUT_FOLDER'] / job_id
    
    # Ensure directories exist
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start thread
    thread = threading.Thread(
        target=run_processing_task,
        args=(job_id, video_path, frames_dir, output_dir, model_name)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': '非同期処理を開始しました。'
    })


@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Stream progress updates via SSE."""
    import json
    
    def generate():
        import time
        while True:
            if job_id in processing_progress:
                data = processing_progress[job_id]
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                if data.get('status') in ['完了', 'エラー']:
                    break
            else:
                yield f"data: {json.dumps({'status': '待機中', 'progress': 0}, ensure_ascii=False)}\n\n"
            
            time.sleep(0.5)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')



@app.route('/download/<job_id>/<file_type>')
def download(job_id, file_type):
    """Download processed files."""
    if file_type == 'webm':
        file_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.webm'
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True, download_name='transparent_video.webm')
    
    elif file_type == 'gif':
        file_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True, download_name='transparent_video.gif')
    
    elif file_type == 'zip':
        webm_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.webm'
        gif_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        
        if webm_path.exists() and gif_path.exists():
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.write(str(webm_path), 'transparent_video.webm')
                zip_file.write(str(gif_path), 'transparent_video.gif')
            
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='transparent_videos.zip'
            )
    
    return jsonify({'error': 'ファイルが見つかりません。'}), 404


@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup(job_id):
    """Clean up job files after download."""
    try:
        cleanup_job_files(job_id)
        
        # Clean result files
        webm_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.webm'
        gif_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        
        if webm_path.exists():
            webm_path.unlink()
        if gif_path.exists():
            gif_path.unlink()
        
        # Remove from progress tracking
        if job_id in processing_progress:
            del processing_progress[job_id]
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job."""
    global processing_progress
    if job_id in processing_progress:
        processing_progress[job_id]['cancel'] = True
        processing_progress[job_id]['status'] = 'キャンセル中...'
        return jsonify({'message': 'キャンセルリクエストを受理しました。'})
    return jsonify({'error': 'ジョブが見つかりません。'}), 404


if __name__ == '__main__':
    print("=" * 50)
    print("Video Background Remover")
    print("=" * 50)
    print(f"FFmpeg available: {check_ffmpeg()}")
    print(f"rembg available: {REMBG_AVAILABLE}")
    print("Starting server at http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, threaded=True)
