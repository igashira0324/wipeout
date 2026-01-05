import os
import shutil
import subprocess
import uuid
import zipfile
import threading
import time
import json
import psutil
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image
import sys

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


try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    UPSCALE_AVAILABLE = True
except ImportError:
    UPSCALE_AVAILABLE = False

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['FRAMES_FOLDER'] = Path(__file__).parent / 'frames'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'output'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'results'
app.config['HISTORY_FILE'] = Path(__file__).parent / 'history.json'

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS
ALLOWED_VIDEO_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/webm'}
ALLOWED_IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}

# Global dictionary to store processing progress and sessions
processing_progress = {}
sessions = {}
upscale_sessions = {}

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


def get_upscale_session(model_name, scale=4):
    """Get or create Real-ESRGAN upsampler."""
    global upscale_sessions
    
    session_key = (model_name, scale)
    if session_key in upscale_sessions:
        return upscale_sessions[session_key]
    
    if not UPSCALE_AVAILABLE:
        print("Upscale libraries not available.")
        return None

    print(f"Loading upscale model: {model_name} (scale={scale})")
    
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] PyTorch device: {device}")
        if device == 'cuda':
            print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
        
        # Model URLs from official Real-ESRGAN releases
        model_urls = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        }
        
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        else:
            raise ValueError(f"Unsupported upscale model: {model_name}")

        model_path = model_urls.get(model_name)
        if not model_path:
            raise ValueError(f"No model URL for: {model_name}")

        # Use half=False for better compatibility (some GPUs have issues with FP16)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=320,  # Reduced tile size for better stability on lower VRAM
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )
        
        upscale_sessions[session_key] = upsampler
        print(f"Successfully loaded upscale model: {model_name} on {device}")
        return upsampler
    except Exception as e:
        print(f"Error loading upscale model {model_name}: {e}")
        return None



@app.route('/history', methods=['GET'])
def get_history():
    """Get processing history."""
    if app.config['HISTORY_FILE'].exists():
        try:
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                history = json.load(f)
            return jsonify(history)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify([])


def log_processing_event(data):
    """Log processing event to history.json"""
    history = []
    if app.config['HISTORY_FILE'].exists():
        try:
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error reading history file: {e}")

    # Add timestamp
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepend new event
    history.insert(0, data)
    
    # Keep last 50 events
    history = history[:50]
    
    try:
        with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing history file: {e}")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename):
    """Check if file is an image."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def is_animated_gif(file_path):
    """Check if a GIF file is animated."""
    try:
        with Image.open(file_path) as img:
            return getattr(img, "is_animated", False)
    except:
        return False


def apply_background_color(img, bg_color):
    """Apply background color to transparent image.
    
    Args:
        img: PIL Image with transparency (RGBA)
        bg_color: Color string ('transparent', '#RRGGBB')
    
    Returns:
        PIL Image with background color applied
    """
    if bg_color == 'transparent' or not bg_color:
        return img
    
    # Ensure image is RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Parse hex color
    try:
        r = int(bg_color[1:3], 16)
        g = int(bg_color[3:5], 16)
        b = int(bg_color[5:7], 16)
    except (ValueError, IndexError):
        return img  # Invalid color, return as-is
    
    # Create background layer
    bg = Image.new('RGBA', img.size, (r, g, b, 255))
    
    # Composite transparent image over background
    result = Image.alpha_composite(bg, img)
    
    return result


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


def extract_frames(video_path, frames_dir, job_id, has_transparency=False):
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
    ]
    
    if has_transparency:
        cmd.extend(['-pix_fmt', 'rgba'])
        
    cmd.append(output_pattern)
    
    # Update status just before running
    processing_progress[job_id]['status'] = 'フレーム分解中 (実行中...)'
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg error: {process.stderr}")
    
    # Count extracted frames
    frames = sorted(frames_dir.glob('frame_*.png'))
    processing_progress[job_id]['total'] = len(frames)
    
    return frames, fps


def remove_background(frames, output_dir, job_id, model_name="u2net_human_seg", bg_color="transparent"):
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
        
        # Apply background color if specified
        output_path = output_dir / frame_path.name
        if bg_color and bg_color != 'transparent':
            # Open as PIL Image, apply bg color, save
            output_img = Image.open(BytesIO(output_data))
            output_img = apply_background_color(output_img, bg_color)
            output_img.save(str(output_path), 'PNG')
        else:
            # Save as-is (transparent)
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


def create_mp4(frames_dir, output_path, fps, job_id):
    """Create MP4 video (used when background is not transparent)."""
    global processing_progress
    processing_progress[job_id]['status'] = 'MP4生成中'
    
    # Check for JPEG or PNG frames
    if list(frames_dir.glob('frame_*.jpg')):
        input_pattern = str(frames_dir / 'frame_%06d.jpg')
    else:
        input_pattern = str(frames_dir / 'frame_%06d.png')
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        str(output_path)
    ]
    
    processing_progress[job_id]['status'] = 'MP4生成中 (実行中...)'
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        raise Exception(f"MP4 creation failed: {process.stderr}")
    
    return output_path


def create_gif(frames_dir, output_path, fps, job_id):
    """Create animated GIF."""
    global processing_progress
    processing_progress[job_id]['status'] = 'GIF生成中'
    
    # Check for JPEG or PNG frames
    if list(frames_dir.glob('frame_*.jpg')):
        input_pattern = str(frames_dir / 'frame_%06d.jpg')
    else:
        input_pattern = str(frames_dir / 'frame_%06d.png')
    
    palette_path = frames_dir.parent / f'{job_id}_palette.png'
    
    # Generate palette
    cmd_palette = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-vf', 'palettegen',
        str(palette_path)
    ]
    
    print(f"[DEBUG] GIF palette command: {' '.join(cmd_palette)}")
    process = subprocess.run(cmd_palette, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"[ERROR] Palette generation failed: {process.stderr}")
        raise Exception(f"Palette generation failed: {process.stderr}")
    
    # Create GIF using palette
    cmd_gif = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-i', str(palette_path),
        '-lavfi', 'paletteuse',
        str(output_path)
    ]
    
    print(f"[DEBUG] GIF creation command: {' '.join(cmd_gif)}")
    processing_progress[job_id]['status'] = 'GIF生成中 (実行中...)'
    
    process = subprocess.run(cmd_gif, capture_output=True, text=True)
    
    # Clean up palette
    if palette_path.exists():
        palette_path.unlink()
    
    if process.returncode != 0:
        print(f"[ERROR] GIF creation failed: {process.stderr}")
        raise Exception(f"GIF creation failed: {process.stderr}")
    
    print(f"[DEBUG] GIF created successfully: {output_path}")
    return output_path


def create_gif_transparent(frames_dir, output_path, fps, job_id):
    """Create animated GIF with transparency preserved."""
    global processing_progress
    processing_progress[job_id]['status'] = 'GIF生成中 (透過)'
    
    # Use PNG frames for transparency
    input_pattern = str(frames_dir / 'frame_%06d.png')
    palette_path = frames_dir.parent / f'{job_id}_palette.png'
    
    # Generate palette with transparency
    cmd_palette = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-vf', 'palettegen=reserve_transparent=1',
        str(palette_path)
    ]
    
    print(f"[DEBUG] Transparent GIF palette command: {' '.join(cmd_palette)}")
    process = subprocess.run(cmd_palette, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"[ERROR] Palette generation failed: {process.stderr}")
        raise Exception(f"Palette generation failed: {process.stderr}")
    
    # Create GIF using palette with transparency
    cmd_gif = [
        'ffmpeg', '-y',
        '-framerate', str(min(fps, 30)),
        '-i', input_pattern,
        '-i', str(palette_path),
        '-lavfi', 'paletteuse=alpha_threshold=128',
        '-gifflags', '-offsetting',
        str(output_path)
    ]
    
    print(f"[DEBUG] Transparent GIF creation command: {' '.join(cmd_gif)}")
    processing_progress[job_id]['status'] = 'GIF生成中 (透過・実行中...)'
    
    process = subprocess.run(cmd_gif, capture_output=True, text=True)
    
    # Clean up palette
    if palette_path.exists():
        palette_path.unlink()
    
    if process.returncode != 0:
        print(f"[ERROR] Transparent GIF creation failed: {process.stderr}")
        raise Exception(f"Transparent GIF creation failed: {process.stderr}")
    
    print(f"[DEBUG] Transparent GIF created successfully: {output_path}")
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
    """Handle file upload and process (image immediately, video async)."""
    if not REMBG_AVAILABLE:
        return jsonify({'error': 'rembgがインストールされていません。'}), 500
    
    # Check for file in request (changed from 'video' to 'file')
    if 'file' not in request.files:
        # Fallback to check 'video' for backward compatibility
        if 'video' not in request.files:
            return jsonify({'error': 'ファイルが選択されていません。'}), 400
        file = request.files['video']
    else:
        file = request.files['file']
    
    model = request.form.get('model', 'u2net_human_seg')
    bg_color = request.form.get('bg_color', 'transparent')
    file_type = request.form.get('file_type', 'video')
    
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません。'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'対応していないファイル形式です。対応形式: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Check if it's an image file
    if is_image_file(file.filename):
        # Process image immediately (synchronous)
        start_time = time.time()
        
        try:
            # Read image
            input_data = file.read()
            
            # Get session
            session, device = get_session(model)
            
            # Remove background
            output_data = remove(input_data, session=session)
            
            # Open as PIL Image
            output_img = Image.open(BytesIO(output_data))
            
            # Apply background color if specified
            output_img = apply_background_color(output_img, bg_color)
            
            # Determine image output format and extension
            is_transparent = bg_color == 'transparent'
            
            # Original input extension
            input_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'png'
            if input_ext == 'jpeg': input_ext = 'jpg'
            
            if is_transparent:
                output_ext = 'webp'
                output_mime = 'image/webp'
                img_format = 'WEBP'
            else:
                output_ext = input_ext if input_ext in ['png', 'jpg'] else 'png'
                output_mime = f'image/{output_ext}'
                img_format = 'PNG' if output_ext == 'png' else 'JPEG'
            
            output_filename = f'{job_id}_output.{output_ext}'
            output_path = app.config['RESULTS_FOLDER'] / output_filename
            
            # Convert to RGB if saving as JPEG
            if img_format == 'JPEG' and output_img.mode == 'RGBA':
                output_img = output_img.convert('RGB')
            
            output_img.save(str(output_path), img_format)
            
            processing_time = round(time.time() - start_time, 2)
            
            # Log to history
            input_size = len(input_data)
            output_size = output_path.stat().st_size
            history_data = {
                'type': 'Image',
                'filename': file.filename,
                'model': model,
                'bg_color': bg_color,
                'resolution': f"{output_img.width}x{output_img.height}",
                'input_size': f"{input_size / 1024:.1f} KB",
                'output_size': f"{output_size / 1024:.1f} KB",
                'processing_time': f"{processing_time}s"
            }
            log_processing_event(history_data)

            # Store job info for cleanup
            processing_progress[job_id] = {
                'status': '完了',
                'type': 'image',
                'output_path': str(output_path)
            }
            
            return jsonify({
                'job_id': job_id,
                'image_url': f'/download/{job_id}/image',
                'metadata': {
                    'width': output_img.width,
                    'height': output_img.height
                },
                'processing_time': processing_time,
                'device': device
            })
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({'error': f'画像処理エラー: {str(e)}'}), 500
    
    else:
        # Video processing (async) - requires FFmpeg
        if not check_ffmpeg():
            return jsonify({'error': 'FFmpegがインストールされていません。動画処理にはFFmpegが必要です。'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        uploaded_path = app.config['UPLOAD_FOLDER'] / f'{job_id}_{filename}'
        file.save(str(uploaded_path))
        
        # Store model and bg_color selection for this job
        processing_progress[job_id] = {
            'status': '待機中',
            'progress': 0,
            'model': model,
            'bg_color': bg_color,
            'type': 'video'
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


def run_processing_task(job_id, video_path, frames_dir, output_dir, model_name, bg_color="transparent"):
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
        
        remove_background(frames, output_dir, job_id, model_name=model_name, bg_color=bg_color)
        
        # 3. Create output files (remaining 2%)
        is_transparent = bg_color == 'transparent'
        video_ext = 'webm' if is_transparent else 'mp4'
        video_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.{video_ext}'
        gif_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        
        if is_transparent:
            processing_progress[job_id]['status'] = 'WebM生成中'
            processing_progress[job_id]['progress'] = 99
            create_webm(output_dir, video_path, fps, job_id)
        else:
            processing_progress[job_id]['status'] = 'MP4生成中'
            processing_progress[job_id]['progress'] = 99
            create_mp4(output_dir, video_path, fps, job_id)
        
        processing_progress[job_id]['status'] = 'GIF生成中'
        processing_progress[job_id]['progress'] = 99.5
        create_gif(output_dir, gif_path, fps, job_id)
        
        processing_progress[job_id]['status'] = '完了'
        processing_progress[job_id]['progress'] = 100
        processing_progress[job_id]['total_time'] = round(time.time() - start_time, 1)
        
        # Log to history
        duration = processing_progress[job_id].get('metadata', {}).get('duration', '-')
        width = processing_progress[job_id].get('metadata', {}).get('width', '-')
        height = processing_progress[job_id].get('metadata', {}).get('height', '-')
        total_time = round(time.time() - start_time, 2)

        # Get input file size (original uploaded video)
        upload_files = list(app.config['UPLOAD_FOLDER'].glob(f'{job_id}_*'))
        input_size = upload_files[0].stat().st_size if upload_files else 0
        output_size = video_path.stat().st_size if video_path.exists() else 0

        history_data = {
            'type': 'Video',
            'filename': str(video_path.name).replace(f'{job_id}_', ''), # Original filename
            'model': model_name,
            'bg_color': bg_color,
            'resolution': f"{width}x{height}",
            'input_size': f"{input_size / (1024*1024):.1f} MB",
            'output_size': f"{output_size / (1024*1024):.1f} MB",
            'processing_time': f"{int(total_time // 60)}m {int(total_time % 60)}s"
        }
        log_processing_event(history_data)
        
        # Set dynamic video URL based on ext
        processing_progress[job_id]['video_url'] = f'/download/{job_id}/{video_ext}'
        processing_progress[job_id]['gif_url'] = f'/download/{job_id}/gif'
        processing_progress[job_id]['zip_url'] = f'/download/{job_id}/zip'
        processing_progress[job_id]['video_ext'] = video_ext
        
    except Exception as e:
        print(f"Processing error: {e}")
        processing_progress[job_id] = {'status': 'エラー', 'progress': 0, 'error': str(e)}


@app.route('/process/<job_id>', methods=['POST'])
def process_video(job_id):
    """Start video processing in a background thread."""
    if job_id not in processing_progress:
        return jsonify({'error': '無効なジョブIDです。'}), 404
        
    model_name = processing_progress[job_id].get('model', 'u2net_human_seg')
    bg_color = processing_progress[job_id].get('bg_color', 'transparent')
    
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
        args=(job_id, video_path, frames_dir, output_dir, model_name, bg_color)
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
    print(f"[DEBUG] SSE connection opened for job: {job_id}")
    def generate():
        count = 0
        while True:
            try:
                if job_id in processing_progress:
                    # Use a shallow copy to prevent concurrent modification issues
                    data = processing_progress[job_id].copy()
                    
                    # Ensure metadata values are serializable (convert numpy types)
                    if 'metadata' in data and data['metadata']:
                        meta = data['metadata'].copy()
                        for k, v in meta.items():
                            if hasattr(v, 'item'): # Convert numpy scalar to python type
                                meta[k] = v.item()
                            elif isinstance(v, (np.int32, np.int64)):
                                meta[k] = int(v)
                        data['metadata'] = meta
                    
                    json_data = json.dumps(data, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                    
                    if count % 5 == 0:
                        print(f"[DEBUG] Sent progress for {job_id}: {data.get('progress')}% - {data.get('status')}")
                        sys.stdout.flush()
                    count += 1
                    
                    if data.get('status') in ['完了', 'エラー']:
                        print(f"[DEBUG] Job {job_id} finished with status: {data.get('status')}")
                        break
                else:
                    yield f"data: {json.dumps({'status': '待機中', 'progress': 0}, ensure_ascii=False)}\n\n"
            except Exception as e:
                # Log but don't break the stream
                print(f"[SSE Error] {e}")
                sys.stdout.flush()
            
            time.sleep(0.5)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')



@app.route('/download/<job_id>/<file_type>')
def download(job_id, file_type):
    """Download processed files."""
    if file_type == 'webm':
        file_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.webm'
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True, download_name='transparent_video.webm')
    
    elif file_type == 'mp4':
        file_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.mp4'
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True, download_name='background_video.mp4')
    
    elif file_type == 'gif':
        file_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True, download_name='processed_video.gif')
    
    elif file_type == 'image':
        # Find which image file exists (png, jpg, webp) - check both _output and _upscaled patterns
        output_results = list(app.config['RESULTS_FOLDER'].glob(f'{job_id}_output.*'))
        upscale_results = list(app.config['RESULTS_FOLDER'].glob(f'{job_id}_upscaled.*'))
        
        all_results = output_results + upscale_results
        image_results = [r for r in all_results if r.suffix in ['.png', '.jpg', '.webp']]
        
        if image_results:
            file_path = image_results[0]
            return send_file(str(file_path), as_attachment=True, download_name=f'processed_image{file_path.suffix}')
    
    elif file_type == 'zip':
        # Result files search
        results = list(app.config['RESULTS_FOLDER'].glob(f'{job_id}_output.*'))
        
        if results:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for res_path in results:
                    # Map to friendly names
                    friendly_name = res_path.name.replace(f'{job_id}_output', 'processed_result')
                    if res_path.suffix == '.webm': friendly_name = 'transparent_video.webm'
                    elif res_path.suffix == '.mp4': friendly_name = 'background_video.mp4'
                    elif res_path.suffix == '.gif': friendly_name = 'processed_animation.gif'
                    elif res_path.suffix == '.webp': friendly_name = 'transparent_image.webp'
                    elif res_path.suffix == '.png': friendly_name = 'processed_image.png'
                    elif res_path.suffix == '.jpg': friendly_name = 'processed_image.jpg'
                    
                    zip_file.write(str(res_path), friendly_name)
            
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='processed_files.zip'
            )
    
    return jsonify({'error': 'ファイルが見つかりません。'}), 404


@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup(job_id):
    """Cleanup job data (called when user switches tabs or starts new process)."""
    global processing_progress
    
    # Remove from progress tracking
    if job_id in processing_progress:
        del processing_progress[job_id]
    
    # Optional: Clean up temporary files (disabled for now to keep results downloadable)
    # Uncommenting would delete results immediately after user leaves result page
    # try:
    #     frames_dir = app.config['FRAMES_FOLDER'] / job_id
    #     output_dir = app.config['OUTPUT_FOLDER'] / job_id
    #     if frames_dir.exists():
    #         shutil.rmtree(str(frames_dir))
    #     if output_dir.exists():
    #         shutil.rmtree(str(output_dir))
    # except Exception as e:
    #     print(f"[DEBUG] Cleanup error: {e}")
    
    return jsonify({'success': True})


@app.route('/upscale', methods=['POST'])
def upscale():
    """Handle image/video upscale request."""
    if not UPSCALE_AVAILABLE:
        return jsonify({'error': '高画質化ライブラリ (realesrgan, basicsr) がインストールされていません。'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません。'}), 400
    
    file = request.files['file']
    model_name = request.form.get('upscaleModel', 'RealESRGAN_x4plus')
    scale = int(request.form.get('upscaleRatio', '4'))
    face_enhance = request.form.get('faceEnhance') == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません。'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'対応していないファイル形式です。対応形式: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    job_id = str(uuid.uuid4())[:10]
    
    # Detect if video/animated-gif
    is_gif = file.filename.lower().endswith('.gif')
    is_animated = False
    has_transparency = False
    
    # Save file first
    uploaded_path = app.config['UPLOAD_FOLDER'] / f'{job_id}_{secure_filename(file.filename)}'
    file.save(str(uploaded_path))
    
    if is_gif:
        is_animated = is_animated_gif(uploaded_path)
        # Check for transparency in GIF
        try:
            with Image.open(uploaded_path) as img:
                # Method 1: Check for transparency index in palette mode
                if 'transparency' in img.info:
                    has_transparency = True
                    print(f"[DEBUG] Transparency detected via palette index: {img.info.get('transparency')}")
                
                # Method 2: Check if mode is RGBA (direct alpha channel)
                elif img.mode == 'RGBA':
                    has_transparency = True
                    print(f"[DEBUG] Transparency detected via RGBA mode")
                
                # Method 3: Check if palette mode has alpha
                elif img.mode == 'P':
                    # Palette mode GIF - check if any color in palette has transparency
                    palette = img.getpalette()
                    if palette and img.info.get('transparency') is not None:
                        has_transparency = True
                        print(f"[DEBUG] Transparency detected in P mode with transparency index")
                
                # Method 4: Convert to RGBA and check for alpha values < 255
                if not has_transparency:
                    try:
                        rgba_img = img.convert('RGBA')
                        # Check a few pixels at corners and center for transparency
                        pixels = [
                            rgba_img.getpixel((0, 0)),
                            rgba_img.getpixel((rgba_img.width - 1, 0)),
                            rgba_img.getpixel((0, rgba_img.height - 1)),
                            rgba_img.getpixel((rgba_img.width // 2, rgba_img.height // 2))
                        ]
                        for px in pixels:
                            if len(px) == 4 and px[3] < 255:
                                has_transparency = True
                                print(f"[DEBUG] Transparency detected via pixel check (alpha < 255)")
                                break
                    except Exception as e2:
                        print(f"[DEBUG] RGBA conversion check failed: {e2}")
                
                print(f"[DEBUG] Final transparency detection result: {has_transparency} (mode: {img.mode})")
        except Exception as e:
            print(f"[DEBUG] Transparency detection error: {e}")
    
    is_video = is_animated or not is_image_file(file.filename)
    
    # Common progress initialization with metadata
    processing_progress[job_id] = {
        'status': '待機中',
        'progress': 0,
        'current': 0,
        'total': 0,
        'start_time': time.time(),
        'type': 'upscale',
        'is_video': is_video,
        'filename': file.filename,
        'metadata': {'width': 0, 'height': 0, 'duration': 0, 'fps': 0},
        'has_transparency': has_transparency
    }
    
    # Start background thread
    frames_dir = app.config['FRAMES_FOLDER'] / job_id
    output_dir = app.config['OUTPUT_FOLDER'] / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thread = threading.Thread(
        target=run_upscale_task, 
        args=(job_id, uploaded_path, frames_dir, output_dir, model_name, scale, face_enhance, is_video, has_transparency)
    )
    thread.start()
    
    return jsonify({'job_id': job_id})


def run_upscale_task(job_id, input_path, frames_dir, output_dir, model_name, scale, face_enhance, is_video, has_transparency=False):
    """Background task to upscale image/video frames."""
    global processing_progress
    start_time = time.time()
    try:
        import cv2
        # 1. Prepare frames
        if is_video:
            processing_progress[job_id]['status'] = 'フレーム分解中...'
            processing_progress[job_id]['progress'] = 5
            frames, fps = extract_frames(input_path, frames_dir, job_id, has_transparency)
            if not frames:
                raise Exception('フレームを抽出できませんでした。')
        else:
            processing_progress[job_id]['status'] = '画像読み込み中...'
            processing_progress[job_id]['progress'] = 5
            frames = [input_path]
            fps = 1
            # Add metadata for image if not present
            img_cv = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                h, w = img_cv.shape[:2]
                processing_progress[job_id]['metadata'] = {'width': w, 'height': h}
            
        # 2. Upscale frames
        processing_progress[job_id]['status'] = 'モデルを起動中...'
        processing_progress[job_id]['progress'] = 10
        upsampler = get_upscale_session(model_name, scale)
        if not upsampler:
            raise Exception("高画質化モデルのロードに失敗しました。")
            
        total = len(frames)
        for i, frame_path in enumerate(frames):
            if processing_progress.get(job_id, {}).get('cancel'):
                break
                
            progress = 10 + (i / total * 85)
            processing_progress[job_id]['progress'] = round(progress, 1)
            processing_progress[job_id]['status'] = f'高画質化中 ({i+1}/{total}) - 処理開始'
            processing_progress[job_id]['current'] = i + 1
            processing_progress[job_id]['total'] = total
            
            # Flush stdout to ensure logs appear immediately
            if i % 5 == 0:
                print(f"[DEBUG] Processing frame {i+1}/{total}")
                sys.stdout.flush()
            
            img_cv = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if img_cv is None: continue
            
            # Upscale
            processing_progress[job_id]['status'] = f'高画質化中 ({i+1}/{total}) - モデル処理中...'
            output, _ = upsampler.enhance(img_cv, outscale=scale)
            
            processing_progress[job_id]['status'] = f'高画質化中 ({i+1}/{total}) - 保存中...'
            if is_video:
                # Use PNG for transparent files to preserve alpha, JPEG for others to save space
                if has_transparency:
                    out_path = output_dir / f'frame_{i:06d}.png'
                    cv2.imwrite(str(out_path), output)
                else:
                    out_path = output_dir / f'frame_{i:06d}.jpg'
                    cv2.imwrite(str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                # Single image result
                input_ext = input_path.suffix.lower()
                if input_ext not in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
                    input_ext = '.png'
                
                output_filename = f'{job_id}_upscaled{input_ext}'
                output_path = app.config['RESULTS_FOLDER'] / output_filename
                
                if input_ext == '.webp':
                    cv2.imwrite(str(output_path), output, [cv2.IMWRITE_WEBP_QUALITY, 95])
                elif input_ext in ['.jpg', '.jpeg']:
                    cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif input_ext == '.gif':
                    if output.shape[2] == 4:
                        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA))
                    else:
                        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    output_pil.save(str(output_path))
                else:
                    cv2.imwrite(str(output_path), output)
            
            # Keep updating resolution metadata (ensure they are Python ints)
            if i == 0 or 'width' not in processing_progress[job_id]['metadata'] or processing_progress[job_id]['metadata']['width'] == 0:
                h, w = output.shape[:2]
                processing_progress[job_id]['metadata']['width'] = int(w)
                processing_progress[job_id]['metadata']['height'] = int(h)

        # 3. Create output
        if is_video:
            processing_progress[job_id]['status'] = 'データ出力中...'
            processing_progress[job_id]['progress'] = 97
            
            is_gif_input = input_path.suffix.lower() == '.gif'
            if is_gif_input:
                output_video_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.gif'
                if has_transparency:
                    create_gif_transparent(output_dir, output_video_path, fps, job_id)
                else:
                    create_gif(output_dir, output_video_path, fps, job_id)
            else:
                output_video_path = app.config['RESULTS_FOLDER'] / f'{job_id}_output.mp4'
                create_mp4(output_dir, output_video_path, fps, job_id)
            
            output_final_path = output_video_path
        else:
            output_final_path = output_path

        # Ensure metadata exists
        if 'metadata' not in processing_progress[job_id]:
            processing_progress[job_id]['metadata'] = {}

        processing_progress[job_id]['status'] = '完了'
        processing_progress[job_id]['progress'] = 100
        total_time = round(time.time() - start_time, 2)
        processing_progress[job_id]['total_time'] = total_time
        
        # History
        metadata = processing_progress[job_id].get('metadata', {})
        input_size = input_path.stat().st_size
        output_size = output_final_path.stat().st_size if output_final_path.exists() else 0
        
        history_data = {
            'type': 'Video Upscale' if is_video else 'Image Upscale',
            'filename': processing_progress[job_id].get('filename', input_path.name).replace(f'{job_id}_', ''),
            'model': model_name,
            'bg_color': 'N/A',
            'resolution': f"{metadata.get('width')}x{metadata.get('height')}",
            'input_size': f"{input_size / (1024*1024):.1f} MB" if is_video else f"{input_size / 1024:.1f} KB",
            'output_size': f"{output_size / (1024*1024):.1f} MB" if is_video else f"{output_size / 1024:.1f} KB",
            'processing_time': f"{int(total_time // 60)}m {int(total_time % 60)}s" if is_video else f"{total_time}s"
        }
        log_processing_event(history_data)
        
        if is_video:
            if is_gif_input:
                processing_progress[job_id]['video_url'] = f'/download/{job_id}/gif'
                processing_progress[job_id]['video_ext'] = 'gif'
            else:
                processing_progress[job_id]['video_url'] = f'/download/{job_id}/mp4'
                processing_progress[job_id]['video_ext'] = 'mp4'
        else:
            processing_progress[job_id]['image_url'] = f'/download/{job_id}/image'
            processing_progress[job_id]['processing_time'] = total_time
        
    except Exception as e:
        print(f"Video upscale error: {e}")
        processing_progress[job_id] = {'status': 'エラー', 'progress': 0, 'error': str(e)}
def cleanup(job_id):
    """Clean up job files after download."""
    try:
        cleanup_job_files(job_id)
        
        # Clean result files
        for res_path in app.config['RESULTS_FOLDER'].glob(f'{job_id}_output.*'):
            try:
                res_path.unlink()
            except Exception as e:
                print(f"Cleanup error for {res_path}: {e}")
        
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
