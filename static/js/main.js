/**
 * Video Background Remover - Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const selectedFile = document.getElementById('selectedFile');
    const fileName = document.getElementById('fileName');
    const clearFile = document.getElementById('clearFile');
    const startProcess = document.getElementById('startProcess');

    const uploadSection = document.getElementById('uploadSection');
    const progressSection = document.getElementById('progressSection');
    const resultSection = document.getElementById('resultSection');
    const errorSection = document.getElementById('errorSection');

    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const frameInfo = document.getElementById('frameInfo');
    const statusBadge = document.getElementById('statusBadge');

    const downloadWebm = document.getElementById('downloadWebm');
    const downloadGif = document.getElementById('downloadGif');
    const downloadZip = document.getElementById('downloadZip');
    const processAnother = document.getElementById('processAnother');
    const cancelProcess = document.getElementById('cancelProcess');
    const retryBtn = document.getElementById('retryBtn');
    const errorMessage = document.getElementById('errorMessage');

    // Model selection
    const modelCards = document.querySelectorAll('.model-card');
    const modelInputs = document.querySelectorAll('input[name="model"]');

    // Processing steps
    const steps = {
        'フレーム分解中': 'step1',
        '画像読み込み中': 'step1',
        'モデルを起動中': 'step2',
        'AIモデルをロード中': 'step2',
        '背景透過処理中': 'step2',
        '高画質化中': 'step2',
        'WebM生成中': 'step3',
        'MP4生成中': 'step3',
        'データ出力中': 'step3',
        'GIF生成中': 'step4'
    };

    let currentFile = null;
    let currentJobId = null;
    let eventSource = null;
    let selectedModel = 'u2net_human_seg';
    let selectedBgColor = 'transparent';
    let customColor = '#00FF00'; // Default to green for chroma key
    let isImageFile = false;

    // Background color options
    const bgOptions = document.querySelectorAll('.bg-option');
    const bgColorInputs = document.querySelectorAll('input[name="bgColor"]');
    const customColorPicker = document.getElementById('customColorPicker');
    const colorPicker = document.getElementById('colorPicker');
    const hexInput = document.getElementById('hexInput');
    const customColorPreview = document.getElementById('customColorPreview');

    // Mode Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const modePanels = document.querySelectorAll('.mode-panel');
    let currentMode = 'transparent'; // 'transparent' or 'upscale'

    // Upscale settings
    let selectedUpscaleModel = 'RealESRGAN_x4plus';
    let selectedUpscaleScale = '1';
    const faceEnhance = document.getElementById('faceEnhance');
    const scaleItems = document.querySelectorAll('.scale-item');

    // For stable ETA
    let progressHistory = [];
    const HISTORY_LIMIT = 20;
    let lastEta = null;

    // Model selection logic
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            const radio = card.querySelector('input[name="model"]');
            if (!radio) return; // Not a model selection card for BG removal

            modelCards.forEach(c => {
                const r = c.querySelector('input[name="model"]');
                if (r) c.classList.remove('active');
            });
            card.classList.add('active');
            radio.checked = true;
            selectedModel = radio.value;
        });
    });

    // Upscale Model selection logic
    const upscaleCards = document.querySelectorAll('.mode-panel#upscaleSettings .model-card');
    upscaleCards.forEach(card => {
        card.addEventListener('click', () => {
            upscaleCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[name="upscaleModel"]');
            radio.checked = true;
            selectedUpscaleModel = radio.value;
        });
    });

    // Scale selection logic
    scaleItems.forEach(item => {
        item.addEventListener('click', () => {
            scaleItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            const radio = item.querySelector('input[type="radio"]');
            radio.checked = true;
            selectedUpscaleScale = radio.value;
        });
    });

    // Tab switching logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            // Check if we are in result screen
            if (resultSection.style.display === 'block') {
                await resetToUploadState();
            } else if (progressSection.style.display === 'block') {
                // If processing, do not allow tab switch or confirm cancel
                // For now, just return to prevent state corruption
                return;
            }

            currentMode = btn.dataset.mode;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Synchronize visibility of settings panels
            showSection('upload');

            // Update UI strings
            const startBtnText = document.getElementById('startBtnText');
            if (currentMode === 'transparent') {
                startBtnText.textContent = '背景透過処理を開始';
                startChromaAnimation();
            } else {
                startBtnText.textContent = '高画質化を開始';
                stopChromaAnimation();
            }
            updateStepLabels();
        });
    });

    function updateStepLabels() {
        const step1 = document.getElementById('step1Text');
        const step2 = document.getElementById('step2Text');
        const step3 = document.getElementById('step3Text');
        const step4 = document.getElementById('step4Text');

        if (currentMode === 'transparent') {
            step1.textContent = 'フレーム分解';
            step2.textContent = '背景透過処理';
            step3.textContent = '動画生成';
            step4.textContent = 'GIF生成';
        } else {
            step1.textContent = 'フレーム分解';
            step2.textContent = '高画質化中';
            step3.textContent = '動画エンコード';
            step4.textContent = '最終処理';
        }
    }

    function disableUpscaleSelection(disabled) {
        const inputs = document.querySelectorAll('#upscaleSettings input');
        inputs.forEach(input => input.disabled = disabled);
        const cards = document.querySelectorAll('#upscaleSettings .model-card, #upscaleSettings .scale-item');
        cards.forEach(card => {
            if (disabled) card.classList.add('disabled');
            else card.classList.remove('disabled');
        });
    }

    // Chroma Key Animation
    let chromaAnimationInterval = null;
    const chromaColors = ['#00FF00', '#0000FF', '#FF00FF']; // Green, Blue, Magenta

    function startChromaAnimation() {
        if (chromaAnimationInterval) return;

        let colorIndex = 0;

        // Only start if not currently in custom mode
        if (selectedBgColor === 'custom' || (typeof selectedBgColor === 'string' && selectedBgColor.startsWith('#') && selectedBgColor !== '#000000' && selectedBgColor !== '#FFFFFF')) {
            return;
        }

        chromaAnimationInterval = setInterval(() => {
            colorIndex = (colorIndex + 1) % chromaColors.length;
            const nextColor = chromaColors[colorIndex];
            customColorPreview.style.transition = 'background-color 1.5s ease';
            customColorPreview.style.backgroundColor = nextColor;
        }, 2500);
    }

    function stopChromaAnimation(captureCurrentColor = false) {
        if (chromaAnimationInterval) {
            clearInterval(chromaAnimationInterval);
            chromaAnimationInterval = null;
        }

        customColorPreview.style.transition = 'none';

        if (captureCurrentColor) {
            // Capture the current computed background color
            const computedStyle = window.getComputedStyle(customColorPreview);
            const rgbColor = computedStyle.backgroundColor;

            // Convert RGB to Hex
            const rgb = rgbColor.match(/\d+/g);
            if (rgb) {
                const hex = '#' + ((1 << 24) + (parseInt(rgb[0]) << 16) + (parseInt(rgb[1]) << 8) + parseInt(rgb[2])).toString(16).slice(1).toUpperCase();
                customColor = hex;
                if (colorPicker) colorPicker.value = hex;
                if (hexInput) hexInput.value = hex;
            }
        }

        customColorPreview.style.backgroundColor = customColor;
    }

    // Start animation initially
    startChromaAnimation();

    // Background color selection logic
    bgOptions.forEach(option => {
        option.addEventListener('click', () => {
            bgOptions.forEach(o => o.classList.remove('active'));
            option.classList.add('active');
            const radio = option.querySelector('input[type="radio"]');
            radio.checked = true;

            if (radio.value === 'custom') {
                customColorPicker.style.display = 'flex';
                // If coming from animation (not already custom), capture current color
                if (selectedBgColor !== 'custom' && selectedBgColor !== customColor) {
                    stopChromaAnimation(true);
                } else {
                    stopChromaAnimation(false);
                }
                selectedBgColor = customColor;
            } else {
                customColorPicker.style.display = 'none';
                selectedBgColor = radio.value;
                startChromaAnimation();
            }
        });
    });

    // Color picker sync
    if (colorPicker) {
        colorPicker.addEventListener('input', (e) => {
            customColor = e.target.value.toUpperCase();
            hexInput.value = customColor;
            customColorPreview.style.backgroundColor = customColor;
            selectedBgColor = customColor;
        });
    }

    if (hexInput) {
        hexInput.addEventListener('input', (e) => {
            let val = e.target.value.toUpperCase();
            if (!val.startsWith('#')) val = '#' + val;
            if (/^#[0-9A-F]{6}$/i.test(val)) {
                customColor = val;
                colorPicker.value = val;
                customColorPreview.style.backgroundColor = val;
                selectedBgColor = val;
            }
        });
    }

    // Drag and drop handlers
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // File handling
    function handleFile(file) {
        const videoTypes = ['video/mp4', 'video/quicktime', 'video/webm'];
        const videoExtensions = ['mp4', 'mov', 'webm'];
        const imageTypes = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
        const imageExtensions = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
        const maxSize = 100 * 1024 * 1024; // 100MB

        const ext = file.name.split('.').pop().toLowerCase();

        const isVideo = videoTypes.includes(file.type) || videoExtensions.includes(ext);
        const isImage = imageTypes.includes(file.type) || imageExtensions.includes(ext);

        if (!isVideo && !isImage) {
            showError('対応していないファイル形式です。動画: MP4, MOV, WebM / 画像: PNG, JPG, WebP');
            return;
        }

        if (file.size > maxSize) {
            showError('ファイルサイズが大きすぎます。100MB以下のファイルを選択してください。');
            return;
        }

        currentFile = file;
        isImageFile = isImage;
        fileName.textContent = file.name;

        // Update file icon based on file type
        const fileIcon = document.querySelector('.file-icon');
        if (fileIcon) {
            fileIcon.textContent = isImage ? '🖼️' : '🎥';
        }

        selectedFile.style.display = 'flex';
        startProcess.disabled = false;
    }

    // Clear file
    clearFile.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        selectedFile.style.display = 'none';
        startProcess.disabled = true;
    });

    // Start processing
    startProcess.addEventListener('click', async () => {
        if (!currentFile) return;

        try {
            // Disable UI
            disableModelSelection(true);
            disableBgColorSelection(true);
            disableUpscaleSelection(true);
            tabBtns.forEach(btn => btn.disabled = true);

            const formData = new FormData();
            formData.append('file', currentFile);

            if (currentMode === 'transparent') {
                formData.append('model', selectedModel);
                formData.append('bg_color', selectedBgColor);
                formData.append('file_type', isImageFile ? 'image' : 'video');
            } else {
                formData.append('upscaleModel', selectedUpscaleModel);
                formData.append('upscaleRatio', selectedUpscaleScale);
                formData.append('faceEnhance', faceEnhance.checked);
            }

            const endpoint = currentMode === 'transparent' ? '/upload' : '/upscale';

            showSection('progress');
            resetProgress();
            updateStepLabels();

            // Determine whether to use synchronous or async processing
            // - Transparent mode for images: synchronous (quick single-frame process)
            // - Upscale mode OR video: always async with SSE (for progress tracking)
            const useAsyncProcessing = !isImageFile || currentMode === 'upscale';

            if (!useAsyncProcessing) {
                // Image processing (transparent mode only) - Unified UI
                // Set initial image steps to match video UI look
                document.getElementById('step1').className = 'step-item completed'; // No decomposition
                document.getElementById('step2').className = 'step-item active';
                statusBadge.textContent = '背景透過処理中';
                progressBar.style.width = '30%';
                progressText.textContent = '処理を開始しています...';

                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'アップロードに失敗しました。');
                }

                // Complete steps visually
                progressBar.style.width = '100%';
                document.getElementById('step2').className = 'step-item completed';
                document.getElementById('step3').className = 'step-item completed';
                document.getElementById('step4').className = 'step-item completed';

                // Handle result
                if (data.image_url) {
                    currentJobId = data.job_id;
                    const videoBtn = document.getElementById('downloadWebm');
                    const gifBtn = document.getElementById('downloadGif');
                    const zipBtn = document.getElementById('downloadZip');

                    videoBtn.href = data.image_url;

                    // Hide GIF/ZIP for images as requested
                    gifBtn.style.display = 'none';
                    zipBtn.style.display = 'none';

                    const isTransparent = selectedBgColor === 'transparent';
                    if (isTransparent) {
                        videoBtn.innerHTML = '<span class="download-icon">🖼️</span><span class="download-text"><strong>WebP</strong><small>透過画像</small></span>';
                    } else {
                        videoBtn.innerHTML = '<span class="download-icon">🖼️</span><span class="download-text"><strong>Image</strong><small>背景付き画像</small></span>';
                    }

                    if (data.metadata) {
                        document.getElementById('summaryResolution').textContent = `${data.metadata.width} x ${data.metadata.height}`;
                        document.getElementById('summaryDuration').textContent = '-';
                    }
                    document.getElementById('summaryTime').textContent = `${data.processing_time}秒`;

                    // Update preview
                    const previewImage = document.getElementById('previewImage');
                    const previewVideo = document.getElementById('previewVideo');

                    // Unified preview for all upscale results (including GIF) using img tag
                    previewImage.src = data.image_url;
                    previewImage.style.display = 'block';
                    previewImage.style.cursor = 'pointer';
                    previewVideo.style.display = 'none';
                    previewImage.onclick = openModal;

                    showSection('result');
                    loadHistory();

                    // Re-enable tabs for next action
                    tabBtns.forEach(btn => btn.disabled = false);
                }
            } else {
                // Video processing or Upscale (always async now, including GIFs)
                const uploadResponse = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });

                const uploadData = await uploadResponse.json();

                if (!uploadResponse.ok) {
                    throw new Error(uploadData.error || 'アップロードに失敗しました。');
                }

                currentJobId = uploadData.job_id;

                // Start SSE for progress updates
                startProgressUpdates(currentJobId);

                if (currentMode === 'transparent') {
                    // Split processing ONLY for transparent mode (video)
                    // (Actually upscale now handles start inside the route)
                    const processResponse = await fetch(`/process/${currentJobId}`, {
                        method: 'POST'
                    });
                    const processData = await processResponse.json();
                    if (!processResponse.ok) throw new Error(processData.error || '処理の開始に失敗しました。');
                }
            }

        } catch (error) {
            if (eventSource) {
                eventSource.close();
            }
            showError(error.message);
        }
    });

    // Cancel processing
    cancelProcess.addEventListener('click', async () => {
        if (!currentJobId) return;

        cancelProcess.disabled = true;
        cancelProcess.textContent = 'キャンセル中...';

        try {
            await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
        } catch (error) {
            console.error('Failed to cancel:', error);
        }
    });

    // Progress updates via SSE
    function startProgressUpdates(jobId) {
        eventSource = new EventSource(`/progress/${jobId}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateProgress(data);

                // If finished, show result
                if (data.status === '完了') {
                    eventSource.close();

                    if (data.image_url) {
                        // Image / Static GIF result (Upscale)
                        const videoBtn = document.getElementById('downloadWebm');
                        const gifBtn = document.getElementById('downloadGif');
                        const zipBtn = document.getElementById('downloadZip');

                        videoBtn.href = data.image_url;
                        videoBtn.innerHTML = `<span class="download-icon">✨</span><span class="download-text"><strong>Download</strong><small>高画質化済み</small></span>`;
                        gifBtn.style.display = 'none';
                        zipBtn.style.display = 'none';

                        if (data.metadata) {
                            document.getElementById('summaryResolution').textContent = `${data.metadata.width} x ${data.metadata.height}`;
                            document.getElementById('summaryDuration').textContent = '-';
                        }
                        document.getElementById('summaryTime').textContent = `${data.processing_time}秒`;

                        const previewImage = document.getElementById('previewImage');
                        const previewVideo = document.getElementById('previewVideo');
                        previewImage.src = data.image_url;
                        previewImage.style.display = 'block';
                        previewVideo.style.display = 'none';
                        previewImage.onclick = openModal;

                    } else {
                        // Video / Animated GIF result
                        const videoUrl = data.video_url || data.webm_url;
                        downloadWebm.href = videoUrl;
                        downloadGif.href = data.gif_url;
                        downloadZip.href = data.zip_url;

                        // Show GIF/ZIP for videos
                        downloadGif.style.display = 'flex';
                        downloadZip.style.display = 'flex';

                        // Update button label based on format
                        if (data.video_ext === 'mp4') {
                            downloadWebm.innerHTML = '<span class="download-icon">📹</span><span class="download-text"><strong>MP4</strong><small>汎用・背景付き</small></span>';
                        } else if (data.video_ext === 'gif') {
                            downloadWebm.innerHTML = '<span class="download-icon">🎞️</span><span class="download-text"><strong>GIF</strong><small>高画質アニメ</small></span>';
                            downloadGif.style.display = 'none'; // Hide redundant GIF button
                        } else {
                            downloadWebm.innerHTML = '<span class="download-icon">📹</span><span class="download-text"><strong>WebM</strong><small>高品質・Web向け</small></span>';
                        }

                        // Update summary
                        if (data.metadata) {
                            document.getElementById('summaryResolution').textContent = `${data.metadata.width} x ${data.metadata.height}`;
                            document.getElementById('summaryDuration').textContent = `${data.metadata.duration}s`;
                        }
                        if (data.total_time) {
                            const totalSeconds = data.total_time;
                            const mins = Math.floor(totalSeconds / 60);
                            const secs = Math.floor(totalSeconds % 60);
                            document.getElementById('summaryTime').textContent = `${mins}分 ${secs}秒`;
                        }

                        // Update preview for video
                        const previewImage = document.getElementById('previewImage');
                        const previewVideo = document.getElementById('previewVideo');

                        if (data.video_ext === 'gif') {
                            // Display animated GIF in img tag
                            previewImage.src = data.video_url;
                            previewImage.style.display = 'block';
                            previewVideo.style.display = 'none';
                            previewImage.onclick = openModal;
                        } else {
                            if (data.video_url || data.webm_url) {
                                previewVideo.src = data.video_url || data.webm_url;
                                previewVideo.style.display = 'block';
                                previewImage.style.display = 'none';
                            } else {
                                // Fallback if URL missing
                                previewImage.src = '/static/images/placeholder.png';
                                previewImage.style.display = 'block';
                                previewVideo.style.display = 'none';
                            }
                        }
                    }

                    // Enable controls
                    tabBtns.forEach(btn => btn.disabled = false);
                    cancelProcess.disabled = false;
                    cancelProcess.innerHTML = '<span class="btn-icon">⏹️</span> キャンセル';

                    showSection('result');
                    loadHistory(); // Added history reload
                } else if (data.status === 'エラー') {
                    eventSource.close();
                    showError(data.error || '処理中にエラーが発生しました。');
                }
            } catch (e) {
                console.error('Failed to parse progress data:', e);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
        };
    }

    // Update progress UI
    function updateProgress(data) {
        const progress = data.progress || 0;
        const status = data.status || '処理中';
        const current = data.current || 0;
        const total = data.total || 0;
        const now = Date.now() / 1000;

        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
        statusBadge.textContent = status;

        if (total > 0) {
            frameInfo.textContent = `${current} / ${total} フレーム`;
            // Calculate more precise progress if current/total available
            if (current > 0 && progress < 95) {
                const preciseProgress = Math.round(10 + (current / total * 85));
                progressBar.style.width = `${preciseProgress}%`;
                progressText.textContent = `${preciseProgress}%`;
            }
        }

        // --- Resource Monitor ---
        if (data.resources) {
            const monitor = document.getElementById('resourceMonitor');
            monitor.style.display = 'grid';

            updateResourceBar('cpu', data.resources.cpu);
            updateResourceBar('ram', data.resources.ram);
            updateResourceBar('vram', data.resources.gpu_vram);
        }

        // --- Stable ETA Calculation ---
        const etaContainer = document.getElementById('etaContainer');
        const etaText = document.getElementById('etaText');

        if (total > 0 && current > 0 && current < total) {
            // Add to history
            progressHistory.push({ current, time: now });
            if (progressHistory.length > HISTORY_LIMIT) progressHistory.shift();

            if (progressHistory.length >= 5) {
                const first = progressHistory[0];
                const last = progressHistory[progressHistory.length - 1];
                const timeDiff = last.time - first.time;
                const frameDiff = last.current - first.current;

                if (timeDiff > 0 && frameDiff > 0) {
                    const fps = frameDiff / timeDiff; // Average FPS in window
                    const remainingFrames = total - current;
                    const remainingSeconds = remainingFrames / fps;

                    // Smoothing: 20% new value, 80% old value
                    let smoothedSeconds = remainingSeconds;
                    if (lastEta !== null) {
                        smoothedSeconds = (lastEta * 0.8) + (remainingSeconds * 0.2);
                    }
                    lastEta = smoothedSeconds;

                    const mins = Math.floor(smoothedSeconds / 60);
                    const secs = Math.floor(smoothedSeconds % 60);
                    etaText.textContent = `${mins}分 ${secs}秒`;
                    etaContainer.style.display = 'flex';
                }
            }
        } else {
            etaContainer.style.display = 'none';
            if (current === 0) progressHistory = [];
        }

        // Update step indicators
        updateStepIndicators(status);
    }

    function updateResourceBar(id, value) {
        const bar = document.getElementById(`${id}Bar`);
        const text = document.getElementById(`${id}Value`);
        if (bar && text) {
            bar.style.width = `${value}%`;
            text.textContent = `${value}%`;
            // Color coding
            if (value > 90) bar.style.background = 'var(--error-gradient)';
            else if (value > 70) bar.style.background = 'var(--warning-gradient)';
            else bar.style.background = 'var(--primary-gradient)';
        }
    }

    // Update step indicators
    function updateStepIndicators(status) {
        const stepOrder = ['step1', 'step2', 'step3', 'step4'];

        let currentStepId = null;
        for (const [key, id] of Object.entries(steps)) {
            if (status.startsWith(key)) {
                currentStepId = id;
                break;
            }
        }

        if (!currentStepId) return;

        const currentIndex = stepOrder.indexOf(currentStepId);

        stepOrder.forEach((stepId, index) => {
            const stepEl = document.getElementById(stepId);
            stepEl.classList.remove('active', 'completed');

            if (index < currentIndex) {
                stepEl.classList.add('completed');
            } else if (index === currentIndex) {
                stepEl.classList.add('active');
            }
        });
    }

    // Reset progress
    function resetProgress() {
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
        frameInfo.textContent = '';
        statusBadge.textContent = '準備中';
        progressHistory = [];
        lastEta = null;

        document.getElementById('resourceMonitor').style.display = 'none';
        document.getElementById('deviceBadge').style.display = 'none';

        cancelProcess.disabled = false;
        cancelProcess.innerHTML = '<span class="btn-icon">⏹️</span> キャンセル';

        ['step1', 'step2', 'step3', 'step4'].forEach(stepId => {
            document.getElementById(stepId).classList.remove('active', 'completed');
        });
    }

    // Show section
    function showSection(section) {
        uploadSection.style.display = section === 'upload' ? 'flex' : 'none';

        // Settings panels are managed by mode
        if (section === 'upload') {
            const transparentSettings = document.getElementById('transparentSettings');
            const upscaleSettings = document.getElementById('upscaleSettings');

            transparentSettings.style.display = currentMode === 'transparent' ? 'block' : 'none';
            upscaleSettings.style.display = currentMode === 'upscale' ? 'block' : 'none';

            // Sync active class for CSS consistency
            if (currentMode === 'transparent') {
                transparentSettings.classList.add('active');
                upscaleSettings.classList.remove('active');
            } else {
                transparentSettings.classList.remove('active');
                upscaleSettings.classList.add('active');
            }
        } else {
            document.getElementById('transparentSettings').style.display = 'none';
            document.getElementById('upscaleSettings').style.display = 'none';
            document.getElementById('transparentSettings').classList.remove('active');
            document.getElementById('upscaleSettings').classList.remove('active');
        }

        progressSection.style.display = section === 'progress' ? 'block' : 'none';
        resultSection.style.display = section === 'result' ? 'block' : 'none';
        errorSection.style.display = section === 'error' ? 'block' : 'none';
    }

    function disableModelSelection(disabled) {
        modelInputs.forEach(input => input.disabled = disabled);
        modelCards.forEach(card => {
            if (disabled) {
                card.style.pointerEvents = 'none';
                card.style.opacity = '0.7';
            } else {
                card.style.pointerEvents = 'auto';
                card.style.opacity = '1';
            }
        });
    }

    function disableBgColorSelection(disabled) {
        bgColorInputs.forEach(input => input.disabled = disabled);
        bgOptions.forEach(option => {
            if (disabled) {
                option.style.pointerEvents = 'none';
                option.style.opacity = '0.7';
            } else {
                option.style.pointerEvents = 'auto';
                option.style.opacity = '1';
            }
        });
        if (colorPicker) colorPicker.disabled = disabled;
        if (hexInput) hexInput.disabled = disabled;
    }

    // Show error
    function showError(message) {
        errorMessage.textContent = message;
        showSection('error');
        disableModelSelection(false);
        disableBgColorSelection(false);
        disableUpscaleSelection(false);
        tabBtns.forEach(btn => btn.disabled = false);
    }

    // Process another file
    processAnother.addEventListener('click', resetToUploadState);

    async function resetToUploadState() {
        // Cleanup current job
        if (currentJobId) {
            try {
                await fetch(`/cleanup/${currentJobId}`, { method: 'POST' });
            } catch (e) {
                console.error('Cleanup failed:', e);
            }
        }

        // Reset state
        currentFile = null;
        currentJobId = null;
        isImageFile = false;
        fileInput.value = '';
        selectedFile.style.display = 'none';
        startProcess.disabled = true;
        disableModelSelection(false);
        disableBgColorSelection(false);
        disableUpscaleSelection(false);
        tabBtns.forEach(btn => btn.disabled = false);

        // Reset download buttons for next use (keep UI clean)
        const webmBtn = document.getElementById('downloadWebm');
        webmBtn.innerHTML = `
            <span class="download-icon">📹</span>
            <span class="download-text">
                <strong>WebM</strong>
                <small>高品質・Web向け</small>
            </span>
        `;
        webmBtn.href = "#";
        document.getElementById('downloadGif').href = "#";
        document.getElementById('downloadZip').href = "#";

        document.getElementById('downloadGif').style.display = '';
        document.getElementById('downloadZip').style.display = '';

        showSection('upload');
    }

    // Retry button
    retryBtn.addEventListener('click', () => {
        showSection('upload');
    });

    // History Functions
    async function loadHistory() {
        try {
            const response = await fetch('/history');
            if (response.ok) {
                const history = await response.json();
                updateHistoryTable(history);
            }
        } catch (e) {
            console.error('Failed to load history:', e);
        }
    }

    function updateHistoryTable(history) {
        const tbody = document.getElementById('historyTableBody');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (history.length === 0) {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td colspan="8" style="text-align:center; color: var(--text-secondary);">履歴はありません</td>';
            tbody.appendChild(tr);
            return;
        }

        history.forEach(item => {
            const tr = document.createElement('tr');
            // Format bg color for display
            let bgDisplay = item.bg_color;
            if (bgDisplay === 'transparent') bgDisplay = '透過';
            else if (bgDisplay === '#000000') bgDisplay = '黒';
            else if (bgDisplay === '#FFFFFF') bgDisplay = '白';
            else if (bgDisplay === 'N/A' || !bgDisplay) bgDisplay = '-';
            else bgDisplay = `<span style="color:${item.bg_color}">■</span> ${item.bg_color}`;

            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td>${item.type}</td>
                <td>${item.model}</td>
                <td>${bgDisplay}</td>
                <td>${item.resolution}</td>
                <td>${item.input_size || item.size || '-'}</td>
                <td>${item.output_size || item.size || '-'}</td>
                <td>${item.processing_time}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Load history on startup
    loadHistory();

    // Download handlers - cleanup after all downloads
    [downloadWebm, downloadGif, downloadZip].forEach(btn => {
        btn.addEventListener('click', () => {
            // Optional: Add download tracking or cleanup logic here
        });
    });

    // Modal logic
    const previewModal = document.getElementById('previewModal');
    const modalImage = document.getElementById('modalImage');
    const modalVideo = document.getElementById('modalVideo');
    const modalClose = document.getElementById('modalClose');
    const modalOverlay = document.getElementById('modalOverlay');
    const previewArea = document.getElementById('previewArea');

    function openModal() {
        const previewImage = document.getElementById('previewImage');
        const previewVideo = document.getElementById('previewVideo');

        if (previewImage.style.display !== 'none') {
            modalImage.src = previewImage.src;
            modalImage.style.display = 'block';
            modalVideo.style.display = 'none';
        } else if (previewVideo.style.display !== 'none') {
            modalVideo.src = previewVideo.src;
            modalVideo.style.display = 'block';
            modalImage.style.display = 'none';
        }

        previewModal.classList.add('active');
        document.body.style.overflow = 'hidden'; // Prevent scrolling
    }

    function closeModal() {
        previewModal.classList.remove('active');
        document.body.style.overflow = '';
        modalVideo.pause();
        modalVideo.src = ''; // Clear source to stop loading
        modalImage.src = '';
    }

    if (previewArea) {
        previewArea.addEventListener('click', openModal);
    }
    if (modalClose) {
        modalClose.addEventListener('click', closeModal);
    }
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeModal);
    }

    // Escape key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && previewModal.classList.contains('active')) {
            closeModal();
        }
    });
});
