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
        'AIモデルをロード中...': 'step2',
        '背景透過処理中': 'step2',
        'WebM生成中': 'step3',
        'GIF生成中': 'step4'
    };

    let currentFile = null;
    let currentJobId = null;
    let eventSource = null;
    let selectedModel = 'u2net_human_seg';

    // For stable ETA
    let progressHistory = [];
    const HISTORY_LIMIT = 20;
    let lastEta = null;

    // Model selection logic
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[type="radio"]');
            radio.checked = true;
            selectedModel = radio.value;
        });
    });

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
        const allowedTypes = ['video/mp4', 'video/quicktime', 'video/webm'];
        const allowedExtensions = ['mp4', 'mov', 'webm'];
        const maxSize = 100 * 1024 * 1024; // 100MB

        const ext = file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(ext)) {
            showError('対応していないファイル形式です。MP4、MOV、WebM形式のファイルを選択してください。');
            return;
        }

        if (file.size > maxSize) {
            showError('ファイルサイズが大きすぎます。100MB以下のファイルを選択してください。');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
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
            // Upload file
            const formData = new FormData();
            formData.append('video', currentFile);
            formData.append('model', selectedModel);

            showSection('progress');
            resetProgress();

            // Disable UI
            disableModelSelection(true);

            const uploadResponse = await fetch('/upload', {
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

            // Start processing (now async)
            const processResponse = await fetch(`/process/${currentJobId}`, {
                method: 'POST'
            });

            const processData = await processResponse.json();

            if (!processResponse.ok) {
                throw new Error(processData.error || '処理の開始に失敗しました。');
            }

            // Note: We don't show result here anymore, SSE will handle it

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

                    downloadWebm.href = data.webm_url;
                    downloadGif.href = data.gif_url;
                    downloadZip.href = data.zip_url;

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

                    showSection('result');
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

        // Device Display
        const deviceBadge = document.getElementById('deviceBadge');
        if (data.device) {
            deviceBadge.textContent = data.device;
            deviceBadge.style.display = 'inline-block';
            // Style based on device
            if (data.device.includes('GPU')) {
                deviceBadge.style.color = '#60a5fa';
                deviceBadge.style.borderColor = 'rgba(96, 165, 250, 0.3)';
            } else {
                deviceBadge.style.color = '#fbbf24';
                deviceBadge.style.borderColor = 'rgba(251, 191, 36, 0.3)';
            }
        }

        if (total > 0) {
            frameInfo.textContent = `${current} / ${total} フレーム`;
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
        document.getElementById('modelSelectionSection').style.display = section === 'upload' ? 'block' : 'none';
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

    // Show error
    function showError(message) {
        errorMessage.textContent = message;
        showSection('error');
        disableModelSelection(false);
    }

    // Process another video
    processAnother.addEventListener('click', async () => {
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
        fileInput.value = '';
        selectedFile.style.display = 'none';
        startProcess.disabled = true;
        disableModelSelection(false);

        showSection('upload');
    });

    // Retry button
    retryBtn.addEventListener('click', () => {
        showSection('upload');
    });

    // Download handlers - cleanup after all downloads
    [downloadWebm, downloadGif, downloadZip].forEach(btn => {
        btn.addEventListener('click', () => {
            // Optional: Add download tracking or cleanup logic here
        });
    });
});
