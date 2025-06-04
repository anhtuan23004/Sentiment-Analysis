// Global variables
let currentTab = 'single';
let uploadedFiles = [];
let analysisResults = [];
let downloadUrl = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadModels();
});

function initializeApp() {
    setupWordCounter('single-text', 'single-word-count');
    setupDragAndDrop();
}

function setupEventListeners() {
    document.getElementById('single-analyze').addEventListener('click', analyzeSingleText);
    document.getElementById('single-clear').addEventListener('click', clearSingleAnalysis);
    document.getElementById('batch-analyze').addEventListener('click', analyzeBatchFiles);
    document.getElementById('batch-clear').addEventListener('click', clearBatchAnalysis);
    document.getElementById('file-input').addEventListener('change', handleFileSelect);
}

function switchTab(tabName) {
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(tabName + '-tab').classList.add('active');
    currentTab = tabName;
}

function setupWordCounter(textareaId, counterId) {
    const textarea = document.getElementById(textareaId);
    const counter = document.getElementById(counterId);
    textarea.addEventListener('input', function() {
        const words = this.value.trim().split(/\s+/).filter(word => word.length > 0);
        const wordCount = this.value.trim() === '' ? 0 : words.length;
        counter.textContent = `${wordCount} / 5000 words`;
    });
}

function setupDragAndDrop() {
    const dragArea = document.getElementById('drag-area');
    const fileInput = document.getElementById('file-input');
    dragArea.addEventListener('click', () => fileInput.click());
    dragArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragArea.classList.add('dragover');
    });
    dragArea.addEventListener('dragleave', () => dragArea.classList.remove('dragover'));
    dragArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dragArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    const validFiles = files.filter(file => {
        const validTypes = ['text/csv', 'text/plain', 'application/csv'];
        return validTypes.includes(file.type) || file.name.endsWith('.csv') || file.name.endsWith('.txt');
    });
    if (validFiles.length !== files.length) {
        showToast('Some files were ignored. Only CSV and TXT files are supported.', 'error');
    }
    uploadedFiles = [...uploadedFiles, ...validFiles];
    updateFileDisplay();
}

function updateFileDisplay() {
    const dragArea = document.getElementById('drag-area');
    if (uploadedFiles.length > 0) {
        dragArea.innerHTML = `
            <i class="fas fa-check-circle" style="color: #28a745;"></i>
            <h3>${uploadedFiles.length} file(s) selected</h3>
            <div style="margin-top: 15px;">
                ${uploadedFiles.map(file => `
                    <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.5); padding: 10px; margin: 5px 0; border-radius: 8px;">
                        <span><i class="fas fa-file"></i> ${file.name}</span>
                        <button onclick="removeFile('${file.name}')" style="background: none; border: none; color: #dc3545; cursor: pointer;">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `).join('')}
            </div>
            <p style="margin-top: 10px; color: #6c757d; font-size: 14px;">
                Click to add more files or drag new files here
            </p>
        `;
    }
}

function removeFile(fileName) {
    uploadedFiles = uploadedFiles.filter(file => file.name !== fileName);
    if (uploadedFiles.length === 0) {
        document.getElementById('drag-area').innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <h3>Drag & Drop Files Here</h3>
            <p>or <strong>click to browse</strong></p>
            <p style="margin-top: 10px; color: #6c757d; font-size: 14px;">
                Supported formats: CSV, TXT (max 5000 words per file)
            </p>
        `;
    } else {
        updateFileDisplay();
    }
}

async function loadModels() {
    try {
        const res = await fetch('/models');
        const models = await res.json();
        const singleSelect = document.getElementById('single-model');
        const batchSelect = document.getElementById('batch-model');
        singleSelect.innerHTML = '';
        batchSelect.innerHTML = '';
        models.forEach(model => {
            const option1 = document.createElement('option');
            option1.value = model.value;
            option1.textContent = model.display_name;
            singleSelect.appendChild(option1);
            const option2 = document.createElement('option');
            option2.value = model.value;
            option2.textContent = model.display_name;
            batchSelect.appendChild(option2);
        });
        singleSelect.selectedIndex = 0;
        batchSelect.selectedIndex = 0;
    } catch (e) {
        showToast('Failed to load models from server.', 'error');
    }
}

async function analyzeSingleText() {
    const text = document.getElementById('single-text').value.trim();
    const model = document.getElementById('single-model').value;
    const analyzeBtn = document.getElementById('single-analyze');
    const progressContainer = document.getElementById('single-progress');
    const resultContainer = document.getElementById('single-result');

    if (!text) {
        showToast('Please enter some text to analyze.', 'error');
        return;
    }
    if (!model) {
        showToast('Please select a model.', 'error');
        return;
    }

    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    if (wordCount > 5000) {
        showToast('Text exceeds 5000 words limit.', 'error');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    progressContainer.style.display = 'block';
    resultContainer.innerHTML = '';

    try {
        console.log("Sending request with:", { text, model });

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model })
        });

        console.log("Response status:", res.status);
        const result = await res.json();
        console.log("Response data:", result);

        if (!res.ok) {
            showToast(result.error || 'Analysis failed.', 'error');
            displayError(result.error || 'Failed to analyze text.', 'single-result');
        } else {
            // Create the expected result format for displaySingleResult
            const formattedResult = {
                sentiment: result.prediction,
                confidence: result.confidence
            };

            displaySingleResult(formattedResult);
            showToast('Analysis completed successfully!', 'success');
        }
    } catch (error) {
        console.error("Analysis error:", error);
        displayError('Failed to analyze text. Please try again.', 'single-result');
        showToast('Analysis failed. Please try again.', 'error');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Sentiment';
        progressContainer.style.display = 'none';
    }
}
async function analyzeBatchFiles() {
    const model = document.getElementById('batch-model').value;
    const analyzeBtn = document.getElementById('batch-analyze');
    const progressContainer = document.getElementById('batch-progress');
    const resultContainer = document.getElementById('batch-result');

    if (uploadedFiles.length === 0) {
        showToast('Please upload at least one file.', 'error');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading-spinner"></span> Processing...';
    progressContainer.style.display = 'block';
    resultContainer.innerHTML = '';

    // Create FormData object
    const formData = new FormData();
    formData.append('model', model);

    // Add each file with the correct field name
    for (let i = 0; i < uploadedFiles.length; i++) {
        formData.append('files[]', uploadedFiles[i]);
        console.log(`Adding file ${i+1}: ${uploadedFiles[i].name}`);
    }

    try {
        console.log(`Sending ${uploadedFiles.length} files for batch analysis with model: ${model}`);

        const res = await fetch('/batch', {
            method: 'POST',
            body: formData
        });

        console.log("Batch response status:", res.status);

        if (!res.ok) {
            const errorText = await res.text();
            console.error("Server error:", errorText);
            throw new Error(`Server responded with status: ${res.status}`);
        }

        const contentType = res.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            console.error("Non-JSON response:", contentType);
            throw new Error("Server returned non-JSON response");
        }

        const result = await res.json();
        console.log("Batch response data:", result);

        if (result.error) {
            throw new Error(result.error);
        }

        if (!result.results || !Array.isArray(result.results) || result.results.length === 0) {
            throw new Error("No valid results returned from server");
        }

        // Process successful results
        analysisResults = result.results.map(r => ({
            text: r.text,
            filename: r.filename,
            sentiment: r.prediction,
            confidence: r.confidence
        }));

        downloadUrl = result.download_url;
        displayBatchResults(analysisResults);
        showToast(`Batch analysis completed! Processed ${analysisResults.length} texts.`, 'success');
    }
    catch (error) {
        console.error("Batch analysis error:", error);
        displayError(`Failed to process batch analysis: ${error.message}`, 'batch-result');
        showToast('Batch analysis failed. Please try again.', 'error');
    }
    finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Batch Analysis';
        progressContainer.style.display = 'none';
    }
}

function displaySingleResult(result) {
    const resultContainer = document.getElementById('single-result');
    if (!resultContainer) {
        console.error('Result container not found');
        return;
    }

    // Clear any previous results
    resultContainer.innerHTML = '';

    // Create result card
    const card = document.createElement('div');
    card.className = 'single-result-card';

    // Determine sentiment class and icon
    const sentimentClass = result.sentiment.toLowerCase();
    let icon = '';
    if (sentimentClass === 'positive') {
        icon = '<i class="fas fa-smile-beam"></i>';
    } else if (sentimentClass === 'negative') {
        icon = '<i class="fas fa-frown"></i>';
    } else {
        icon = '<i class="fas fa-meh"></i>';
    }

    // Format confidence as percentage
    const confidencePercent = result.confidence * 100;
    const confidenceText = `${confidencePercent.toFixed(2)}%`;

    // Add content to the card
    card.innerHTML = `
        <div class="result-header">
            <h3>Analysis Result</h3>
        </div>

        <div class="result-content">
            <div class="sentiment-section">
                <div class="sentiment-icon ${sentimentClass}">
                    ${icon}
                </div>
                <div class="sentiment-details">
                    <div class="sentiment-label">Sentiment:</div>
                    <div class="sentiment-value ${sentimentClass}">
                        ${result.sentiment}
                    </div>
                </div>
            </div>

            <div class="confidence-section">
                <div class="confidence-label">Confidence:</div>
                <div class="confidence-bar-container">
                    <div class="confidence-value">${confidenceText}</div>
                    <div class="confidence-meter">
                        <div class="confidence-progress" style="width: ${confidencePercent}%"></div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add style for single result
    const styleElement = document.createElement('style');
    styleElement.id = 'single-result-styles';
    styleElement.textContent = `
        .single-result-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-top: 20px;
        }

        .result-header {
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }

        .result-header h3 {
            margin: 0;
            color: #333;
            font-size: 1.2rem;
        }

        .result-content {
            padding: 20px;
        }

        .sentiment-section {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .sentiment-icon {
            font-size: 3.5rem;
            margin-right: 20px;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .sentiment-icon.positive {
            color: #28a745;
            background-color: rgba(40, 167, 69, 0.1);
        }

        .sentiment-icon.negative {
            color: #dc3545;
            background-color: rgba(220, 53, 69, 0.1);
        }

        .sentiment-details {
            flex-grow: 1;
        }

        .sentiment-label {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 5px;
        }

        .sentiment-value {
            font-size: 1.8rem;
            font-weight: bold;
        }

        .sentiment-value.positive {
            color: #28a745;
        }

        .sentiment-value.negative {
            color: #dc3545;
        }

        .confidence-section {
            margin-top: 15px;
        }

        .confidence-label {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 10px;
        }

        .confidence-bar-container {
            margin-top: 5px;
        }

        .confidence-value {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .confidence-meter {
            height: 8px;
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .confidence-progress {
            height: 100%;
            background-color: #6c757d;
            transition: width 0.8s ease-out;
        }
    `;

    // Remove any existing style element with the same ID
    const existingStyle = document.getElementById('single-result-styles');
    if (existingStyle) {
        existingStyle.remove();
    }
    document.head.appendChild(styleElement);

    // Add card to result container
    resultContainer.appendChild(card);
    resultContainer.style.display = 'block';
}

function displayBatchResults(results) {
    const resultContainer = document.getElementById('batch-result');
    if (!resultContainer) return;

    resultContainer.innerHTML = '';

    if (results.length === 0) {
        resultContainer.innerHTML = '<div class="no-results">No valid results returned</div>';
        return;
    }

    // Download button
    if (downloadUrl) {
        const downloadBtn = document.createElement('button');
        downloadBtn.id = 'download-btn';
        downloadBtn.className = 'btn btn-primary mb-4';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Results (CSV)';
        downloadBtn.onclick = function() {
            window.location.href = downloadUrl;
            showToast('Download started successfully!', 'success');
        };
        resultContainer.appendChild(downloadBtn);
    }

    // Calculate sentiment distribution
    const totalPositive = results.filter(r => r.sentiment === 'positive').length;
    const totalNegative = results.filter(r => r.sentiment === 'negative').length;

    // Create dashboard container
    const dashboardContainer = document.createElement('div');
    dashboardContainer.className = 'dashboard-container';
    resultContainer.appendChild(dashboardContainer);

    // Summary cards
    const summaryContainer = document.createElement('div');
    summaryContainer.className = 'summary-container';
    dashboardContainer.appendChild(summaryContainer);

    // Total texts card
    const totalCard = document.createElement('div');
    totalCard.className = 'summary-card';
    totalCard.innerHTML = `
        <div class="card-value">${results.length}</div>
        <div class="card-label">Total Texts</div>
    `;
    summaryContainer.appendChild(totalCard);

    // Positive texts card
    const positiveCard = document.createElement('div');
    positiveCard.className = 'summary-card positive';
    positiveCard.innerHTML = `
        <div class="card-value">${totalPositive}</div>
        <div class="card-label">Positive (${((totalPositive/results.length)*100).toFixed(2)}%)</div>
    `;
    summaryContainer.appendChild(positiveCard);

    // Negative texts card
    const negativeCard = document.createElement('div');
    negativeCard.className = 'summary-card negative';
    negativeCard.innerHTML = `
        <div class="card-value">${totalNegative}</div>
        <div class="card-label">Negative (${((totalNegative/results.length)*100).toFixed(2)}%)</div>
    `;
    summaryContainer.appendChild(negativeCard);

    // Charts container
    const chartsContainer = document.createElement('div');
    chartsContainer.className = 'charts-container';
    dashboardContainer.appendChild(chartsContainer);

    // Pie chart container
    const pieChartContainer = document.createElement('div');
    pieChartContainer.className = 'chart-box';
    chartsContainer.appendChild(pieChartContainer);

    const pieChartCanvas = document.createElement('canvas');
    pieChartCanvas.id = 'sentiment-pie';
    pieChartContainer.appendChild(pieChartCanvas);

    // Average confidence container
    const avgConfContainer = document.createElement('div');
    avgConfContainer.className = 'chart-box';
    chartsContainer.appendChild(avgConfContainer);

    // Calculate average confidence per sentiment
    const posConfidences = results.filter(r => r.sentiment === 'positive').map(r => r.confidence);
    const negConfidences = results.filter(r => r.sentiment === 'negative').map(r => r.confidence);

    const avgPosConf = posConfidences.length ?
        posConfidences.reduce((a, b) => a + b, 0) / posConfidences.length : 0;
    const avgNegConf = negConfidences.length ?
        negConfidences.reduce((a, b) => a + b, 0) / negConfidences.length : 0;

    // Create gauge charts for avg confidence
    const avgConfChartDiv = document.createElement('div');
    avgConfChartDiv.className = 'gauge-charts';
    avgConfContainer.appendChild(avgConfChartDiv);

    // Positive gauge
    const posGaugeDiv = document.createElement('div');
    posGaugeDiv.className = 'gauge-chart positive';
    posGaugeDiv.innerHTML = `
        <h4>Positive Confidence</h4>
        <div class="gauge-value">${(avgPosConf * 100).toFixed(2)}%</div>
        <div class="gauge-meter">
            <div class="gauge-progress" style="width: ${avgPosConf * 100}%"></div>
        </div>
    `;
    avgConfChartDiv.appendChild(posGaugeDiv);

    // Negative gauge
    const negGaugeDiv = document.createElement('div');
    negGaugeDiv.className = 'gauge-chart negative';
    negGaugeDiv.innerHTML = `
        <h4>Negative Confidence</h4>
        <div class="gauge-value">${(avgNegConf * 100).toFixed(2)}%</div>
        <div class="gauge-meter">
            <div class="gauge-progress" style="width: ${avgNegConf * 100}%"></div>
        </div>
    `;
    avgConfChartDiv.appendChild(negGaugeDiv);

    // Create table
    const tableContainer = document.createElement('div');
    tableContainer.className = 'table-container';
    resultContainer.appendChild(tableContainer);

    const tableHeading = document.createElement('h3');
    tableHeading.className = 'table-heading';
    tableHeading.textContent = 'Detailed Results';
    tableContainer.appendChild(tableHeading);

    const table = document.createElement('table');
    table.className = 'batch-results-table';
    tableContainer.appendChild(table);

    // Add header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>File</th>
            <th>Text</th>
            <th>Sentiment</th>
            <th>Confidence</th>
        </tr>
    `;
    table.appendChild(thead);

    // Add body
    const tbody = document.createElement('tbody');
    results.forEach(result => {
        const row = document.createElement('tr');

        // Truncate long text
        const textDisplay = result.text.length > 100
            ? result.text.substring(0, 100) + '...'
            : result.text;

        // Format confidence as percentage
        const confidence = (result.confidence * 100).toFixed(2) + '%';

        // Set sentiment class
        const sentimentClass = result.sentiment.toLowerCase();

        row.innerHTML = `
            <td>${result.filename || 'N/A'}</td>
            <td title="${result.text}">${textDisplay}</td>
            <td>
                <span class="sentiment-badge ${sentimentClass}">
                    ${result.sentiment}
                </span>
            </td>
            <td>
                <div class="confidence-bar-container">
                    <div class="confidence-bar ${sentimentClass}" style="width: ${result.confidence * 100}%"></div>
                    <span>${confidence}</span>
                </div>
            </td>
        `;

        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    // Add styles
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .dashboard-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        .summary-card {
            padding: 20px;
            border-radius: 8px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .summary-card.positive {
            border-left: 5px solid #28a745;
        }
        .summary-card.negative {
            border-left: 5px solid #dc3545;
        }
        .card-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .card-label {
            margin-top: 10px;
            color: #666;
            font-size: 1rem;
        }
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .chart-box {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 300px;
        }
        .gauge-charts {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 15px;
        }
        .gauge-chart {
            text-align: center;
        }
        .gauge-chart.positive h4 {
            color: #28a745;
        }
        .gauge-chart.negative h4 {
            color: #dc3545;
        }
        .gauge-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .gauge-meter {
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .gauge-progress {
            height: 100%;
        }
        .gauge-chart.positive .gauge-progress {
            background-color: #28a745;
        }
        .gauge-chart.negative .gauge-progress {
            background-color: #dc3545;
        }
        .table-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .table-heading {
            margin-top: 0;
            margin-bottom: 15px;
            color: #333;
        }
        .batch-results-table {
            width: 100%;
            border-collapse: collapse;
        }
        .batch-results-table th {
            background: #f8f9fa;
            padding: 12px 15px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        .batch-results-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            font-size: 0.8rem;
        }
        .sentiment-badge.positive {
            background-color: #28a745;
        }
        .sentiment-badge.negative {
            background-color: #dc3545;
        }
        .confidence-bar-container {
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 4px;
            height: 20px;
            position: relative;
            overflow: hidden;
        }
        .confidence-bar {
            height: 100%;
            position: absolute;
            left: 0;
            top: 0;
        }
        .confidence-bar-container span {
            position: absolute;
            color: #212529;
            font-size: 0.8rem;
            font-weight: bold;
            padding-left: 5px;
            line-height: 20px;
            z-index: 1;
        }
    `;
    document.head.appendChild(styleElement);

    // Initialize pie chart
    new Chart(pieChartCanvas, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                data: [totalPositive, totalNegative],
                backgroundColor: ['#28a745', '#dc3545'],
                borderColor: ['#ffffff', '#ffffff'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Sentiment Distribution',
                    font: {
                        size: 16
                    }
                }
            },
            cutout: '60%'
        }
    });

    resultContainer.style.display = 'block';
}

function clearSingleAnalysis() {
    document.getElementById('single-text').value = '';
    document.getElementById('single-word-count').textContent = '0 / 500 words';
    document.getElementById('single-word-count').classList.remove('warning', 'error');
    document.getElementById('single-result').innerHTML = '';
    showToast('Single analysis cleared.', 'success');
}

function clearBatchAnalysis() {
    uploadedFiles = [];
    analysisResults = [];
    downloadUrl = null;

    // Use querySelector instead of getElementById for more flexibility
    const fileInput = document.querySelector('#file-input');
    if (fileInput) {
        fileInput.value = '';
    }

    const batchResult = document.querySelector('#batch-result');
    if (batchResult) {
        batchResult.innerHTML = '';
    }

    const dragArea = document.querySelector('#drag-area');
    if (dragArea) {
        dragArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <h3>Drag & Drop Files Here</h3>
            <p>or <strong>click to browse</strong></p>
            <p style="margin-top: 10px; color: #6c757d; font-size: 14px;">
                Supported formats: CSV, TXT (max 500 words per text)
            </p>
        `;
    }

    showToast('Batch analysis cleared.', 'success');
}


function displayError(message, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="result-card error">
            <div class="result-header">
                <i class="fas fa-exclamation-triangle"></i>
                <h4>Error</h4>
            </div>
            <p style="margin: 0; color: #721c24;">${message}</p>
        </div>
    `;
}

function showToast(message, type = 'success') {
    const existingToast = document.querySelector('.toast');
    if (existingToast) existingToast.remove();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        ${message}
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 100);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (currentTab === 'single') analyzeSingleText();
        else analyzeBatchFiles();
    }
    if (e.key === 'Escape') {
        if (currentTab === 'single') clearSingleAnalysis();
        else clearBatchAnalysis();
    }
});

// Auto-save functionality (optional)
let autoSaveTimeout;
document.getElementById('single-text').addEventListener('input', function() {
    clearTimeout(autoSaveTimeout);
    autoSaveTimeout = setTimeout(() => {
        localStorage.setItem('sentiment_draft', this.value);
    }, 1000);
});
window.addEventListener('load', function() {
    const draft = localStorage.getItem('sentiment_draft');
    if (draft) {
        document.getElementById('single-text').value = draft;
        document.getElementById('single-text').dispatchEvent(new Event('input'));
    }
});