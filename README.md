# Binary Prediction of Smoker Status using Bio-Signals

The tobacco epidemic remains a major public health threat, causing over 8 million deaths annually. Smoking is a key driver of this crisis. This project aims to classify adult smoking status using biosignals such as age, height, weight, and cholesterol levels, providing valuable insights for public health agencies. Our best model, XGBoost, used 500 rounds, η = 0.09 to reduce overfitting, max depth = 7, and γ = 2, achieving an AUC of 0.881 on test data and a Kaggle score of 0.87131. The second-best model, a neural network with two hidden layers (50 units each), ReLU activation, and 0.4 dropout, achieved a Kaggle score of 0.85661.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smoker Status Prediction using Bio-Signals</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            height: 400px;
            max-height: 50vh;
        }
        @media (min-width: 768px) {
            .chart-container {
                height: 450px;
            }
        }
        .nav-link {
            transition: color 0.3s, border-bottom-color 0.3s;
        }
        .nav-link.active {
            color: #ca8a04;
            border-bottom-color: #ca8a04;
        }
        .stat-card {
            background-color: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -2px rgb(0 0 0 / 0.1);
        }
        .heatmap-cell {
            width: 2.5rem;
            height: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            color: white;
            position: relative;
        }
        .heatmap-label {
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            text-align: right;
            padding-right: 8px;
            font-weight: 500;
        }
        .tooltip {
            visibility: hidden;
            background-color: #1f2937;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 10px;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            width: 120px;
        }
        .heatmap-cell:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body class="text-slate-700">

    <header class="bg-white/80 backdrop-blur-md sticky top-0 z-50 shadow-sm">
        <nav class="container mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex-shrink-0">
                    <h1 class="text-xl font-bold text-slate-800">Smoker Prediction Analysis</h1>
                </div>
                <div class="hidden md:block">
                    <div class="ml-10 flex items-baseline space-x-4">
                        <a href="#overview" class="nav-link px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:text-amber-600 border-b-2 border-transparent">Overview</a>
                        <a href="#data-insights" class="nav-link px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:text-amber-600 border-b-2 border-transparent">Data Insights</a>
                        <a href="#model-showdown" class="nav-link px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:text-amber-600 border-b-2 border-transparent">Model Showdown</a>
                        <a href="#deep-dive" class="nav-link px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:text-amber-600 border-b-2 border-transparent">Predictor Deep Dive</a>
                    </div>
                </div>
            </div>
        </nav>
    </header>

    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        
        <section id="overview" class="my-12 scroll-mt-20">
            <div class="max-w-4xl mx-auto text-center">
                <h2 class="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">Predicting Smoker Status with Bio-Signals</h2>
                <p class="mt-4 text-lg text-slate-600">This project explores the classification of an individual's smoking status using a variety of bio-signal data. By applying several machine learning models, from logistic regression to neural networks, the analysis identifies key predictive factors and determines the most effective algorithm for this public health challenge.</p>
            </div>
        </section>

        <section id="data-insights" class="my-16 scroll-mt-20">
            <div class="text-center mb-12">
                <h3 class="text-2xl font-bold text-slate-800">Exploring the Data</h3>
                <p class="mt-2 text-md text-slate-500">Interactive visualizations to understand the relationships between bio-signals and smoking status.</p>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <div class="stat-card">
                    <h4 class="font-bold text-lg mb-4 text-slate-800">Bio-Signal Distributions</h4>
                    <div class="flex items-center mb-4">
                        <label for="variable-selector" class="mr-2 text-sm font-medium">Select a variable:</label>
                        <select id="variable-selector" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-amber-500 focus:border-amber-500 block w-full p-2.5">
                        </select>
                    </div>
                    <div class="chart-container" style="height: 350px; max-height: 45vh;">
                        <canvas id="boxplotChart"></canvas>
                    </div>
                    <p class="text-xs text-slate-500 mt-2 text-center">This chart shows the distribution of the selected bio-signal for smokers and non-smokers. Different distributions suggest the variable may be a useful predictor.</p>
                </div>

                <div class="stat-card">
                    <h4 class="font-bold text-lg mb-4 text-slate-800">Predictor Correlation Matrix</h4>
                    <div id="heatmap-container" class="flex justify-center overflow-x-auto"></div>
                     <p class="text-xs text-slate-500 mt-2 text-center">This heatmap reveals multicollinearity between predictor variables. Bright red indicates a strong positive correlation, while bright blue indicates a strong negative one. This insight informed the use of regularization techniques like Ridge and Lasso.</p>
                </div>
            </div>
        </section>

        <section id="model-showdown" class="my-16 scroll-mt-20">
            <div class="text-center mb-12">
                <h3 class="text-2xl font-bold text-slate-800">Model Performance Showdown</h3>
                <p class="mt-2 text-md text-slate-500">Comparing the predictive power of various machine learning models using the Area Under Curve (AUC) metric.</p>
            </div>
            <div class="stat-card">
                <div class="grid grid-cols-1 md:grid-cols-5 gap-8">
                    <div class="md:col-span-3">
                        <div class="chart-container" style="height: 400px; max-height: 60vh;">
                            <canvas id="aucChart"></canvas>
                        </div>
                    </div>
                    <div id="model-info" class="md:col-span-2 bg-slate-50 p-6 rounded-lg flex flex-col justify-center">
                        <h4 id="model-info-title" class="font-bold text-lg text-slate-800 mb-2">Select a model</h4>
                        <p id="model-info-text" class="text-slate-600 text-sm">Click on a bar in the chart to see details about the model's performance and characteristics.</p>
                    </div>
                </div>
            </div>
        </section>

        <section id="deep-dive" class="my-16 scroll-mt-20">
            <div class="text-center mb-12">
                <h3 class="text-2xl font-bold text-slate-800">A Deep Dive into the Champion Model: XGBoost</h3>
                <p class="mt-2 text-md text-slate-500">The XGBoost model emerged as the top performer. Let's explore what makes it effective.</p>
            </div>
             <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <div class="stat-card">
                    <h4 class="font-bold text-lg mb-4 text-slate-800">Key Predictors of Smoking Status</h4>
                    <div class="chart-container" style="height: 450px; max-height: 60vh;">
                        <canvas id="importanceChart"></canvas>
                    </div>
                    <p class="text-xs text-slate-500 mt-2 text-center">This plot shows the most influential bio-signals in the XGBoost model. Hemoglobin, height, and Gtp levels are the strongest predictors.</p>
                </div>
                <div class="stat-card">
                    <h4 class="font-bold text-lg mb-4 text-slate-800">XGBoost Training Performance</h4>
                    <div class="chart-container" style="height: 450px; max-height: 60vh;">
                        <canvas id="loglossChart"></canvas>
                    </div>
                    <p class="text-xs text-slate-500 mt-2 text-center">The training and testing log loss curves show the model learning over 500 iterations. The testing loss (red) plateaus, indicating the model learned effectively without significant overfitting.</p>
                </div>
            </div>
        </section>
        
        <footer class="text-center mt-16 py-8 border-t border-slate-200">
            <p class="text-sm text-slate-500">Project by Souvik Bag | STAT 8330, University of Missouri</p>
        </footer>
    </main>

    <script>
        const projectData = {
            variables: ['age', 'height.cm', 'weight.kg', 'waist.cm', 'eyesight.left', 'eyesight.right', 'systolic', 'relaxation', 'fasting.blood.sugar', 'Cholesterol', 'triglyceride', 'HDL', 'LDL', 'hemoglobin', 'serum.creatinine', 'AST', 'ALT', 'Gtp', 'dental.caries', 'Urine.protein'],
            distributions: {
                'age': { smoker: [40, 45, 50, 55, 60], nonsmoker: [35, 40, 45, 50, 55] },
                'height.cm': { smoker: [165, 170, 175, 180, 185], nonsmoker: [155, 160, 165, 170, 175] },
                'weight.kg': { smoker: [65, 70, 75, 80, 90], nonsmoker: [55, 60, 65, 70, 75] },
                'waist.cm': { smoker: [80, 85, 90, 95, 100], nonsmoker: [75, 80, 85, 90, 95] },
                'eyesight.left': { smoker: [0.8, 1.0, 1.0, 1.2, 1.5], nonsmoker: [0.9, 1.0, 1.0, 1.2, 1.5] },
                'eyesight.right': { smoker: [0.8, 1.0, 1.0, 1.2, 1.5], nonsmoker: [0.9, 1.0, 1.0, 1.2, 1.5] },
                'systolic': { smoker: [115, 120, 130, 135, 140], nonsmoker: [110, 118, 125, 130, 138] },
                'relaxation': { smoker: [70, 75, 80, 85, 90], nonsmoker: [68, 72, 78, 82, 88] },
                'fasting.blood.sugar': { smoker: [90, 95, 100, 110, 120], nonsmoker: [88, 92, 98, 105, 115] },
                'Cholesterol': { smoker: [180, 200, 220, 240, 260], nonsmoker: [175, 195, 215, 235, 255] },
                'triglyceride': { smoker: [130, 150, 180, 220, 280], nonsmoker: [80, 100, 130, 160, 200] },
                'HDL': { smoker: [40, 45, 50, 55, 60], nonsmoker: [45, 50, 55, 60, 70] },
                'LDL': { smoker: [100, 110, 130, 150, 170], nonsmoker: [95, 105, 125, 145, 165] },
                'hemoglobin': { smoker: [14.5, 15.5, 16.0, 16.5, 17.0], nonsmoker: [13.0, 13.5, 14.0, 14.5, 15.0] },
                'serum.creatinine': { smoker: [0.8, 0.9, 1.0, 1.1, 1.2], nonsmoker: [0.7, 0.8, 0.9, 1.0, 1.1] },
                'AST': { smoker: [20, 25, 30, 35, 45], nonsmoker: [18, 22, 26, 30, 38] },
                'ALT': { smoker: [25, 35, 45, 60, 80], nonsmoker: [15, 20, 28, 35, 45] },
                'Gtp': { smoker: [30, 45, 60, 80, 120], nonsmoker: [15, 20, 25, 30, 40] },
                'dental.caries': { smoker: [0, 0, 0, 1, 1], nonsmoker: [0, 0, 0, 0, 1] },
                'Urine.protein': { smoker: [1, 1, 1, 1, 2], nonsmoker: [1, 1, 1, 1, 1] },
            },
            correlation: {
                labels: ['height', 'weight', 'waist', 'trigly', 'HDL', 'LDL', 'hemo', 'Gtp'],
                matrix: [
                    [1.0, 0.7, 0.8, 0.3, -0.4, 0.3, 0.5, 0.2],
                    [0.7, 1.0, 0.8, 0.6, -0.5, 0.4, 0.5, 0.2],
                    [0.8, 0.8, 1.0, 0.5, -0.4, 0.4, 0.4, 0.3],
                    [0.3, 0.6, 0.5, 1.0, -0.5, 0.5, 0.5, 0.3],
                    [-0.4, -0.5, -0.4, -0.5, 1.0, -0.4, -0.3, -0.2],
                    [0.3, 0.4, 0.4, 0.5, -0.4, 1.0, 0.3, 0.1],
                    [0.5, 0.5, 0.4, 0.5, -0.3, 0.3, 1.0, 0.2],
                    [0.2, 0.2, 0.3, 0.3, -0.2, 0.1, 0.2, 1.0]
                ]
            },
            models: [
                { name: 'Logistic Regression', auc: 0.847, text: 'A solid baseline model. It establishes the initial benchmark for performance but struggles with the correlated nature of the bio-signals.' },
                { name: 'Ridge Regression', auc: 0.828, text: 'A regularized model intended to handle multicollinearity. Performance was slightly lower than the baseline, suggesting its form of regularization was not optimal for this dataset.' },
                { name: 'Random Forest', auc: 0.870, text: 'A tree-based ensemble model that significantly improved upon logistic regression by capturing non-linear relationships. It demonstrated strong predictive power.' },
                { name: 'XGBoost', auc: 0.881, text: 'The top-performing model. Extreme Gradient Boosting excels by building sequential trees that correct prior errors, leading to the highest AUC score of 0.881.' },
                { name: 'Neural Network', auc: 0.857, text: 'A deep learning approach with two hidden layers. It performed well, outperforming the baseline, but was not as effective as the XGBoost model for this structured dataset.' },
            ],
            importance: [
                { variable: 'hemoglobin', value: 8500 },
                { variable: 'height.cm', value: 8200 },
                { variable: 'Gtp', value: 7800 },
                { variable: 'triglyceride', value: 6000 },
                { variable: 'LDL', value: 4500 },
                { variable: 'HDL', value: 4400 },
                { variable: 'Cholesterol', value: 4300 },
                { variable: 'weight.kg', value: 4200 },
                { variable: 'waist.cm', value: 4100 },
                { variable: 'ALT', value: 4000 },
            ],
            logloss: {
                iterations: Array.from({ length: 50 }, (_, i) => (i + 1) * 10),
                training: [0.65, 0.60, 0.55, 0.51, 0.48, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.405, 0.40, 0.395, 0.39, 0.387, 0.385, 0.383, 0.381, 0.38, 0.378, 0.377, 0.376, 0.375, 0.374, 0.373, 0.372, 0.371, 0.37, 0.369, 0.368, 0.367, 0.366, 0.365, 0.364, 0.363, 0.362, 0.361, 0.36, 0.359, 0.358, 0.357, 0.356, 0.355, 0.354, 0.353, 0.352, 0.351, 0.35, 0.349],
                testing: [0.66, 0.62, 0.58, 0.54, 0.51, 0.49, 0.47, 0.46, 0.45, 0.44, 0.435, 0.43, 0.428, 0.426, 0.425, 0.424, 0.423, 0.422, 0.421, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
            }
        };

        document.addEventListener('DOMContentLoaded', () => {
            const selector = document.getElementById('variable-selector');
            projectData.variables.forEach(v => {
                const option = document.createElement('option');
                option.value = v;
                option.textContent = v.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                selector.appendChild(option);
            });

            const boxplotCtx = document.getElementById('boxplotChart').getContext('2d');
            let boxplotChart;

            function updateBoxplot(variable) {
                if (boxplotChart) {
                    boxplotChart.destroy();
                }
                const data = projectData.distributions[variable];
                if (!data) return;

                boxplotChart = new Chart(boxplotCtx, {
                    type: 'boxplot',
                    data: {
                        labels: ['Non-Smoker', 'Smoker'],
                        datasets: [{
                            label: variable,
                            data: [data.nonsmoker, data.smoker],
                            backgroundColor: 'rgba(202, 138, 4, 0.2)',
                            borderColor: 'rgba(202, 138, 4, 1)',
                            borderWidth: 1,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const stats = context.raw;
                                        return [
                                            `Max: ${stats.max}`,
                                            `Q3: ${stats.q3}`,
                                            `Median: ${stats.median}`,
                                            `Q1: ${stats.q1}`,
                                            `Min: ${stats.min}`,
                                        ];
                                    }
                                }
                            }
                        },
                        scales: {
                             y: {
                                beginAtZero: false,
                                title: { display: true, text: 'Value' }
                             }
                        }
                    }
                });
            }
            
            Chart.register({
                id: 'boxplot',
                datasetElementType: 'bar',
                dataElementType: 'bar',
                defaults: {
                    horizontal: false,
                },
                update: function (chart, args, options) {
                    const me = this;
                    const ds = chart.data.datasets[0];
                    const data = ds.data.map(d => {
                        const sorted = d.slice().sort((a,b) => a - b);
                        const min = sorted[0];
                        const q1 = sorted[Math.floor(sorted.length / 4)];
                        const median = sorted[Math.floor(sorted.length / 2)];
                        const q3 = sorted[Math.floor(sorted.length * 3 / 4)];
                        const max = sorted[sorted.length - 1];
                        return {min, q1, median, q3, max};
                    });
                    
                    const meta = chart.getDatasetMeta(0);
                    meta.data.forEach((bar, index) => {
                        const box = data[index];
                        const vm = bar;
                        vm.x = meta.xScale.getPixelForValue(chart.data.labels[index]);
                        vm.width = meta.xScale.getBandwidth() * 0.8;
                        vm.y = meta.yScale.getPixelForValue(box.median);
                        vm.minY = meta.yScale.getPixelForValue(box.min);
                        vm.q1Y = meta.yScale.getPixelForValue(box.q1);
                        vm.q3Y = meta.yScale.getPixelForValue(box.q3);
                        vm.maxY = meta.yScale.getPixelForValue(box.max);
                        bar.custom = box; 
                    });
                },
                draw: function (chart, args, options) {
                    const meta = chart.getDatasetMeta(0);
                    const ctx = chart.ctx;
                    const ds = chart.data.datasets[0];

                    ctx.save();
                    meta.data.forEach(bar => {
                        const vm = bar;
                        ctx.strokeStyle = ds.borderColor;
                        ctx.lineWidth = ds.borderWidth;
                        ctx.fillStyle = ds.backgroundColor;

                        ctx.beginPath();
                        ctx.moveTo(vm.x - vm.width / 2, vm.maxY);
                        ctx.lineTo(vm.x + vm.width / 2, vm.maxY);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(vm.x, vm.maxY);
                        ctx.lineTo(vm.x, vm.q3Y);
                        ctx.stroke();

                        ctx.strokeRect(vm.x - vm.width / 2, vm.q3Y, vm.width, vm.q1Y - vm.q3Y);
                        ctx.fillRect(vm.x - vm.width / 2, vm.q3Y, vm.width, vm.q1Y - vm.q3Y);
                        
                        ctx.beginPath();
                        ctx.moveTo(vm.x - vm.width / 2, vm.y);
                        ctx.lineTo(vm.x + vm.width / 2, vm.y);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(vm.x, vm.q1Y);
                        ctx.lineTo(vm.x, vm.minY);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(vm.x - vm.width / 2, vm.minY);
                        ctx.lineTo(vm.x + vm.width / 2, vm.minY);
                        ctx.stroke();
                    });
                    ctx.restore();
                }
            });

            selector.addEventListener('change', (e) => updateBoxplot(e.target.value));
            updateBoxplot(projectData.variables[0]);
            
            const heatmapContainer = document.getElementById('heatmap-container');
            const { labels, matrix } = projectData.correlation;
            const table = document.createElement('table');
            table.className = "border-collapse";
            const headerRow = document.createElement('tr');
            headerRow.appendChild(document.createElement('th'));
            labels.forEach(label => {
                const th = document.createElement('th');
                th.textContent = label;
                th.className = 'p-2 text-xs font-medium text-slate-600';
                headerRow.appendChild(th);
            });
            table.appendChild(headerRow);
            
            matrix.forEach((row, i) => {
                const tr = document.createElement('tr');
                const labelTh = document.createElement('th');
                labelTh.className = "heatmap-label text-xs text-slate-600";
                labelTh.textContent = labels[i];
                tr.appendChild(labelTh);
                row.forEach((value, j) => {
                    const td = document.createElement('td');
                    td.className = 'heatmap-cell';
                    const opacity = Math.abs(value);
                    const color = value > 0 ? `rgba(220, 38, 38, ${opacity})` : `rgba(37, 99, 235, ${opacity})`;
                    td.style.backgroundColor = value === 1 ? '#6b7280' : color;
                    
                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    tooltip.textContent = `${labels[i]} / ${labels[j]}: ${value.toFixed(2)}`;
                    td.appendChild(tooltip);
                    
                    tr.appendChild(td);
                });
                table.appendChild(tr);
            });
            heatmapContainer.appendChild(table);


            const aucCtx = document.getElementById('aucChart').getContext('2d');
            const modelInfoTitle = document.getElementById('model-info-title');
            const modelInfoText = document.getElementById('model-info-text');

            const aucChart = new Chart(aucCtx, {
                type: 'bar',
                data: {
                    labels: projectData.models.map(m => m.name),
                    datasets: [{
                        label: 'AUC Score',
                        data: projectData.models.map(m => m.auc),
                        backgroundColor: [
                            'rgba(59, 130, 246, 0.6)',
                            'rgba(16, 185, 129, 0.6)',
                            'rgba(239, 68, 68, 0.6)',
                            'rgba(245, 158, 11, 0.6)',
                            'rgba(139, 92, 246, 0.6)',
                        ],
                        borderColor: [
                            'rgba(59, 130, 246, 1)',
                            'rgba(16, 185, 129, 1)',
                            'rgba(239, 68, 68, 1)',
                            'rgba(245, 158, 11, 1)',
                            'rgba(139, 92, 246, 1)',
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, },
                    scales: { x: { beginAtZero: true, suggestedMax: 0.9, title: {display: true, text: 'AUC Score'} } },
                    onClick: (event, elements) => {
                        if (elements.length > 0) {
                            const index = elements[0].index;
                            const model = projectData.models[index];
                            modelInfoTitle.textContent = model.name;
                            modelInfoText.textContent = model.text;
                        }
                    }
                }
            });
            
            const importanceCtx = document.getElementById('importanceChart').getContext('2d');
            const sortedImportance = [...projectData.importance].sort((a, b) => a.value - b.value);
            const importanceChart = new Chart(importanceCtx, {
                type: 'bar',
                data: {
                    labels: sortedImportance.map(item => item.variable),
                    datasets: [{
                        label: 'Importance',
                        data: sortedImportance.map(item => item.value),
                        backgroundColor: 'rgba(217, 119, 6, 0.7)',
                        borderColor: 'rgba(217, 119, 6, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, },
                    scales: { x: { title: { display: true, text: 'Importance Score' } } }
                }
            });

            const loglossCtx = document.getElementById('loglossChart').getContext('2d');
            const loglossChart = new Chart(loglossCtx, {
                type: 'line',
                data: {
                    labels: projectData.logloss.iterations,
                    datasets: [
                        {
                            label: 'Training Log Loss',
                            data: projectData.logloss.training,
                            borderColor: 'rgba(37, 99, 235, 1)',
                            backgroundColor: 'rgba(37, 99, 235, 0.1)',
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: 'Testing Log Loss',
                            data: projectData.logloss.testing,
                            borderColor: 'rgba(220, 38, 38, 1)',
                            backgroundColor: 'rgba(220, 38, 38, 0.1)',
                            fill: false,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Iteration' } },
                        y: { title: { display: true, text: 'Log Loss' } }
                    }
                }
            });
            
            const sections = document.querySelectorAll('section');
            const navLinks = document.querySelectorAll('.nav-link');

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        navLinks.forEach(link => {
                            link.classList.toggle('active', link.getAttribute('href').substring(1) === entry.target.id);
                        });
                    }
                });
            }, { rootMargin: "-50% 0px -50% 0px" });

            sections.forEach(section => observer.observe(section));

            navLinks.forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });

        });
    </script>
</body>
</html>
