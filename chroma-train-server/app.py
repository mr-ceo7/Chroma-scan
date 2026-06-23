import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_file
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

# ═══════════════════════════════════════════════════════════
#  PREMIUM GLASSMORPHIC HTML TEMPLATE
# ═══════════════════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chroma-Train AI Studio</title>
  <style>
    :root {
      --bg-color: #080c14;
      --panel-bg: rgba(15, 23, 42, 0.7);
      --border-color: rgba(56, 189, 248, 0.2);
      --glow-color: #38bdf8;
      --text-color: #f8fafc;
      --text-muted: #64748b;
      --accent-color: #6366f1;
    }
    body {
      background-color: var(--bg-color);
      color: var(--text-color);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 30px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      width: 100%;
      max-width: 750px;
    }
    header {
      text-align: center;
      margin-bottom: 40px;
    }
    h1 {
      font-size: 2.8rem;
      margin: 0;
      background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.25));
    }
    .subtitle {
      color: var(--text-muted);
      margin-top: 8px;
      font-size: 1.1rem;
    }
    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      padding: 32px;
      margin-bottom: 28px;
      backdrop-filter: blur(16px);
      box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    h2 {
      margin-top: 0;
      font-size: 1.4rem;
      color: var(--glow-color);
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 12px;
    }
    .form-group {
      margin-bottom: 24px;
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 8px;
      color: var(--text-color);
    }
    select {
      width: 100%;
      background-color: #0f172a;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      color: white;
      padding: 12px;
      font-size: 1rem;
      outline: none;
      cursor: pointer;
    }
    .btn {
      background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
      border: none;
      color: white;
      padding: 14px 28px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: bold;
      transition: all 0.2s ease;
      display: inline-block;
      text-decoration: none;
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
      text-align: center;
      width: 100%;
      box-sizing: border-box;
      font-size: 1rem;
    }
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 0 25px rgba(56, 189, 248, 0.45);
    }
    .btn-danger {
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.2);
    }
    .btn-danger:hover {
      box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    .file-input {
      display: none;
    }
    .file-label {
      border: 2px dashed var(--border-color);
      border-radius: 12px;
      padding: 30px;
      display: flex;
      flex-direction: column;
      align-items: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.4);
    }
    .file-label:hover {
      border-color: var(--glow-color);
      background: rgba(15, 23, 42, 0.6);
    }
    .file-label span {
      margin-top: 12px;
      color: var(--text-muted);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      padding: 16px;
      border-radius: 8px;
      text-align: center;
    }
    .metric-val {
      font-size: 1.8rem;
      font-weight: bold;
      color: var(--glow-color);
      margin-bottom: 4px;
    }
    .metric-label {
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
    }
    th, td {
      border-bottom: 1px solid var(--border-color);
      padding: 10px;
      text-align: left;
    }
    th {
      color: var(--glow-color);
    }
    pre {
      background-color: #020617;
      border: 1px solid var(--border-color);
      padding: 16px;
      border-radius: 8px;
      overflow-x: auto;
      color: #38bdf8;
      max-height: 250px;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Chroma-Train Studio</h1>
      <p class="subtitle">Cloud-Based Edge Neural Calibration Engine</p>
    </header>

    {% if not trained %}
    <div class="panel">
      <h2>Upload Dataset & Select AI Model</h2>
      <form action="/train" method="POST" enctype="multipart/form-data">
        <div class="form-group">
          <label class="file-label">
            <svg style="width: 54px; height: 54px; fill: var(--text-muted);" viewBox="0 0 24 24">
              <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
            </svg>
            <span id="file-name-display">Drop standards.csv here or click to upload</span>
            <input type="file" name="csv_file" class="file-input" id="csv_file" required onchange="updateFileName()">
          </label>
        </div>

        <div class="form-group">
          <label for="model_type">Regression Algorithm</label>
          <select name="model_type" id="model_type">
            <option value="linear">Multivariate Linear Regression</option>
            <option value="mlp">Multi-layer Perceptron (MLP) Neural Network</option>
          </select>
        </div>

        <button type="submit" class="btn">Execute Machine Learning Pipeline</button>
      </form>
    </div>
    {% else %}
    <div class="panel">
      <h2>Training Evaluation & Verification</h2>
      
      <div class="metrics">
        <div class="metric-card">
          <div class="metric-val">{{ r2 }}</div>
          <div class="metric-label">R&sup2; Score (Target: &gt;0.985)</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{{ mae }}</div>
          <div class="metric-label">Mean Absolute Error (MAE)</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{{ model_name }}</div>
          <div class="metric-label">Active Model Class</div>
        </div>
      </div>

      <h2>Actual vs. Predicted Molarity</h2>
      <table>
        <thead>
          <tr>
            <th>Sample #</th>
            <th>Actual Conc</th>
            <th>Predicted Conc</th>
            <th>Deviation</th>
          </tr>
        </thead>
        <tbody>
          {% for r in results_table %}
          <tr>
            <td>{{ r.idx }}</td>
            <td>{{ r.actual }}</td>
            <td>{{ r.pred }}</td>
            <td>{{ r.dev }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Step 3: Download Edge Parameters</h2>
      <p style="margin-bottom: 20px; color: var(--text-muted);">
        Save the model file below and upload it directly to your ESP-12S Control Panel web interface. No firmware compiles needed!
      </p>
      <a href="/download" class="btn" style="margin-bottom: 20px;">Download model_params.json</a>
      <a href="/" class="btn btn-danger" style="background: linear-gradient(135deg, #334155 0%, #1e293b 100%);">Train Another Model</a>

      <h2 style="margin-top: 30px;">Model Parameters Preview</h2>
      <pre><code>{{ params_preview }}</code></pre>
    </div>
    {% endif %}

    <div class="footer">
      KSEF 2026 Project 6 | Powered by Galvaniy Technologies
    </div>
  </div>

  <script>
    function updateFileName() {
      const fileInput = document.getElementById('csv_file');
      const nameDisplay = document.getElementById('file-name-display');
      if (fileInput.files.length > 0) {
        nameDisplay.textContent = fileInput.files[0].name;
      }
    }
  </script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
#  FLASK ROUTING LOGIC
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, trained=False)

@app.route('/train', methods=['POST'])
def train():
    if 'csv_file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['csv_file']
    model_type = request.form.get('model_type', 'linear')
    
    if file.filename == '':
        return "Empty file uploaded", 400
    
    # Load dataset
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Failed to parse CSV file: {e}", 400
    
    # Check headers
    required_cols = ['molarity', 'ambient', 'red', 'green', 'blue']
    for c in required_cols:
        if c not in df.columns:
            return f"Missing required column in CSV: {c}", 400
            
    # Find baseline row (where molarity == 0)
    blank_rows = df[df['molarity'] == 0.0]
    if blank_rows.empty:
        # Fallback: take minimum concentration as blank
        I0_row = df.loc[df['molarity'].idxmin()]
    else:
        I0_row = blank_rows.iloc[0]
        
    I0_r = float(I0_row['red'])
    I0_g = float(I0_row['green'])
    I0_b = float(I0_row['blue'])
    
    # Calculate absorbances
    # Protect against division by zero and negative logs
    df['abs_r'] = -np.log10(np.clip(df['red'] / max(I0_r, 1.0), 1e-5, 0.999))
    df['abs_g'] = -np.log10(np.clip(df['green'] / max(I0_g, 1.0), 1e-5, 0.999))
    df['abs_b'] = -np.log10(np.clip(df['blue'] / max(I0_b, 1.0), 1e-5, 0.999))
    
    # Prepare features and targets
    # Features: [Abs_Red, Abs_Green, Abs_Blue, Ambient]
    X = df[['abs_r', 'abs_g', 'abs_b', 'ambient']].values
    y = df['molarity'].values
    
    # Train selected model class
    params_dict = {}
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate scores
        y_pred = model.predict(X)
        
        # Format model_params
        params_dict = {
            "model_type": "linear",
            "w": [
                float(model.intercept_),
                float(model.coef_[0]),
                float(model.coef_[1]),
                float(model.coef_[2]),
                float(model.coef_[3])
            ]
        }
        model_name = "Linear Regression"
    else: # MLP
        # Fit a small MLP Regressor matching the C++ layout (1 hidden layer, 8 neurons, Relu)
        model = MLPRegressor(
            hidden_layer_sizes=(8,),
            activation='relu',
            solver='lbfgs',
            max_iter=3000,
            random_state=42
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Format MLP arrays to nested list format
        w1 = model.coefs_[0].T.tolist() # transpose to [8, 4]
        b1 = model.intercepts_[0].tolist() # [8]
        w2 = model.coefs_[1].flatten().tolist() # [8]
        b2 = float(model.intercepts_[1][0]) # float
        
        params_dict = {
            "model_type": "mlp",
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2
        }
        model_name = "Neural Network (MLP)"

    # Compute evaluation statistics
    r2_score = float(np.clip(1.0 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)), 0.0, 1.0))
    mae = float(np.mean(np.abs(y - y_pred)))
    
    # Save parameters to a temporary path
    with open('/tmp/model_params.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
        
    # Format a table for the HTML page
    results_table = []
    for idx, (act, prd) in enumerate(zip(y, y_pred)):
        dev = abs(act - prd) / max(act, 0.01) * 100.0
        results_table.append({
            "idx": idx + 1,
            "actual": f"{act:.4f}",
            "pred": f"{prd:.4f}",
            "dev": f"{dev:.1f}"
        })
        
    preview = json.dumps(params_dict, indent=2)
    
    return render_template_string(
        HTML_TEMPLATE,
        trained=True,
        r2=f"{r2_score:.4f}",
        mae=f"{mae:.4f}",
        model_name=model_name,
        results_table=results_table,
        params_preview=preview
    )

@app.route('/download')
def download():
    path = '/tmp/model_params.json'
    if not os.path.exists(path):
        return "Model not trained yet", 400
    return send_file(path, as_attachment=True, download_name='model_params.json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
