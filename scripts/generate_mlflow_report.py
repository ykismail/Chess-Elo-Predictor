"""
Generate Static MLflow Reports
===============================
Converts MLflow experiment tracking data to static HTML reports.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    import mlflow
except ImportError:
    mlflow = None


class StaticMLflowReportGenerator:
    """Generate static HTML reports from MLflow runs."""
    
    def __init__(self, project_root: str = project_dir):
        self.project_root = Path(project_root)
        self.mlruns_dir = self.project_root / "mlruns"
        self.dist_dir = self.project_root / "dist"
        self.mlflow_dir = self.dist_dir / "mlflow"
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)
    
    def load_mlflow_data(self) -> Dict:
        """Load MLflow runs from local mlruns directory."""
        data = {
            "experiments": [],
            "runs": [],
            "total_runs": 0,
            "models_registered": 0
        }
        
        if not self.mlruns_dir.exists():
            print("⚠ No mlruns directory found")
            return data
        
        try:
            if mlflow is None:
                print("⚠ MLflow not available, generating sample report")
                return data
            
            mlflow.set_tracking_uri(f"file:{self.mlruns_dir}")
            
            # Get all experiments
            experiments = mlflow.search_experiments()
            for exp in experiments:
                exp_data = {
                    "name": exp.name,
                    "id": exp.experiment_id,
                    "runs": []
                }
                
                # Get runs for this experiment
                try:
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                    for _, run in runs.iterrows():
                        run_info = {
                            "run_id": run.run_id,
                            "status": run.status,
                            "duration": run.get("end_time", 0) - run.get("start_time", 0) if "end_time" in run and "start_time" in run else 0,
                            "metrics": {}
                        }
                        
                        # Extract metrics
                        for col in runs.columns:
                            if col.startswith("metrics."):
                                metric_name = col.replace("metrics.", "")
                                run_info["metrics"][metric_name] = round(run[col], 4) if isinstance(run[col], float) else run[col]
                        
                        exp_data["runs"].append(run_info)
                        data["total_runs"] += 1
                
                except Exception as e:
                    print(f"⚠ Could not load runs for {exp.name}: {e}")
                
                if exp_data["runs"]:
                    data["experiments"].append(exp_data)
            
            print(f"✓ Loaded {data['total_runs']} MLflow runs from {len(data['experiments'])} experiments")
            
        except Exception as e:
            print(f"⚠ Error loading MLflow data: {e}")
        
        return data
    
    def generate_html(self, mlflow_data: Dict) -> str:
        """Generate HTML MLflow report."""
        
        experiments_html = ""
        for exp in mlflow_data["experiments"]:
            runs_html = ""
            best_accuracy = 0
            best_run_id = ""
            
            for run in exp["runs"]:
                accuracy = run["metrics"].get("accuracy", 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run_id = run["run_id"]
                
                metrics_html = "".join([
                    f'<tr><td>{k}</td><td>{v}</td></tr>'
                    for k, v in run["metrics"].items()
                ])
                
                best_badge = '<span class="badge badge-best">Best</span>' if run["run_id"] == best_run_id else ''
                
                runs_html += f"""
                <tr>
                    <td><code>{run['run_id'][:8]}...</code> {best_badge}</td>
                    <td>{run['status']}</td>
                    <td>
                        <table class="metrics-table">
                            {metrics_html}
                        </table>
                    </td>
                </tr>
                """
            
            if runs_html:
                experiments_html += f"""
                <div class="experiment-card">
                    <h3>{exp['name']}</h3>
                    <p class="exp-id">Experiment ID: {exp['id']}</p>
                    <p class="run-count">{len(exp['runs'])} runs</p>
                    <table class="runs-table">
                        <thead>
                            <tr>
                                <th>Run ID</th>
                                <th>Status</th>
                                <th>Metrics</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs_html}
                        </tbody>
                    </table>
                </div>
                """
        
        if not experiments_html:
            experiments_html = '<p style="color: #9ca3af;">No MLflow runs recorded yet. Run the training phase to track experiments.</p>'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLflow Experiments · Chess Elo Predictor</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d0f14 0%, #1a1d2e 100%);
            color: #e8e4dc;
            line-height: 1.6;
        }}
        
        header {{
            background: rgba(19, 22, 32, 0.95);
            border-bottom: 1px solid #1e2230;
            padding: 2rem;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #c9a227 0%, #e8e4dc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat {{
            background: rgba(19, 22, 32, 0.8);
            border: 1px solid #1e2230;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .stat h3 {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        
        .stat .value {{
            font-size: 2.5rem;
            color: #c9a227;
            font-weight: bold;
        }}
        
        section {{
            margin-bottom: 3rem;
        }}
        
        section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: #c9c3b8;
            padding-bottom: 1rem;
            border-bottom: 2px solid #1e2230;
        }}
        
        .experiment-card {{
            background: rgba(19, 22, 32, 0.8);
            border: 1px solid #1e2230;
            border-left: 3px solid #c9a227;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }}
        
        .experiment-card h3 {{
            color: #c9a227;
            margin-bottom: 0.5rem;
        }}
        
        .exp-id {{
            color: #7c8a9e;
            font-size: 0.9rem;
        }}
        
        .run-count {{
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}
        
        .runs-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .runs-table thead {{
            background: rgba(0, 0, 0, 0.3);
            border-bottom: 2px solid #1e2230;
        }}
        
        .runs-table th {{
            padding: 0.75rem;
            text-align: left;
            color: #c9a227;
            font-weight: 600;
        }}
        
        .runs-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid #1e2230;
            color: #9ca3af;
        }}
        
        .runs-table tbody tr:hover {{
            background: rgba(201, 162, 39, 0.05);
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}
        
        .metrics-table td {{
            padding: 0.3rem 0.5rem;
            border: none;
            color: #9ca3af;
        }}
        
        .metrics-table tr:nth-child(odd) {{
            background: rgba(0, 0, 0, 0.2);
        }}
        
        code {{
            background: rgba(0, 0, 0, 0.3);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #c9a227;
        }}
        
        .badge {{
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            margin-left: 0.5rem;
        }}
        
        .badge-best {{
            background: rgba(90, 158, 111, 0.2);
            color: #5a9e6f;
            border: 1px solid #5a9e6f;
        }}
        
        .info-box {{
            background: rgba(201, 162, 39, 0.1);
            border: 1px solid #c9a227;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .info-box h3 {{
            color: #c9a227;
            margin-bottom: 0.5rem;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.9rem;
            border-top: 1px solid #1e2230;
            margin-top: 3rem;
        }}
        
        .nav-links {{
            text-align: center;
            margin: 2rem 0;
        }}
        
        .nav-links a {{
            color: #c9a227;
            text-decoration: none;
            margin: 0 1rem;
        }}
        
        .nav-links a:hover {{
            text-decoration: underline;
        }}
        
        @media (max-width: 768px) {{
            header h1 {{ font-size: 1.5rem; }}
            .container {{ padding: 1rem; }}
            .stat .value {{ font-size: 2rem; }}
            .runs-table {{ font-size: 0.8rem; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 MLflow Experiments</h1>
        <p>Model Training & Experiment Tracking</p>
    </header>
    
    <div class="container">
        <div class="nav-links">
            <a href="../index.html">← Back to Dashboard</a>
        </div>
        
        <section>
            <h2>Experiment Summary</h2>
            <div class="stats">
                <div class="stat">
                    <h3>Total Runs</h3>
                    <div class="value">{mlflow_data['total_runs']}</div>
                </div>
                <div class="stat">
                    <h3>Experiments</h3>
                    <div class="value">{len(mlflow_data['experiments'])}</div>
                </div>
                <div class="stat">
                    <h3>Status</h3>
                    <div class="value" style="font-size: 1.5rem;">✓</div>
                </div>
                <div class="stat">
                    <h3>Last Updated</h3>
                    <div class="value" style="font-size: 1rem;">{datetime.now().strftime('%H:%M')}</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Experiments & Runs</h2>
            {experiments_html}
        </section>
        
        <section>
            <h2>How to Integrate MLflow</h2>
            <div class="info-box">
                <h3>Local MLflow Server</h3>
                <p>Run locally for interactive dashboards:</p>
                <code style="display: block; margin-top: 0.5rem;">mlflow ui --backend-store-uri file:./mlruns</code>
                <p style="margin-top: 0.5rem; color: #9ca3af;">Opens at: http://localhost:5000</p>
            </div>
            <div class="info-box">
                <h3>Tracking Your Experiments</h3>
                <p>In your training code:</p>
                <code style="display: block; margin-top: 0.5rem;">
with mlflow.start_run():<br/>
&nbsp;&nbsp;&nbsp;&nbsp;mlflow.log_metrics({{'accuracy': 0.95}})<br/>
&nbsp;&nbsp;&nbsp;&nbsp;mlflow.log_params({{'lr': 0.001}})<br/>
&nbsp;&nbsp;&nbsp;&nbsp;mlflow.sklearn.log_model(model, 'model')
                </code>
            </div>
        </section>
    </div>
    
    <footer>
        <p>Chess Elo Predictor · MLflow Experiment Tracking</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
</body>
</html>
"""
        return html
    
    def write_html(self, html: str) -> Path:
        """Write HTML to file."""
        output_path = self.mlflow_dir / "index.html"
        output_path.write_text(html)
        print(f"✓ Generated {output_path}")
        return output_path
    
    def generate(self):
        """Generate complete MLflow report."""
        print("\n" + "="*70)
        print("GENERATING STATIC MLFLOW REPORTS")
        print("="*70 + "\n")
        
        mlflow_data = self.load_mlflow_data()
        html = self.generate_html(mlflow_data)
        output_path = self.write_html(html)
        
        print(f"\n✓ MLflow report ready: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"  Deploy to GitHub Pages: dist/mlflow/")
        print("\n" + "="*70)


if __name__ == "__main__":
    generator = StaticMLflowReportGenerator()
    generator.generate()
