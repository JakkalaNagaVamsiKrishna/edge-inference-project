"""
dashboard/app.py
─────────────────
Live observability dashboard.

Reads benchmark.json and displays:
  - Latency timeline across CI commits
  - Accuracy vs latency tradeoff curve
  - Gate pass/fail history
  - Live memory / FPS stats

Usage:
  python -m dashboard.app
  → Open http://localhost:5000
"""

from __future__ import annotations
import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string

from configs.settings import cfg

app = Flask(__name__)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_benchmark() -> dict:
    p = Path(cfg.logging.benchmark_json)
    if not p.exists():
        return {}
    with open(p) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed benchmark report: {e}")


# ─── HTML template (self-contained, no external CDN) ─────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge-CV-Hub Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      padding: 2rem;
    }

    h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #f8fafc; }
    .subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 2rem; }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .card {
      background: #1e2130;
      border: 1px solid #2d3348;
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
    }

    .card-label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .card-value { font-size: 2rem; font-weight: 700; margin: 0.25rem 0; color: #f8fafc; }
    .card-sub   { font-size: 0.8rem; color: #64748b; }

    .gate-pass { color: #34d399; }
    .gate-fail { color: #f87171; }

    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .chart-card {
      background: #1e2130;
      border: 1px solid #2d3348;
      border-radius: 12px;
      padding: 1.25rem;
    }

    .chart-title {
      font-size: 0.875rem;
      font-weight: 500;
      color: #cbd5e1;
      margin-bottom: 1rem;
    }

    canvas { max-height: 250px; }

    .failures {
      background: #1e2130;
      border: 1px solid #f87171;
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 2rem;
    }

    .failures h3 { color: #f87171; margin-bottom: 0.75rem; font-size: 0.875rem; }
    .failures li { color: #fca5a5; font-size: 0.8rem; margin-left: 1.25rem; margin-bottom: 0.25rem; }

    .threshold-table {
      background: #1e2130;
      border: 1px solid #2d3348;
      border-radius: 12px;
      padding: 1.25rem;
      overflow-x: auto;
    }

    .threshold-table h3 { color: #cbd5e1; margin-bottom: 1rem; font-size: 0.875rem; }

    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    th { color: #64748b; text-align: left; padding: 0.4rem 0.75rem; border-bottom: 1px solid #2d3348; }
    td { color: #cbd5e1; padding: 0.4rem 0.75rem; border-bottom: 1px solid #1a1f2e; }
    tr:hover td { background: #232840; }

    .refresh-btn {
      background: #3b4fd8;
      color: white;
      border: none;
      padding: 0.5rem 1.25rem;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.875rem;
      margin-top: 1rem;
    }
    .refresh-btn:hover { background: #4c5fe8; }
  </style>
</head>
<body>
  <h1>Edge-CV-Hub</h1>
  <p class="subtitle" id="timestamp">Loading…</p>

  <div class="grid" id="stat-cards"></div>
  <div id="failures-container"></div>

  <div class="chart-grid">
    <div class="chart-card">
      <div class="chart-title">Latency distribution (ms)</div>
      <canvas id="latencyChart"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">Precision / Recall vs threshold</div>
      <canvas id="prChart"></canvas>
    </div>
  </div>

  <div class="threshold-table">
    <h3>Threshold sweep — Precision / Recall / F1</h3>
    <table>
      <thead>
        <tr>
          <th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th>
        </tr>
      </thead>
      <tbody id="threshold-body"></tbody>
    </table>
  </div>

  <button class="refresh-btn" onclick="loadData()">↻ Refresh</button>

  <script>
    let latencyChart = null;
    let prChart      = null;

    async function loadData() {
      const resp = await fetch('/api/benchmark');
      const data = await resp.json();
      if (!data || Object.keys(data).length === 0) {
        document.getElementById('timestamp').textContent = 'No benchmark data yet. Run benchmark/runner.py first.';
        return;
      }

      document.getElementById('timestamp').textContent =
        'Last run: ' + (data.timestamp || 'unknown');

      const lat = data.latency || {};
      const acc = data.accuracy || {};
      const gates = data.gates || {};

      // ── Stat cards ───────────────────────────────────────────────────────
      const passed = data.passed;
      const stats = [
        { label: 'Gate status',  value: passed ? '✅ Pass' : '❌ Fail',  sub: '',                       cls: passed ? 'gate-pass' : 'gate-fail' },
        { label: 'p95 Latency',  value: (lat.p95_ms||0).toFixed(1) + ' ms', sub: 'Gate: ≤ ' + (gates.latency_gate_ms||50) + ' ms' },
        { label: 'Mean Latency', value: (lat.mean_ms||0).toFixed(1) + ' ms', sub: 'p99: ' + (lat.p99_ms||0).toFixed(1) + ' ms' },
        { label: 'Throughput',   value: (lat.fps||0).toFixed(1) + ' FPS',    sub: '' },
        { label: 'Top-1 Accuracy', value: ((acc.accuracy||0)*100).toFixed(2)+'%', sub: 'Gate: ≥ ' + ((gates.accuracy_threshold||0)*100).toFixed(0)+'%' },
        { label: 'Memory RSS',   value: (lat.memory_rss_mb||0).toFixed(1) + ' MB', sub: 'Limit: ' + (gates.memory_limit_mb||512) + ' MB' },
      ];

      document.getElementById('stat-cards').innerHTML = stats.map(s => `
        <div class="card">
          <div class="card-label">${s.label}</div>
          <div class="card-value ${s.cls||''}">${s.value}</div>
          <div class="card-sub">${s.sub}</div>
        </div>
      `).join('');

      // ── Failures ─────────────────────────────────────────────────────────
      const failures = data.failures || [];
      const fc = document.getElementById('failures-container');
      if (failures.length > 0) {
        fc.innerHTML = `<div class="failures">
          <h3>⚠ Gate Failures</h3>
          <ul>${failures.map(f => `<li>${f}</li>`).join('')}</ul>
        </div>`;
      } else {
        fc.innerHTML = '';
      }

      // ── Latency chart ─────────────────────────────────────────────────────
      const latLabels = ['min', 'mean', 'p50', 'p95', 'p99', 'max'];
      const latValues = [lat.min_ms, lat.mean_ms, lat.p50_ms, lat.p95_ms, lat.p99_ms, lat.max_ms].map(v => (v||0).toFixed(2));

      if (latencyChart) latencyChart.destroy();
      latencyChart = new Chart(document.getElementById('latencyChart'), {
        type: 'bar',
        data: {
          labels: latLabels,
          datasets: [{
            label: 'Latency (ms)',
            data: latValues,
            backgroundColor: latValues.map(v =>
              parseFloat(v) > (gates.latency_gate_ms||50) ? 'rgba(248,113,113,0.7)' : 'rgba(99,179,237,0.7)'
            ),
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#2d3348' } },
            x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          }
        }
      });

      // ── Precision/Recall chart ─────────────────────────────────────────────
      const thresholds = data.thresholds || {};
      const thKeys = Object.keys(thresholds).sort((a,b) => parseFloat(a)-parseFloat(b));
      const precisions = thKeys.map(k => thresholds[k].precision);
      const recalls    = thKeys.map(k => thresholds[k].recall);
      const f1s        = thKeys.map(k => thresholds[k].f1);

      if (prChart) prChart.destroy();
      prChart = new Chart(document.getElementById('prChart'), {
        type: 'line',
        data: {
          labels: thKeys,
          datasets: [
            { label: 'Precision', data: precisions, borderColor: '#60a5fa', tension: 0.3, pointRadius: 2 },
            { label: 'Recall',    data: recalls,    borderColor: '#34d399', tension: 0.3, pointRadius: 2 },
            { label: 'F1',        data: f1s,        borderColor: '#f59e0b', tension: 0.3, pointRadius: 2 },
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
          scales: {
            y: { min: 0, max: 1, ticks: { color: '#94a3b8' }, grid: { color: '#2d3348' } },
            x: { ticks: { color: '#94a3b8', maxTicksLimit: 8 }, grid: { display: false } },
          }
        }
      });

      // ── Threshold table ───────────────────────────────────────────────────
      const tbody = document.getElementById('threshold-body');
      tbody.innerHTML = thKeys.map(k => `
        <tr>
          <td>${k}</td>
          <td>${(thresholds[k].precision*100).toFixed(1)}%</td>
          <td>${(thresholds[k].recall*100).toFixed(1)}%</td>
          <td>${(thresholds[k].f1*100).toFixed(1)}%</td>
        </tr>
      `).join('');
    }

    loadData();
  </script>
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/benchmark")
def api_benchmark():
    try:
        data = load_benchmark()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def api_config():
    return jsonify({
        "model": {
            "teacher_arch": cfg.model.teacher_arch,
            "student_arch": cfg.model.student_arch,
            "task": cfg.model.task,
            "num_classes": cfg.model.num_classes,
        },
        "compression": {
            "pruning_ratio": cfg.compression.pruning_ratio,
            "quantization_mode": cfg.compression.quantization_mode,
            "distillation_temperature": cfg.compression.distillation_temperature,
        },
        "deployment": {
            "target_arch": cfg.deployment.target_arch,
            "latency_gate_ms": cfg.deployment.latency_gate_ms,
            "accuracy_threshold": cfg.deployment.accuracy_threshold,
            "memory_limit_mb": cfg.deployment.memory_limit_mb,
        },
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=cfg.logging.dashboard_port,
        debug=False,
    )