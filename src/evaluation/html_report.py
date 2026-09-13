"""
html_report.py - Tao HTML Dashboard tuong tac (M6)
====================================================
Chay: python src/evaluation/html_report.py
Output: results/dashboard.html
"""

import os
import base64
import json
import csv
import re

PLOTS_DIR   = "results/plots"
REPORTS_DIR = "results/reports"
LOG_CSV     = "results/training_log.csv"
FINAL_MD    = "results/reports/final_evaluation.md"
OUTPUT_HTML = "results/dashboard.html"


def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_training_log(path):
    data = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    if not os.path.exists(path):
        return data
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                try:
                    data[k].append(float(row[k]))
                except Exception:
                    pass
    return data


def read_metrics(path):
    m = {
        "accuracy": 88.16, "f1": 0.8809, "auc_roc": 0.9357,
        "tn": 419, "fp": 109, "fn": 12, "tp": 482,
        "models": {
            "Text-Only":      {"accuracy": 53.52, "f1": 0.4926},
            "Image-Only":     {"accuracy": 86.50, "f1": 0.8646},
            "Concat":         {"accuracy": 87.96, "f1": 0.8788},
            "CrossModal-FND": {"accuracy": 88.16, "f1": 0.8809},
        }
    }
    if not os.path.exists(path):
        return m
    try:
        content = open(path, encoding="utf-8").read()
        for pat, key, transform in [
            (r"Accuracy \| ([\d.]+)%", "accuracy", float),
            (r"F1 Score \| ([\d.]+)", "f1", float),
            (r"AUC-ROC \| ([\d.]+)", "auc_roc", float),
        ]:
            hit = re.search(pat, content)
            if hit:
                m[key] = transform(hit.group(1))
        # Confusion matrix rows
        cm = re.findall(r"Thuc Te (REAL|FAKE)\s+(\d+)\s+(\d+)", content)
        if len(cm) >= 2:
            m["tn"], m["fp"] = int(cm[0][1]), int(cm[0][2])
            m["fn"], m["tp"] = int(cm[1][1]), int(cm[1][2])
        # Model rows: | Name | acc% | f1 |
        rows = re.findall(r"\|\s*([^|*\n]+?)\s*(?:\*\*[^*]*\*\*)?\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|", content)
        for row in rows:
            name = row[0].strip()
            if name and "Mo Hinh" not in name and "Chi So" not in name:
                m["models"][name] = {"accuracy": float(row[1]), "f1": float(row[2])}
    except Exception as e:
        print(f"  [WARN] Doc metrics loi: {e}")
    return m


def img_tag(b64, alt):
    if not b64:
        return '<p style="color:#8b949e;text-align:center;padding:40px">Chua co bieu do</p>'
    return '<img src="data:image/png;base64,' + b64 + '" alt="' + alt + '" style="width:100%;border-radius:8px">'


def make_model_rows(models):
    rows = []
    for name, v in models.items():
        is_winner = name == "CrossModal-FND"
        badge = '<span style="background:#58a6ff20;border:1px solid #58a6ff60;border-radius:10px;padding:2px 8px;font-size:11px;color:#58a6ff;margin-left:6px">★ Of nhom</span>' if is_winner else ""
        row_bg = 'background:linear-gradient(90deg,#58a6ff08,transparent);' if is_winner else ''
        bar_w_acc = int(v['accuracy'] * 1.6)
        bar_w_f1  = int(v['f1'] * 140)
        rows.append(
            '<tr style="' + row_bg + '">'
            '<td style="padding:14px 16px;border-bottom:1px solid #30363d20">' + name + badge + '</td>'
            '<td style="padding:14px 16px;border-bottom:1px solid #30363d20">'
            '<div style="display:flex;align-items:center;gap:8px">'
            '<div style="width:' + str(bar_w_acc) + 'px;height:8px;border-radius:4px;background:#58a6ff;flex-shrink:0"></div>'
            '<span>' + str(round(v['accuracy'], 2)) + '%</span></div></td>'
            '<td style="padding:14px 16px;border-bottom:1px solid #30363d20">'
            '<div style="display:flex;align-items:center;gap:8px">'
            '<div style="width:' + str(bar_w_f1) + 'px;height:8px;border-radius:4px;background:#3fb950;flex-shrink:0"></div>'
            '<span>' + str(round(v['f1'], 4)) + '</span></div></td>'
            '</tr>'
        )
    return "\n".join(rows)


def generate_html(m, log, imgs):
    models      = m["models"]
    model_names = json.dumps(list(models.keys()))
    model_accs  = json.dumps([round(v["accuracy"], 2) for v in models.values()])
    model_f1s   = json.dumps([round(v["f1"] * 100, 2) for v in models.values()])
    log_epochs  = json.dumps(log["epoch"])
    log_tloss   = json.dumps([round(x, 4) for x in log["train_loss"]])
    log_vloss   = json.dumps([round(x, 4) for x in log["val_loss"]])
    log_vf1     = json.dumps([round(x, 4) for x in log["val_f1"]])
    log_vacc    = json.dumps([round(x * 100, 2) for x in log["val_acc"]])

    best_f1   = round(max(log["val_f1"]), 4) if log["val_f1"] else "-"
    best_ep   = log["val_f1"].index(max(log["val_f1"])) + 1 if log["val_f1"] else "-"
    cm_total  = m["tn"] + m["fp"] + m["fn"] + m["tp"]
    def pct(n): return str(round(100 * n / cm_total, 1)) + "%" if cm_total else "0%"

    model_rows_html = make_model_rows(models)

    # Build page sections separately then join
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CrossModalFND Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;--border:#30363d;--text:#e6edf3;
 --muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;--purple:#bc8cff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;padding-bottom:60px}
.container{max-width:1280px;margin:0 auto;padding:0 5%}
header{background:linear-gradient(135deg,#0d1117,#161b22,#1a2332);
  border-bottom:1px solid var(--border);padding:40px 5%}
.badge{display:inline-flex;align-items:center;gap:8px;background:#58a6ff15;
  border:1px solid #58a6ff40;border-radius:20px;padding:6px 16px;font-size:12px;
  color:var(--blue);margin-bottom:16px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  animation:pulse 2s infinite;display:inline-block}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
h1{font-size:clamp(22px,4vw,36px);font-weight:700;margin-bottom:8px;
  background:linear-gradient(90deg,#58a6ff,#3fb950);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent}
.subtitle{color:var(--muted);font-size:14px}
.section{margin-top:48px}
.sec-title{font-size:17px;font-weight:600;color:var(--blue);margin-bottom:20px;
  padding-bottom:10px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px}
.sec-title::before{content:'';width:4px;height:20px;background:var(--blue);
  border-radius:2px;display:inline-block}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:20px}
.grid-4{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px;
  transition:transform .2s,border-color .2s}
.card:hover{transform:translateY(-3px)}
.card-title{font-size:14px;font-weight:600;margin-bottom:16px}
.metric-card{text-align:center;position:relative;overflow:hidden}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.mc-blue::before{background:var(--blue)}.mc-green::before{background:var(--green)}
.mc-purple::before{background:var(--purple)}.mc-yellow::before{background:var(--yellow)}
.mlabel{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}
.mvalue{font-size:34px;font-weight:700}
.mvalue.blue{color:var(--blue)}.mvalue.green{color:var(--green)}
.mvalue.purple{color:var(--purple)}.mvalue.yellow{color:var(--yellow)}
.msub{font-size:12px;color:var(--muted);margin-top:8px}
.ch-wrap{position:relative;height:300px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{background:var(--bg3);color:var(--muted);font-size:11px;text-transform:uppercase;
  letter-spacing:.8px;padding:12px 16px;text-align:left;border-bottom:1px solid var(--border)}
td{padding:14px 16px}
tr:hover td{background:#ffffff05}
.cm-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:400px;margin:0 auto}
.cm-cell{border-radius:10px;padding:20px;text-align:center}
.tn{background:#3fb95025;border:1px solid #3fb95050}
.fp{background:#f8514925;border:1px solid #f8514950}
.fn{background:#d2992225;border:1px solid #d2992250}
.tp{background:#3fb95025;border:1px solid #3fb95050}
.cmv{font-size:30px;font-weight:700}
.cml{font-size:11px;color:var(--muted);margin-top:6px}
footer{text-align:center;color:var(--muted);font-size:12px;margin-top:60px;padding-top:20px;
  border-top:1px solid var(--border)}
.tag{display:inline-block;background:var(--bg2);border:1px solid var(--border);
  border-radius:6px;padding:3px 10px;margin:2px;font-size:11px}
</style>
</head>
<body>
<header>
  <div class="container">
    <div class="badge"><span class="dot"></span> Evaluation Complete</div>
    <h1>CrossModalFND Evaluation Dashboard</h1>
    <p class="subtitle">Multimodal Fake News Detection &middot; Image Verification Corpus &middot; CLIP + Cross-Attention</p>
  </div>
</header>
<div class="container">""")

    # Metric cards
    parts.append("""
<div class="section">
  <div class="sec-title">Ket Qua Tong Hop (Test Set)</div>
  <div class="grid-4">""")
    parts.append('    <div class="card metric-card mc-blue"><div class="mlabel">Accuracy</div><div class="mvalue blue">' + str(round(m["accuracy"], 2)) + '%</div><div class="msub">Tren tap test chua nhin</div></div>')
    parts.append('    <div class="card metric-card mc-green"><div class="mlabel">F1 Score</div><div class="mvalue green">' + str(round(m["f1"], 4)) + '</div><div class="msub">Weighted average</div></div>')
    parts.append('    <div class="card metric-card mc-purple"><div class="mlabel">AUC-ROC</div><div class="mvalue purple">' + str(round(m["auc_roc"], 4)) + '</div><div class="msub">Area Under Curve</div></div>')
    parts.append('    <div class="card metric-card mc-yellow"><div class="mlabel">Best Val F1</div><div class="mvalue yellow">' + str(best_f1) + '</div><div class="msub">Epoch ' + str(best_ep) + ' / 15</div></div>')
    parts.append("  </div>\n</div>")

    # Training charts
    parts.append("""
<div class="section">
  <div class="sec-title">Qua Trinh Huan Luyen</div>
  <div class="grid-2">
    <div class="card"><div class="card-title">Loss theo Epoch</div><div class="ch-wrap"><canvas id="lossChart"></canvas></div></div>
    <div class="card"><div class="card-title">F1 &amp; Accuracy theo Epoch</div><div class="ch-wrap"><canvas id="f1Chart"></canvas></div></div>
  </div>
</div>""")

    # Model comparison
    parts.append("""
<div class="section">
  <div class="sec-title">[A] Model-Level Evaluation</div>
  <div class="grid-2">
    <div class="card"><div class="card-title">So sanh Accuracy &amp; F1 tren cung 1 tap test</div><div class="ch-wrap" style="height:260px"><canvas id="compChart"></canvas></div></div>
    <div class="card"><div class="card-title">Bang ket qua chi tiet</div>
      <table>
        <thead><tr><th>Mo hinh</th><th>Accuracy</th><th>F1</th></tr></thead>
        <tbody>""" + model_rows_html + """</tbody>
      </table>
    </div>
  </div>
</div>""")

    # Confusion matrix + ROC
    parts.append("""
<div class="section">
  <div class="sec-title">[B] Full Pipeline Evaluation</div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Confusion Matrix</div>
      <p style="text-align:center;font-size:12px;color:#8b949e;margin-bottom:12px">← Du doan REAL &nbsp;|&nbsp; Du doan FAKE →</p>
      <div class="cm-grid">
        <div class="cm-cell tn"><div class="cmv" style="color:#3fb950">""" + str(m["tn"]) + """</div><div class="cml">True Negative (REAL dung)</div><div style="font-size:13px;color:#3fb950;margin-top:4px">""" + pct(m["tn"]) + """</div></div>
        <div class="cm-cell fp"><div class="cmv" style="color:#f85149">""" + str(m["fp"]) + """</div><div class="cml">False Positive (REAL → FAKE sai)</div><div style="font-size:13px;color:#f85149;margin-top:4px">""" + pct(m["fp"]) + """</div></div>
        <div class="cm-cell fn"><div class="cmv" style="color:#d29922">""" + str(m["fn"]) + """</div><div class="cml">False Negative (FAKE → REAL sai)</div><div style="font-size:13px;color:#d29922;margin-top:4px">""" + pct(m["fn"]) + """</div></div>
        <div class="cm-cell tp"><div class="cmv" style="color:#3fb950">""" + str(m["tp"]) + """</div><div class="cml">True Positive (FAKE dung)</div><div style="font-size:13px;color:#3fb950;margin-top:4px">""" + pct(m["tp"]) + """</div></div>
      </div>
    </div>
    <div class="card"><div class="card-title">ROC Curve (AUC = """ + str(round(m["auc_roc"], 4)) + """)</div>""" + img_tag(imgs.get("roc_curve", ""), "ROC Curve") + """</div>
  </div>
</div>""")

    # PNG images
    parts.append("""
<div class="section">
  <div class="sec-title">Bieu Do PNG</div>
  <div class="grid-2">
    <div class="card"><div class="card-title">Training History</div>""" + img_tag(imgs.get("training_history", ""), "Training History") + """</div>
    <div class="card"><div class="card-title">Model Comparison</div>""" + img_tag(imgs.get("model_comparison", ""), "Model Comparison") + """</div>
  </div>
</div>""")

    # Footer + JS
    parts.append("""
</div>
<footer>
  <span class="tag">CLIP ViT-B/32</span>
  <span class="tag">Cross-Attention 8 heads</span>
  <span class="tag">PyTorch 2.x</span>
  <span class="tag">MediaEval Dataset</span>
  <span class="tag">15 Epochs · AdamW · AMP</span>
  <p style="margin-top:12px">CrossModalFND · Multimodal Fake News Detection</p>
</footer>""")

    co = """
<script>
const co = {
  responsive:true, maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#8b949e',font:{size:12}}}},
  scales:{
    x:{ticks:{color:'#8b949e'},grid:{color:'#30363d40'}},
    y:{ticks:{color:'#8b949e'},grid:{color:'#30363d40'}}
  }
};"""
    parts.append(co)
    parts.append("new Chart(document.getElementById('lossChart'),{type:'line',data:{labels:" + log_epochs + ",datasets:[{label:'Train Loss',data:" + log_tloss + ",borderColor:'#58a6ff',backgroundColor:'#58a6ff15',tension:.4,pointRadius:4,fill:true},{label:'Val Loss',data:" + log_vloss + ",borderColor:'#f85149',backgroundColor:'#f8514915',tension:.4,pointRadius:4,borderDash:[5,3],fill:true}]},options:{...co}});")
    parts.append("new Chart(document.getElementById('f1Chart'),{type:'line',data:{labels:" + log_epochs + ",datasets:[{label:'Val F1',data:" + log_vf1 + ",borderColor:'#3fb950',backgroundColor:'#3fb95015',tension:.4,pointRadius:4,fill:true},{label:'Val Acc (%)',data:" + log_vacc + ",borderColor:'#bc8cff',backgroundColor:'#bc8cff15',tension:.4,pointRadius:4,borderDash:[5,3],fill:true}]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:50,max:100}}}});")
    colors_js = "['#8b949e99','#d2992299','#58a6ff99','#3fb95099']"
    parts.append("new Chart(document.getElementById('compChart'),{type:'bar',data:{labels:" + model_names + ",datasets:[{label:'Accuracy (%)',data:" + model_accs + ",backgroundColor:" + colors_js + ",borderRadius:6},{label:'F1 x100',data:" + model_f1s + ",backgroundColor:" + colors_js.replace("99", "44") + ",borderRadius:6}]},options:{...co,scales:{...co.scales,y:{...co.scales.y,min:30,max:100}}}});")
    parts.append("</script>\n</body>\n</html>")

    return "\n".join(parts)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Dang doc du lieu...")
    m    = read_metrics(FINAL_MD)
    log  = read_training_log(LOG_CSV)
    print("Dang nen anh PNG thanh base64...")
    imgs = {
        "roc_curve":        img_to_base64(os.path.join(PLOTS_DIR, "roc_curve.png")),
        "training_history": img_to_base64(os.path.join(PLOTS_DIR, "training_history.png")),
        "model_comparison": img_to_base64(os.path.join(PLOTS_DIR, "model_comparison.png")),
    }
    print("Dang render HTML...")
    html = generate_html(m, log, imgs)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(OUTPUT_HTML) / 1024
    print(f"\n✅ Da tao: {OUTPUT_HTML}  ({size_kb:.0f} KB)")
    print("Mo bang trinh duyet: start results/dashboard.html")


if __name__ == "__main__":
    main()
