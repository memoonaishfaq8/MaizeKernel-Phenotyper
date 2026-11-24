# app.py — About + App both styled
import gradio as gr
import os, tempfile, shutil, traceback
import cv2
import numpy as np
import pandas as pd
from pipeline import load_models, test_user_image

CUSTOM_CSS = r""":root{
  --bg:#050815; --bg2:#07112a;
  --stroke: rgba(255,255,255,0.10);
  --muted: rgba(255,255,255,0.72);
  --muted2: rgba(255,255,255,0.52);
  --p:#9b6cff; --pk:#ff79c6;
  --r:16px; --r2:22px;
  --shadow2: 0 14px 36px rgba(0,0,0,0.30);
  --theme-grad: linear-gradient(90deg, #9b6cff 0%, #ff79c6 100%);
  --btn-grad: linear-gradient(135deg, #9b6cff 0%, #ff79c6 100%);
  --btn-shadow: 0 16px 44px rgba(155,108,255,0.22), 0 16px 44px rgba(255,121,198,0.16);
  --good:#34d399;
  --warn:#fbbf24;
  --bad:#fb7185;
}
/* Kill orange-ish theme vars (some gradio builds) */
:root{
  --primary-600: #9b6cff !important;
  --primary-500: #b58cff !important;
  --primary-400: #ff79c6 !important;
  --button-primary-background-fill: transparent !important;
  --button-primary-background-fill-hover: transparent !important;
  --button-secondary-background-fill: transparent !important;
  --button-secondary-background-fill-hover: transparent !important;
}
html, body, .gradio-container{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
  color: #eef3ff;
  background:
    radial-gradient(900px 620px at 10% 10%, rgba(155,108,255,0.12), transparent 35%),
    radial-gradient(860px 560px at 90% 20%, rgba(255,121,198,0.11), transparent 32%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  background-attachment: fixed;
}
.gradio-container .wrap{ max-width: 1240px !important; margin: 18px auto; }
/* Topbar */
.webTopbar{
  border-radius: var(--r2);
  padding: 14px 16px;
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--stroke);
  box-shadow: var(--shadow2);
  display:flex; align-items:center; justify-content: space-between;
  gap: 12px; flex-wrap: wrap;
  margin-bottom: 16px;
}.webBrand{ display:flex; align-items:center; gap: 12px; }.webLogo{
  width: 44px; height: 44px; border-radius: 16px;
  background: var(--btn-grad);
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: var(--btn-shadow);
  display:grid; place-items:center;}
/* ---------- ABOUT PAGE STYLES ---------- */
.aboutPage{
  border-radius: var(--r2);
  padding: 22px 22px 18px;
  background: radial-gradient(120% 160% at 0% 0%, rgba(155,108,255,0.16), transparent 55%)
              ,radial-gradient(120% 160% at 100% 0%, rgba(255,121,198,0.15), transparent 55%)
              ,rgba(5,8,21,0.96);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 22px 60px rgba(0,0,0,0.45);
  margin-top: 10px;
}.aboutHeader{
  display:flex;
  flex-wrap:wrap;
  justify-content:space-between;
  gap:14px;
  align-items:flex-start;
  margin-bottom:18px;
}.aboutTitleBlock{
  max-width: 640px;
}.aboutTitle{
  font-size: 32px;
  font-weight: 950;
  letter-spacing:-0.03em;
  margin:0 0 6px;
  background: var(--theme-grad);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}.aboutSubtitle{
  margin:0;
  font-size: 14px;
  color: rgba(255,255,255,0.78);
  line-height:1.6;
}.aboutMeta{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}.miniTag{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:6px 11px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.18);
  background: rgba(2,6,23,0.78);
  font-size: 11.5px;
  font-weight: 800;
  color: rgba(255,255,255,0.90);
}.miniDot{
  width:8px;height:8px;border-radius:999px;
  background:var(--btn-grad);
  box-shadow:0 0 10px rgba(155,108,255,0.6);}
/* Grid layout */
.aboutGrid{
  display:grid;
  grid-template-columns: 1.3fr 1fr;
  gap:14px;
}@media (max-width: 980px){
  .aboutGrid{ grid-template-columns:1fr; }}
.aboutCard{
  border-radius:18px;
  padding:14px 14px 13px;
  background: rgba(10,16,40,0.94);
  border:1px solid rgba(255,255,255,0.12);
  box-shadow:0 14px 32px rgba(0,0,0,0.40);
}.aboutH{
  font-weight:950;
  letter-spacing:.14em;
  font-size:11px;
  color:rgba(255,255,255,0.76);
  text-transform:uppercase;
  margin:0 0 10px;
}.aboutP{
  margin:0;
  color:rgba(255,255,255,0.82);
  line-height:1.65;
  font-size:13.5px;
}.bullets{
  margin:0;
  padding-left:18px;
  color:rgba(255,255,255,0.82);
  line-height:1.7;
  font-size:13.2px;
}.kv{
  display:grid;
  grid-template-columns:1fr;
  gap:10px;
}.kvItem{
  border-radius:14px;
  padding:10px 11px;
  border:1px solid rgba(255,255,255,0.12);
  background:linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,64,175,0.58));
}.kvK{
  font-size:11px;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:rgba(226,232,255,0.85);
  font-weight:900;
}.kvV{
  margin-top:4px;
  color:rgba(241,245,255,0.96);
  font-size:13.4px;
  line-height:1.55;
}.aboutFooterMeta{
  margin-top:16px;
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}
/* ---------- APP LAYOUT ---------- */
.sidebar{
  border-radius: var(--r2);
  background: rgba(15,23,42,0.92);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: var(--shadow2);
  padding: 14px;
  margin-top: 14px;
}.sectionTitle{
  font-weight: 950;
  font-size: 12px;
  letter-spacing: .12em;
  color: rgba(255,255,255,0.86);
  margin: 4px 0 10px;
}.helper{
  color: var(--muted2);
  font-size: 12px;
  line-height: 1.45;
  margin-bottom: 10px;
}.card{
  border-radius: var(--r);
  background: rgba(15,23,42,0.98);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 12px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.30);
}.gradio-container .gr-input,.gradio-container .gr-textbox,.gradio-container .gr-image,.gradio-container .gr-number,.gradio-container .gr-slider{
  background: rgba(15,23,42,0.92) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);}
/* Buttons row */
#action_row{ gap: 12px !important; }
#action_row > div{ flex: 1 1 0 !important; min-width: 0 !important; }
#action_row button{ width: 100% !important; height: 56px !important; border-radius: 16px !important; }
#btn_analyze button,#btn_reset button,#btn_download button,#btn_about button{
  background: var(--btn-grad) !important;
  border: 0 !important;
  box-shadow: var(--btn-shadow) !important;
  color: transparent !important;
  -webkit-background-clip: text;
  background-clip: text;
  font-weight: 950 !important;
  letter-spacing: 0.01em !important;
  text-shadow: 0 1px 10px rgba(0,0,0,0.35);}
#btn_analyze button:hover,#btn_reset button:hover,#btn_download button:hover,#btn_about button:hover{
  transform: translateY(-2px) !important;
  filter: brightness(1.06) !important;
  background: linear-gradient(135deg, #ff79c6, #9b6cff) !important;}
/* Tabs styling */
.gradio-container .tabs{
  border-radius: var(--r2) !important;
  overflow:hidden;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(15,23,42,0.92);
  box-shadow: var(--shadow2);
}.gradio-container .tab-nav button.selected::after,.gradio-container .tab-nav .selected::after{
  background: linear-gradient(90deg, #9b6cff, #ff79c6) !important;
}
/* Main results panel */
.mainPanel{
  margin-top:14px;
}
.resultsCardTitle{
  font-size:12px;
  letter-spacing:.12em;
  text-transform:uppercase;
  font-weight:900;
  color:rgba(255,255,255,0.76);
  margin-bottom:6px;
}

/* 🎯 REMOVE GRAY FROM DOWNLOAD BUTTON */
#btn_download button {
    background: var(--btn-grad) !important;
    border: 0 !important;
    box-shadow: var(--btn-shadow) !important;
    color: transparent !important;
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: 950 !important;
    letter-spacing: 0.01em !important;
    text-shadow: 0 1px 10px rgba(0,0,0,0.35);
    height: 44px !important;
    border-radius: 12px !important;
    width: 100% !important;
}
#btn_download button:hover {
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
}

/* 🎯 Compact Kernels tab */
.kernels-tab .gr-download {
    margin-top: 12px;
}
/* =====================================================
   ✅ REMOVE GREY BACKGROUND — APP SECTION ONLY
   (No model / logic changes)
   ===================================================== */

/* Kill default Gradio grey variables */
.gradio-container{
  --block-background-fill: transparent !important;
  --block-background-fill-hover: transparent !important;
}

/* App cards, groups, panels */
.mainPanel,
.sidebar,
#results_card,
.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .gr-panel{
  background: rgba(15,23,42,0.96) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: 0 14px 32px rgba(0,0,0,0.40) !important;
}

/* Remove tab grey (Summary / Stats / Kernels) */
.gradio-container .tabitem,
.gradio-container .tabitem > div{
  background: transparent !important;
}

/* Inputs (sliders, textboxes, dropdowns) */
.gradio-container input,
.gradio-container textarea,
.gradio-container select{
  background: rgba(15,23,42,0.96) !important;
  color: #eef3ff !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}

/* ✅ Buttons – remove grey even when disabled */
.gradio-container button{
  background: var(--btn-grad) !important;
  border: 0 !important;
  box-shadow: var(--btn-shadow) !important;
  color: #fff !important;
}

.gradio-container button:disabled{
  opacity: 0.55 !important;
  filter: none !important;
}

/* ✅ Download button (biggest grey culprit) */
#btn_download button{
  background: var(--btn-grad) !important;
  border: 0 !important;
  box-shadow: var(--btn-shadow) !important;
  color: #fff !important;
}

/* ✅ Kernel table – remove grey rows */
.gradio-container table{
  background: transparent !important;
}

.gradio-container thead{
  background: rgba(15,23,42,0.9) !important;
}

.gradio-container tbody tr{
  background: rgba(10,16,40,0.9) !important;
}

.gradio-container tbody tr:nth-child(even){
  background: rgba(22,26,70,0.95) !important;
}

/* ✅ Image / file wrappers */
.gradio-container .gr-image,
.gradio-container .gr-file,
.gradio-container .gr-download{
  background: rgba(15,23,42,0.96) !important;
  border-radius: 16px !important;
}
"""

# ----------------------------
# Load models
# ----------------------------
MODELS_DIR = "models"
print("Loading models...")
unet, embedder, feature_columns, scaler, umap_model, gmm_model, device = load_models(models_dir=MODELS_DIR)
print("✓ Models loaded successfully!")

# ----------------------------
# Sample discovery
# ----------------------------
SAMPLE_PLACEHOLDER = "— Samples —"
def discover_samples():
    choices = [SAMPLE_PLACEHOLDER]
    for f in sorted(os.listdir(".")):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            choices.append(f)
    return gr.update(choices=choices, value=choices[0])

def load_sample(choice):
    if not choice or choice == SAMPLE_PLACEHOLDER:
        return None
    path = os.path.join(".", choice)
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ----------------------------
# NEW: Format kernel table as HTML
# ----------------------------
def format_table_html(kernel_df: pd.DataFrame):
    if kernel_df.empty:
        return "<div style='color:rgba(255,255,255,0.7);font-style:italic;padding:8px;'>No kernels detected</div>"

    html = """
    <div style="overflow-x:auto;margin-top:8px;">
    <table style="
        width:100%;
        border-collapse:collapse;
        font-size:12px;
        color:rgba(255,255,255,0.88);
    ">
        <thead>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.12);">
                <th style="text-align:left;padding:6px 8px;">ID</th>
                <th style="text-align:left;padding:6px 8px;">Cluster</th>
                <th style="text-align:left;padding:6px 8px;">Phenotype</th>
                <th style="text-align:left;padding:6px 8px;">Quality</th>
                <th style="text-align:left;padding:6px 8px;">Conf.</th>
                <th style="text-align:left;padding:6px 8px;">Morphology</th>
                <th style="text-align:right;padding:6px 8px;">W</th>
                <th style="text-align:right;padding:6px 8px;">H</th>
                <th style="text-align:right;padding:6px 8px;">Area</th>
            </tr>
        </thead>
        <tbody>
    """
    for _, row in kernel_df.iterrows():
        html += "<tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>"
        html += f'<td style="padding:6px 8px;">{row.get("kernel_id", "")}</td>'
        cluster = row.get("gmm_cluster")
        cluster_display = int(cluster) if pd.notna(cluster) else "—"
        html += f'<td style="padding:6px 8px;">{cluster_display}</td>'
        html += f'<td style="padding:6px 8px;">{row.get("phenotype_name", "")}</td>'
        html += f'<td style="padding:6px 8px;">{row.get("quality", "")}</td>'
        conf = row.get("confidence", 0.0)
        html += f'<td style="padding:6px 8px;">{conf:.3f}</td>'
        html += f'<td style="padding:6px 8px;">{row.get("morphology_description", "")}</td>'
        html += f'<td style="padding:6px 8px;text-align:right;">{int(row.get("width_px", 0))}</td>'
        html += f'<td style="padding:6px 8px;text-align:right;">{int(row.get("height_px", 0))}</td>'
        html += f'<td style="padding:6px 8px;text-align:right;">{int(row.get("area_px", 0))}</td>'
        html += "</tr>"
    html += """
        </tbody>
    </table>
    </div>
    """
    return html

# ----------------------------
# Pipeline helpers
# ----------------------------
def process(img_np):
    try:
        if img_np is None:
            return ("No image uploaded", None, gr.update(value=None, interactive=False), "", 0, "N/A", 0.0)
        
        # Save original image directly to temp
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, "input.jpg")
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(img_path, bgr)
        if not ok:
            return ("Failed to save image", None, gr.update(value=None, interactive=False), "", 0, "N/A", 0.0)

        result = test_user_image(img_path, unet, embedder, feature_columns, scaler, umap_model, gmm_model, device)

        if result is None:
            raise RuntimeError("No kernels detected")

        kernel_df: pd.DataFrame = result["kernel_features"]
        overlay_rgb = result.get("overlay_image", None)
        csv_path = os.path.join(temp_dir, "kernels.csv")
        kernel_df.to_csv(csv_path, index=False)

        n = int(len(kernel_df))
        summary_txt = (
            f"✅ Detected {n} kernels\n"
            f"📊 Overall quality: {result.get('image_phenotype', 'Unknown')}\n\n"
            "🟩 Green: High-quality kernels \n"
            "🟨 Yellow: Medium-quality kernels\n"
            "🟥 Red: Low-quality kernels"
        )

        if "quality" in kernel_df.columns and n > 0:
            q = kernel_df["quality"].value_counts(normalize=True).mul(100).round(1)
            quality_dist = "\n".join([f"{k}: {v}%" for k, v in q.items()])
        else:
            quality_dist = "N/A"

        conf_avg = float(kernel_df["confidence"].mean()) if "confidence" in kernel_df.columns and n > 0 else 0.0
        table_html = format_table_html(kernel_df)

        return (
            summary_txt,
            overlay_rgb,
            gr.update(value=csv_path, interactive=True),
            table_html,
            n,
            quality_dist,
            conf_avg,
        )
    except Exception as e:
        traceback.print_exc()
        return (
            f"Error: {str(e)}",
            None,
            gr.update(value=None, interactive=False),
            "<div style='color:#fb7185;padding:8px;'>Kernel table unavailable due to error.</div>",
            0,
            "N/A",
            0.0,
        )
def clear_all():
    return (
        None,  # input_image
        "",    # summary
        None,  # overlay_img
        gr.update(value=None, interactive=False),  # csv_download
        "",    # kernel_table_html
        0,     # total_kernels
        "N/A", # quality_dist
        0.0    # conf_avg
    )

def go_about():
    return gr.update(selected=0)

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="Maize Phenotype Analyzer") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    # ===== Welcome overlay =====
    # ===== Welcome spinning overlay =====
    welcome_html = gr.HTML("""
    <div id="welcome_overlay" style="
        position:fixed;top:0;left:0;width:100%;height:100%;
        background:#050815; color:#fff; display:flex; justify-content:center; align-items:center;
        font-size:2.5em; font-weight:900; z-index:9999; text-align:center;
        flex-direction:column;
        animation: fadeOut 1s ease forwards 3s;">
        <div style="animation: spin 3s linear infinite;">🌽</div>
        <div style="margin-top:12px; font-size:1em; font-weight:600; color:#9b6cff;">Welcome to Maize Phenotype Analyzer</div>
    </div>
    <style>
    @keyframes fadeOut {
        0% {opacity:1;}
        90% {opacity:1;}
        100% {opacity:0; display:none;}
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """)

    with gr.Row(elem_classes=["webTopbar"]):
        gr.HTML("""
        <div class="webBrand">
          <div class="webLogo" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none">
              <path d="M12 2c3.8 2.2 6 5.2 6 9 0 5-3.6 9-6 11-2.4-2-6-6-6-11 0-3.8 2.2-6.8 6-9Z"
                stroke="white" stroke-width="1.4" opacity="0.95"/>
            </svg>
          </div>
          <div style="display:flex;flex-direction:column;gap:2px">
            <div style="font-weight:950;font-size:16px">Maize Phenotype Analyzer</div>
          </div>
        </div>
        """)


    app_tabs = gr.Tabs(selected=1)

    with app_tabs:
        # ---------- ABOUT TAB ----------
        with gr.TabItem("About"):
            gr.HTML("""
            <div class="aboutPage">
              <div class="aboutHeader">
                <div class="aboutTitleBlock">
                  <div class="aboutTitle">About Maize Phenotype Analyzer</div>
                  <p class="aboutSubtitle">
                    A compact web application for high-throughput maize kernel phenotyping from RGB imagery.
                  </p>
                </div>
                <div class="aboutMeta">
                  <span class="miniTag"><span class="miniDot"></span> U-Net segmentation</span>
                  <span class="miniTag"><span class="miniDot"></span> ResNet50 embeddings</span>
                  <span class="miniTag"><span class="miniDot"></span> 2051-D feature space</span>
                  <span class="miniTag"><span class="miniDot"></span> UMAP + GMM clustering</span>
                </div>
              </div>
              <div class="aboutGrid">
                <div class="aboutCard">
                  <div class="aboutH">WHAT THE TOOL DOES</div>
                  <p class="aboutP">
                    Maize Phenotype Analyzer ingests a raw RGB image of maize kernels and runs a complete
                    phenotyping pipeline in a single click. The model stack:
                  </p>
                  <ul class="bullets" style="margin-top:8px">
                    <li><b>Segmentation (U-Net)</b> – isolates individual kernels from the background.</li>
                    <li><b>Feature extraction (ResNet50)</b> – generates a 2048-D embedding per kernel.</li>
                    <li><b>Morphology features</b> – size and shape descriptors are appended to form a 2051-D vector.</li>
                    <li><b>Standardization</b> – features are normalized via <code>StandardScaler</code>.</li>
                    <li><b>UMAP projection</b> – reduces to 2D while preserving local structure for clustering.</li>
                    <li><b>Gaussian Mixture Model (GMM)</b> – clusters kernels into 9 phenotype groups.</li>
                    <li><b>CLUSTER_TO_PHENOTYPE mapping</b> – converts cluster IDs into interpretable phenotype labels.</li>
                  </ul>
                  <p class="aboutP" style="margin-top:10px">
                    The result is a per-kernel table and overlay visualization that are consistent, reproducible,
                    and immediately usable in downstream analysis workflows.
                  </p>
                </div>
                <div class="aboutCard">
                  <div class="aboutH">OUTPUT AT A GLANCE</div>
                  <div class="kv">
                    <div class="kvItem">
                      <div class="kvK">Overlay View</div>
                      <div class="kvV">
                        Color-coded overlay image showing detected kernels and cluster/phenotype assignments
                        for quick visual QA.
                      </div>
                    </div>
                    <div class="kvItem">
                      <div class="kvK">Kernel Table</div>
                      <div class="kvV">
                        One row per kernel, including cluster ID, phenotype name, model confidence,
                        morphology description, and size statistics in pixels.
                      </div>
                    </div>
                    <div class="kvItem">
                      <div class="kvK">CSV Export</div>
                      <div class="kvV">
                        A ready-to-use <b>kernels.csv</b> file for statistical analysis, dashboards,
                        or as labeled data for ML experiments.
                      </div>
                    </div>
                  </div>
                </div>
                <div class="aboutCard">
                  <div class="aboutH">HOW TO USE</div>
                  <ul class="bullets">
                    <li><b>Upload</b> a maize-kernel image or pick one from the sample dropdown.</li>
                    <li><b>Crop</b> tightly around the kernel region using the sliders (recommended for best segmentation).</li>
                    <li>Click <b>Analyze Cropped Image</b> to run the full pipeline.</li>
                    <li>Review the <b>Summary & Overlay</b> on the App tab for a quick visual check.</li>
                    <li>Inspect the <b>Kernel Table</b> for per-kernel metrics.</li>
                    <li>Use <b>Download CSV</b> to export results for downstream work.</li>
                  </ul>
                </div>
                <div class="aboutCard">
                  <div class="aboutH">BEST PRACTICES</div>
                  <ul class="bullets">
                    <li>Use well-lit, high-contrast images with kernels clearly separated from the background.</li>
                    <li>Avoid extreme perspective distortion; a flat, top-down view yields the most stable results.</li>
                    <li>If the image contains irrelevant regions, crop tightly to reduce false detections.</li>
                    <li>When comparing batches, keep imaging conditions (lighting, distance, resolution) consistent.</li>
                  </ul>
                </div>
              </div>
              <div class="aboutFooterMeta">
                <span class="miniTag">Built by Eman Nawaz</span>
                <span class="miniTag">© 2025 • Research / internal use only</span>
              </div>
            </div>
            """)

        # ---------- APP TAB ----------
        with gr.TabItem("App"):
            with gr.Row(equal_height=True):
                # Sidebar (left)
                with gr.Column(scale=1, min_width=340, elem_classes=["sidebar"]):
                    gr.HTML('<div class="sectionTitle">INPUT</div>')
                   
                    sample_dd = gr.Dropdown(
                        label="Samples",
                        choices=[SAMPLE_PLACEHOLDER],
                        value=SAMPLE_PLACEHOLDER,
                        interactive=True,
                    )
                    input_image = gr.Image(label="Upload Image", type="numpy")

                    with gr.Row(elem_id="action_row"):
                        run_btn = gr.Button("Analyze Image", elem_id="btn_analyze")
                        clear_btn = gr.Button("Reset", elem_id="btn_reset")

                # Main panel (right)
                with gr.Column(scale=2, min_width=720, elem_classes=["mainPanel"]):
                    # 🔥 TABBED RESULTS CARD
                    with gr.Group(elem_classes=["card"], elem_id="results_card"):
                        with gr.Tabs():
                            with gr.TabItem("Summary"):
                                gr.HTML('<div class="resultsCardTitle">Summary & overlay</div>')
                                summary = gr.Textbox(
                                    lines=4, interactive=False, show_label=False,
                                    placeholder="Run an analysis to see summary text here."
                                )
                                overlay_img = gr.Image(
                                    label=None, show_label=False, type="numpy"
                                )

                            with gr.TabItem("Stats"):
                                gr.HTML('<div class="resultsCardTitle">Image statistics</div>')
                                total_kernels = gr.Number(
                                    label="Detected kernels", value=0, interactive=False
                                )
                                conf_avg = gr.Number(
                                    label="Avg. confidence", value=0.0, interactive=False
                                )
                                quality_dist = gr.Textbox(
                                    label="Quality distribution", lines=3, interactive=False
                                )

                            with gr.TabItem("Kernels"):
                                gr.HTML('<div class="resultsCardTitle">Kernel table</div>')
                                kernel_table_html = gr.HTML()
                                # ✅ SMALL, ATTRACTIVE DOWNLOAD BUTTON — NO GRAY!
                                csv_download = gr.DownloadButton(
                                    label="📥 Download CSV",
                                    value=None,
                                    interactive=False,
                                    elem_id="btn_download"
                                )

    # Bind buttons & events
    demo.load(fn=discover_samples, inputs=None, outputs=sample_dd, queue=False)
    sample_dd.change(fn=load_sample, inputs=sample_dd, outputs=input_image, queue=False)
    
   
    clear_btn.click(
    fn=clear_all,
    inputs=None,
    outputs=[input_image, summary, overlay_img, csv_download, kernel_table_html, total_kernels, quality_dist, conf_avg],
    queue=False
    )

    run_btn.click(
    fn=process,
    inputs=[input_image],
    outputs=[summary, overlay_img, csv_download, kernel_table_html, total_kernels, quality_dist, conf_avg],
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
    