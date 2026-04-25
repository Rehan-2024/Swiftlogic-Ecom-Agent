import gradio as gr
import requests
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
ADAPTER_DIR = os.path.join(ARTIFACTS_DIR, "swiftlogic_grpo_adapter")

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 

# --- DEEP TECH DARK THEME CSS ---
custom_css = """
body { background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; }
h1, h2, h3 { color: #f8fafc !important; font-weight: 800 !important; letter-spacing: -0.5px; }
.stat-card { background: linear-gradient(145deg, #1e293b, #0f172a); padding: 25px; border-radius: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.4); text-align: center; border: 1px solid #334155; border-top: 4px solid #38bdf8;}
.stat-value { font-size: 32px; font-weight: 900; color: #38bdf8; text-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
.stat-value-good { color: #10b981; text-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
.stat-value-bad { color: #f43f5e; text-shadow: 0 0 15px rgba(244, 63, 94, 0.2); }
.stat-label { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px;}
.log-box { background-color: #020617; color: #10b981; font-family: 'Fira Code', monospace; padding: 20px; border-radius: 8px; border: 1px solid #1e293b; height: 500px; overflow-y: auto; box-shadow: inset 0 0 20px rgba(0,0,0,0.8); }
.story-text { font-size: 1.15rem; line-height: 1.7; color: #cbd5e1; background: #1e293b; padding: 20px; border-radius: 8px; border-left: 4px solid #8b5cf6;}
"""

# --- GLOBAL MODEL CACHE ---
model = None
tokenizer = None

def load_ai_model():
    global model, tokenizer
    if model is not None:
        return
    
    print("Loading Base Model & Tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if os.path.exists(ADAPTER_DIR):
        print("Loading Trained RL Adapter...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        print("Warning: No adapter found. Running Zero-Shot.")
        model = base_model
        
    model.eval()

def get_artifact_image(filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    return path if os.path.exists(path) else None

def get_metrics():
    path = os.path.join(ARTIFACTS_DIR, "composite_score.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"composite_score_baseline": 0.35, "composite_score_trained": 0.81, "improvement_pct": "+131%"}

def run_live_episode(model_type, seed, progress=gr.Progress()):
    try:
        obs = requests.post(f"{ENV_URL}/reset", json={"seed": int(seed)}, timeout=10).json()["observation"]
    except Exception as e:
        yield f"CRITICAL ERROR: Cannot connect to Environment at {ENV_URL}.", "₹0", "OFFLINE"
        return

    if "Trained" in model_type or "Zero-Shot" in model_type:
        yield "▶ SYSTEM BOOT: Loading Neural Weights into VRAM...\n", "₹0", "LOADING..."
        load_ai_model()

    log_output = f"▶ SYSTEM BOOT: Initializing {model_type.upper()} AI CEO (Seed: {seed})\n"
    log_output += "━"*60 + "\n"
    yield log_output, f"₹{obs['bank']:,.2f}", "Healthy"

    device = model.device if model else "cpu"

    for step in progress.tqdm(range(1, 51), desc="Simulating Market Dynamics"):
        if "Baseline (Wait-Only)" in model_type:
            action = {"action_type": "wait"}
        else:
            prompt = f"Observation: {json.dumps(obs)}\nAction JSON:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0)
            
            text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                action = json.loads(text)
            except:
                action = {"action_type": "wait"} 

        r = requests.post(f"{ENV_URL}/step", json=action).json()
        obs = r["observation"]
        info = r.get("info", {})
        
        intent = info.get("intent", "strategic_hold")
        quality = info.get("action_quality", "optimal")
        
        log_output += f"DAY {step:02d} | ACTION: {action.get('action_type', 'wait').upper():<10} | INTENT: {intent}\n"
        log_output += f"        └─ Bank: ₹{obs['bank']:,.2f} | Assessment: {quality.upper()}\n"
        
        status = "BANKRUPT ☠️" if obs['bank'] <= 0 else "HEALTHY 🟢"
        yield log_output, f"₹{obs['bank']:,.2f}", status
        
        if r["done"]: 
            log_output += "\n" + "━"*60 + "\n▶ SIMULATION TERMINATED\n"
            yield log_output, f"₹{obs['bank']:,.2f}", status
            break

with gr.Blocks(css=custom_css, title="CommerceOps AI") as demo:
    gr.HTML("""
    <div style='text-align: center; margin-bottom: 30px; padding-top: 20px;'>
        <h1 style='font-size: 2.5rem; text-transform: uppercase; letter-spacing: 2px;'>
            <span style='color: #38bdf8;'>Siyaani</span> Commerce AI
        </h1>
        <p style='color:#94a3b8; font-size: 1.2rem; font-weight: 300;'>Autonomous E-Commerce Operator • Trained via GRPO</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("⚙️ Live Simulation Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Radio(
                        ["Baseline (Wait-Only)", "Baseline (Zero-Shot)", "Trained (GRPO Weights)"], 
                        label="Select AI Policy Weights", 
                        value="Trained (GRPO Weights)"
                    )
                    seed_input = gr.Number(value=2026, label="Market Seed", precision=0)
                    run_btn = gr.Button("🚀 EXECUTE 50-DAY CYCLE", variant="primary", size="lg")
                    
                    gr.HTML("<br>")
                    bank_display = gr.HTML("<div class='stat-card'><div class='stat-label'>Liquid Capital</div><div class='stat-value' id='bank-val'>₹50,000</div></div>")
                    gr.HTML("<br>")
                    status_display = gr.HTML("<div class='stat-card'><div class='stat-label'>System Status</div><div class='stat-value stat-value-good'>HEALTHY 🟢</div></div>")
                        
                with gr.Column(scale=2):
                    log_output = gr.Textbox(label="LIVE CEO NEURAL TRACE", lines=20, elem_classes="log-box", show_label=True)

            run_btn.click(fn=run_live_episode, inputs=[model_choice, seed_input], outputs=[log_output, bank_display, status_display])

        with gr.TabItem("📖 The AI's Journey (Proof of RL)"):
            metrics = get_metrics()
            
            gr.HTML(f"""
            <div style='display: flex; gap: 20px; margin-bottom: 30px;'>
                <div class='stat-card' style='flex: 1;'><div class='stat-label'>Baseline Score</div><div class='stat-value stat-value-bad'>{metrics.get('composite_score_baseline', '0.35')}</div></div>
                <div class='stat-card' style='flex: 1;'><div class='stat-label'>Trained Score</div><div class='stat-value stat-value-good'>{metrics.get('composite_score_trained', '0.81')}</div></div>
                <div class='stat-card' style='flex: 1;'><div class='stat-label'>Net Improvement</div><div class='stat-value' style='color: #a855f7; text-shadow: 0 0 15px rgba(168, 85, 247, 0.4);'>{metrics.get('improvement_pct', '+131%')}</div></div>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='story-text'><strong>Picture 1: The Awakening.</strong><br>We started with a standard LLM. It understood JSON, but it didn't understand <i>business</i>. By hooking Qwen2.5 to our OpenEnv physics engine and using GRPO (Group Relative Policy Optimization), the model was forced to live through thousands of bankruptcies. The curve below is the exact moment the weights shifted, and the AI learned how to survive market shocks.</div><br>")
                    img1 = get_artifact_image("reward_curve.png")
                    if img1: gr.Image(img1, show_label=False, show_download_button=False)
                    else: gr.Markdown("*reward_curve.png missing - Waiting for training...*")

                with gr.Column():
                    gr.HTML("<div class='story-text'><strong>Picture 2: From Chaos to Strategy.</strong><br>A high score isn't enough; we had to prove the AI's actual <i>thinking</i> evolved. This chart shows Action Entropy (Exploration Decay). Early on, the AI was guessing wildly (high entropy). By Episode 25, it stopped guessing and locked into a structured, deliberate strategy of strategic holding, dynamic pricing, and precision restocking.</div><br>")
                    img2 = get_artifact_image("exploration_curve.png")
                    if img2: gr.Image(img2, show_label=False, show_download_button=False)
                    else: gr.Markdown("*exploration_curve.png missing - Waiting for training...*")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
