import os
import re
import warnings

warnings.filterwarnings("ignore")

import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import altair as alt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FAO = os.path.join(BASE_DIR, "Datasets", "Fao_merged_clean.csv")
DATA_DEMAND = os.path.join(BASE_DIR, "Datasets", "module2_matched_demand.csv")
DATA_SPOILAGE = os.path.join(BASE_DIR, "Datasets", "module3_fao_spoilage_v2.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "Results")

MODEL_FAO = os.path.join(RESULTS_DIR, "fao_model.joblib")
MODEL_DEMAND = os.path.join(RESULTS_DIR, "demand_model.joblib")
ENCODERS = os.path.join(RESULTS_DIR, "demand_encoders.joblib")
MODEL_SPOILAGE = os.path.join(RESULTS_DIR, "spoilage_model.joblib")

METRICS_FAO = os.path.join(RESULTS_DIR, "metrics.txt")
METRICS_DEMAND = os.path.join(RESULTS_DIR, "demand_metrics.txt")
METRICS_SPOILAGE = os.path.join(RESULTS_DIR, "spoilage_metrics.txt")

# ============================================================
# THEME / DESIGN SYSTEM
# ============================================================

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ============================================================
       SUSTAINFLOW AI — VIBRANT COLORFUL ANIMATED THEME
       ============================================================ */

    /* ---------- Base ---------- */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 20px; line-height: 1.75; font-weight: 700; }
    .stApp {
        background:
            radial-gradient(1400px 700px at 5% -10%, rgba(16,185,129,0.35), transparent 60%),
            radial-gradient(1100px 600px at 108% 0%, rgba(236,72,153,0.25), transparent 55%),
            radial-gradient(900px 600px at 50% 115%, rgba(59,130,246,0.20), transparent 60%),
            linear-gradient(135deg, #f0fdf4 0%, #d1fae5 30%, #fce7f3 60%, #dbeafe 100%);
        color: #0f172a;
        overflow-x: hidden;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 3rem; max-width: 1280px; }

    ::selection { background: rgba(16,185,129,0.40); color: #ffffff; }
    ::-webkit-scrollbar { width: 14px; height: 14px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #10b981, #3b82f6); border-radius: 10px; }
    ::-webkit-scrollbar-track { background: rgba(20,83,45,0.10); border-radius: 10px; }

    /* ---------- Keyframe Animations ---------- */
    @keyframes floaty { 0%,100% { transform: translate(0,0); } 50% { transform: translate(30px,-25px); } }
    @keyframes floaty2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-20px,20px); } }
    @keyframes fadeUp { from { opacity:0; transform: translateY(30px); } to { opacity:1; transform:none; } }
    @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
    @keyframes slideInLeft { from { opacity:0; transform: translateX(-40px); } to { opacity:1; transform: translateX(0); } }
    @keyframes slideInRight { from { opacity:0; transform: translateX(40px); } to { opacity:1; transform: translateX(0); } }
    @keyframes bounceIn { from { opacity:0; transform: scale(0.3); } 50% { opacity:1; transform: scale(1.05); } 70% { transform: scale(0.9); } to { opacity:1; transform: scale(1); } }
    @keyframes zoomIn { from { opacity:0; transform: scale(0.8); } to { opacity:1; transform: scale(1); } }
    @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 0 rgba(16,185,129,0.4); } 50% { box-shadow: 0 0 50px rgba(16,185,129,0.8); } }
    @keyframes rainbow { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(360deg); } }
    @keyframes shake { 0%,100% { transform: translateX(0); } 25% { transform: translateX(-5px); } 75% { transform: translateX(5px); } }
    @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

    /* ---------- Animation utility classes ---------- */
    .sf-anim { animation: fadeUp 0.7s ease-out both; }
    .sf-anim-2 { animation: fadeUp 0.9s ease-out both; }
    .sf-anim-3 { animation: fadeUp 1.1s ease-out both; }
    .sf-anim-4 { animation: fadeUp 1.3s ease-out both; }
    .sf-fade-in { animation: fadeIn 0.6s ease-out both; }
    .sf-slide-left { animation: slideInLeft 0.7s ease-out both; }
    .sf-slide-right { animation: slideInRight 0.7s ease-out both; }
    .sf-bounce { animation: bounceIn 0.8s ease-out both; }
    .sf-zoom { animation: zoomIn 0.6s ease-out both; }
    .sf-pulse { animation: pulseGlow 3s infinite; }
    .sf-rainbow { animation: rainbow 4s ease-in-out infinite; }
    .sf-shake { animation: shake 0.5s ease-in-out both; }
    .sf-blink { animation: blink 1.5s infinite; }
    .sf-delay-1 { animation-delay: 0.1s; }
    .sf-delay-2 { animation-delay: 0.2s; }
    .sf-delay-3 { animation-delay: 0.3s; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(600px 400px at 0% 0%, rgba(16,185,129,0.45), transparent 60%),
            radial-gradient(500px 400px at 100% 100%, rgba(236,72,153,0.45), transparent 60%),
            linear-gradient(180deg, #064e3b 0%, #0f172a 60%, #1e1b4b 100%);
        border-right: 4px solid linear-gradient(180deg, #10b981, #ec4899);
        box-shadow: 14px 0 44px rgba(4,41,30,0.40);
    }
    [data-testid="stSidebar"] * { color: #ffffff; }
    [data-testid="stSidebar"] > div { padding-bottom: 32px; }

    /* Brand */
    .stSidebarWelcome {
        display: block; text-align: center;
        color: #a7f3d0; font-size: 18px; font-weight: 900;
        letter-spacing: 1px; margin: 0 0 12px 0;
        text-shadow: 0 3px 12px rgba(0,0,0,0.45);
    }
    .stSidebarBrand {
        font-size: 34px; font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #a7f3d0, #fce7f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border: 3px solid rgba(255,255,255,0.20);
        padding: 26px 18px; border-radius: 24px;
        color: #ffffff; margin-bottom: 24px;
        text-align: center; letter-spacing: 0.8px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 12px 32px rgba(0,0,0,0.40);
        position: relative; overflow: hidden;
    }
    .stSidebarBrand::before {
        content: ""; position: absolute; top:0; left:-100%; width:100%; height:100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer { 0% { left: -100%; } 100% { left: 100%; } }
    .stSidebarBrand small {
        display: block; font-size: 14px; font-weight: 700; opacity: 0.95; margin-top: 5px;
        letter-spacing: 1.2px; text-transform: uppercase;
    }

    /* Navigation = one-click clear pill buttons */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 18px;
        padding: 18px 26px !important;
        margin: 12px 0;
        cursor: pointer;
        transition: all 0.30s ease;
        font-weight: 800; font-size: 18px;
        color: #ffffff;
        box-shadow: 0 6px 16px rgba(0,0,0,0.30);
        width: 100%;
        text-align: left;
        line-height: 1.5;
        position: relative; overflow: hidden;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button::before {
        content: ""; position: absolute; top:0; left:-100%; width:100%; height:100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover::before { left: 100%; }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.25);
        transform: translateX(10px) scale(1.02);
        border-color: rgba(255,255,255,0.50);
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #10b981, #34d399, #ec4899);
        border: 2px solid #86efac;
        color: #ffffff;
        box-shadow: 0 12px 32px rgba(16,185,129,0.55);
        border-left: 8px solid #22d3ee;
        animation: pulseGlow 3s infinite;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #34d399, #10b981, #f472b6);
        transform: translateX(12px) scale(1.03);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:active { transform: translateY(2px); }
    [data-testid="stSidebar"] .stMarkdown p { color: #ffffff; opacity: 0.98; font-size: 18px; font-weight: 700; }

    /* ---------- Hero banner ---------- */
    .sf-hero {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #064e3b 0%, #0f766e 25%, #10b981 50%, #22d3ee 75%, #3b82f6 100%);
        border-radius: 36px; padding: 64px 64px; color: #ffffff;
        box-shadow: 0 32px 72px rgba(6,78,59,0.55); margin-bottom: 38px;
        border: 3px solid rgba(255,255,255,0.25);
        animation: fadeUp 0.7s ease, floaty 6s ease-in-out infinite;
    }
    .sf-hero::before {
        content: "🌱"; position: absolute; right: 30px; bottom: -40px;
        font-size: 200px; opacity: 0.15; transform: rotate(-15deg);
        animation: floaty2 8s ease-in-out infinite;
    }
    .sf-hero::after {
        content: ""; position: absolute; right: -100px; top: -100px;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.30), transparent 70%);
    }
    .sf-hero h1 {
        position: relative; z-index: 1; margin: 0 0 18px 0;
        font-size: 62px; font-weight: 900; letter-spacing: -0.5px;
        text-shadow: 0 4px 16px rgba(0,0,0,0.35);
        background: linear-gradient(135deg, #ffffff, #a7f3d0, #bae6fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sf-hero p {
        position: relative; z-index: 1; margin: 0; font-size: 25px; opacity: 0.98;
        max-width: 1000px; line-height: 1.9;
    }
    .sf-hero-tag {
        position: relative; z-index: 1; display: inline-block; margin-bottom: 20px;
        background: rgba(255,255,255,0.22); border: 2px solid rgba(255,255,255,0.40);
        border-radius: 999px; padding: 10px 26px; font-size: 15px; font-weight: 900;
        letter-spacing: 1.8px; text-transform: uppercase;
        backdrop-filter: blur(10px);
        animation: sf-bounce 2s ease-in-out infinite;
    }

    /* ---------- Cards ---------- */
    .sf-card {
        background: #ffffff; border-radius: 30px; padding: 38px 40px;
        box-shadow: 0 18px 44px rgba(20,83,45,0.15);
        border: 3px solid rgba(34,197,94,0.18);
        border-left: 10px solid #10b981;
        margin-bottom: 28px; transition: all 0.35s ease;
        animation: fadeUp 0.6s ease;
    }
    .sf-card:hover { transform: translateY(-8px) scale(1.01); box-shadow: 0 28px 60px rgba(20,83,45,0.22); border-color: #34d399; }
    .sf-card h3 { margin: 0 0 20px 0; color: #0f172a; font-size: 28px; font-weight: 900; }
    .sf-card p { font-size: 21px; font-weight: 800; color: #0f172a; line-height: 1.9; }
    .sf-card li { font-size: 21px; font-weight: 800; color: #0f172a; line-height: 1.9; }

    /* ---------- Module-specific colored cards ---------- */
    .sf-card-m1 { border-left-color: #3b82f6; border-color: rgba(59,130,246,0.20); }
    .sf-card-m1 h3 { color: #1e40af; }
    .sf-card-m1:hover { border-color: #3b82f6; box-shadow: 0 28px 60px rgba(59,130,246,0.20); }
    .sf-card-m2 { border-left-color: #10b981; border-color: rgba(16,185,129,0.20); }
    .sf-card-m2 h3 { color: #065f46; }
    .sf-card-m2:hover { border-color: #10b981; box-shadow: 0 28px 60px rgba(16,185,129,0.20); }
    .sf-card-m3 { border-left-color: #ef4444; border-color: rgba(239,68,68,0.20); }
    .sf-card-m3 h3 { color: #991b1b; }
    .sf-card-m3:hover { border-color: #ef4444; box-shadow: 0 28px 60px rgba(239,68,68,0.20); }
    .sf-card-m4 { border-left-color: #f59e0b; border-color: rgba(245,158,11,0.20); }
    .sf-card-m4 h3 { color: #b45309; }
    .sf-card-m4:hover { border-color: #f59e0b; box-shadow: 0 28px 60px rgba(245,158,11,0.20); }

    /* ---------- KPI cards ---------- */
    .sf-kpi {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #064e3b 0%, #0f766e 30%, #10b981 60%, #22d3ee 100%);
        border-radius: 30px; padding: 36px 28px; color: #ffffff;
        box-shadow: 0 20px 48px rgba(6,78,59,0.50); text-align: center;
        margin-bottom: 28px; border: 3px solid rgba(255,255,255,0.25);
        transition: all 0.35s ease; animation: fadeUp 0.6s ease;
    }
    .sf-kpi::after {
        content: ""; position: absolute; right: -45px; top: -45px;
        width: 160px; height: 160px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.40), transparent 70%);
    }
    .sf-kpi:hover { transform: translateY(-10px) scale(1.02); box-shadow: 0 30px 64px rgba(6,78,59,0.60); animation: pulseGlow 3s infinite; }
    .sf-kpi-icon { font-size: 38px; margin-bottom: 12px; }
    .sf-kpi-value { font-size: 72px; font-weight: 900; color: #ffffff; letter-spacing: 0.1px; }
    .sf-kpi-label {
        font-size: 17px; letter-spacing: 2.2px; opacity: 0.98; margin-top: 16px;
        text-transform: uppercase; font-weight: 900;
    }

    /* ---------- Module showcase cards (Home) ---------- */
    .sf-mcard {
        position: relative; overflow: hidden; border-radius: 32px;
        padding: 36px 34px; color: #ffffff; min-height: 210px;
        border: 3px solid rgba(255,255,255,0.30);
        box-shadow: 0 20px 48px rgba(0,0,0,0.40);
        transition: all 0.35s ease; animation: fadeUp 0.7s ease;
    }
    .sf-mcard::after {
        content: ""; position: absolute; right: -40px; bottom: -40px; font-size: 130px; opacity: 0.25;
    }
    .sf-mcard:hover { transform: translateY(-12px) scale(1.04); box-shadow: 0 32px 68px rgba(0,0,0,0.48); }
    .sf-mcard .mc-ic { font-size: 56px; margin-bottom: 16px; display: block; }
    .sf-mcard .mc-t { font-size: 26px; font-weight: 900; margin: 16px 0 10px 0; letter-spacing: -0.5px; }
    .sf-mcard .mc-d { font-size: 20px; opacity: 0.98; line-height: 1.9; }
    .sf-mc1 { background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%); }
    .sf-mc1::after { content: "📉"; animation: floaty 5s ease-in-out infinite; }
    .sf-mc2 { background: linear-gradient(135deg, #065f46 0%, #10b981 50%, #34d399 100%); }
    .sf-mc2::after { content: "📦"; animation: floaty2 6s ease-in-out infinite; }
    .sf-mc3 { background: linear-gradient(135deg, #991b1b 0%, #ef4444 50%, #f87171 100%); }
    .sf-mc3::after { content: "🛑"; animation: floaty 4s ease-in-out infinite; }
    .sf-mc4 { background: linear-gradient(135deg, #92400e 0%, #f59e0b 50%, #fb923c 100%); }
    .sf-mc4::after { content: "💡"; animation: floaty2 5s ease-in-out infinite; }

    /* ---------- Result display ---------- */
    .sf-result-low, .sf-result-med, .sf-result-high {
        border-radius: 32px; padding: 42px 46px; color: #ffffff;
        box-shadow: 0 24px 56px rgba(0,0,0,0.35); text-align: center;
        border: 3px solid rgba(255,255,255,0.28);
        animation: fadeUp 0.6s ease, floaty 5s ease-in-out infinite;
    }
    .sf-result-low  { background: linear-gradient(135deg, #15803d 0%, #22c55e 50%, #4ade80 100%); }
    .sf-result-med  { background: linear-gradient(135deg, #b45309 0%, #f59e0b 50%, #fb923c 100%); }
    .sf-result-high { background: linear-gradient(135deg, #b91c1c 0%, #ef4444 50%, #f87171 100%); }
    .sf-result-value { font-size: 84px; font-weight: 900; line-height: 1.0; }
    .sf-result-caption { font-size: 20px; opacity: 0.98; margin-top: 12px; font-weight: 800; }

    .sf-risk-badge {
        display: inline-block; padding: 14px 36px; border-radius: 100px;
        font-weight: 900; font-size: 20px; color: #fff; letter-spacing: 1.4px;
        background: linear-gradient(135deg, #0f766e, #10b981, #22d3ee);
        box-shadow: 0 10px 28px rgba(6,78,59,0.55);
        animation: sf-bounce 2s ease-in-out infinite;
    }
    .sf-health { display: inline-block; background: #22c55e; }

    /* ---------- Small chips ---------- */
    .sf-chip {
        display: inline-block; padding: 12px 26px; border-radius: 100px;
        background: rgba(16,185,129,0.15); color: #065f46; font-size: 15px;
        font-weight: 900; letter-spacing: 0.8px;
        border: 3px solid rgba(16,185,129,0.40);
        box-shadow: 0 4px 12px rgba(16,185,129,0.20);
        animation: sf-pulse 2s ease-in-out infinite;
    }

    /* ---------- Section headers ---------- */
    .sf-section {
        font-size: 22px; font-weight: 900; color: #0f172a;
        letter-spacing: 3.5px; text-transform: uppercase; margin: 44px 0 20px 0;
        display: flex; align-items: center; gap: 16px;
    }
    .sf-section::before {
        content: ""; display: inline-block; width: 46px; height: 7px;
        border-radius: 7px; background: linear-gradient(90deg, #10b981, #3b82f6, #ec4899);
        box-shadow: 0 0 20px rgba(16,185,129,0.6);
    }

    /* ---------- Small text helpers ---------- */
    .sf-hint { font-size: 20px; font-weight: 800; color: #0f172a; }
    .sf-muted { color: #0f172a; font-size: 21px; font-weight: 800; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #34d399, #22d3ee);
        color: #ffffff; border: none; font-weight: 900; font-size: 20px;
        border-radius: 18px; padding: 22px 44px; width: 100%;
        box-shadow: 0 12px 30px rgba(16,185,129,0.45);
        transition: all 0.30s ease;
        white-space: pre-line; line-height: 1.6;
        position: relative; overflow: hidden;
    }
    .stButton > button::before {
        content: ""; position: absolute; top:0; left:-100%; width:100%; height:100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: 0.5s;
    }
    .stButton > button:hover::before { left: 100%; }
    .stButton > button:hover {
        background: linear-gradient(135deg, #34d399, #22d3ee, #3b82f6);
        color: #ffffff; transform: translateY(-6px) scale(1.03);
        box-shadow: 0 20px 40px rgba(16,185,129,0.60);
    }
    .stButton > button:active { transform: translateY(2px); }

    /* ---------- Download button ---------- */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #064e3b, #0f766e, #10b981);
        color: #ffffff; border: none; font-weight: 900; font-size: 18px;
        border-radius: 18px; padding: 20px 42px; width: 100%;
        box-shadow: 0 12px 30px rgba(6,78,59,0.42);
        transition: all 0.30s ease; opacity: 1; visibility: visible;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stDownloadButton"] button:focus,
    [data-testid="stDownloadButton"] button:active {
        background: linear-gradient(135deg, #0f766e, #10b981, #34d399);
        color: #ffffff; opacity: 1 !important; visibility: visible !important;
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 18px 38px rgba(6,78,59,0.55);
    }

    /* ---------- Form fields ---------- */
    [data-testid="stSelectbox"] { margin-bottom: 16px; }
    [data-testid="stSelectbox"] label { font-size: 19px; font-weight: 900; color: #0f172a; margin-bottom: 12px; letter-spacing: 0.5px; }
    [data-testid="stSelectbox"] small, [data-baseweb="select"] > div {
        border-radius: 22px !important;
        border: 4px solid rgba(16,185,129,0.45) !important;
        background: #ffffff !important;
        font-size: 21px;
        font-weight: 900;
        color: #0f172a !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        min-height: 80px !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 80px !important;
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }
    [data-baseweb="select"] > div:hover { border-color: #3b82f6 !important; box-shadow: 0 10px 28px rgba(59,130,246,0.30); }
    [data-baseweb="select"] [aria-expanded="true"] > div { border-color: #10b981 !important; box-shadow: 0 10px 28px rgba(16,185,129,0.35); }
    [data-baseweb="select"] span { font-size: 21px !important; font-weight: 900 !important; color: #0f172a !important; }
    [data-baseweb="select"] svg { color: #10b981 !important; }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: rgba(255,255,255,0.80);
        border: 3px solid rgba(16,185,129,0.25);
        border-radius: 28px;
        padding: 12px;
        box-shadow: 0 8px 22px rgba(20,83,45,0.12);
    }
    .stTabs [data-baseweb="tab"] {
        position: relative;
        background: transparent; border-radius: 24px; padding: 18px 36px;
        border: 3px solid transparent; font-weight: 900;
        color: #0f172a; font-size: 20px;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(16,185,129,0.18);
        color: #059669;
        transform: translateY(-3px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"]:has([aria-selected="true"]) {
        background: linear-gradient(135deg, #064e3b, #0f766e, #10b981) !important;
        color: #ffffff !important;
        border-color: #34d399 !important;
        box-shadow: 0 12px 32px rgba(6,78,59,0.55) !important;
        transform: translateY(-4px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after,
    .stTabs [data-baseweb="tab"]:has([aria-selected="true"])::after {
        content: ""; position: absolute; left: 50%; bottom: -5px; transform: translateX(-50%);
        width: 80%; height: 7px; border-radius: 7px;
        background: linear-gradient(90deg, #34d399, #22d3ee, #3b82f6);
    }

    /* ---------- Dataframes ---------- */
    .stDataFrame { border-radius: 18px; overflow: hidden; box-shadow: 0 10px 26px rgba(20,83,45,0.14); border: 3px solid rgba(16,185,129,0.20); }
    [data-testid="stTableStyledHeaderCell"] { background: #dcfce7 !important; color: #047857 !important; font-weight: 900; font-size: 18px; }
    [data-testid="stTableStyledCell"] { font-size: 17px; color: #0f172a; font-weight: 700; }

    /* ---------- Caption / live match text ---------- */
    .stCaption, caption { font-size: 18px !important; font-weight: 800; color: #059669; }

    /* ---------- Markdown inside cards ---------- */
    .sf-card p, .sf-card li { font-size: 21px; font-weight: 800; color: #0f172a; line-height: 1.9; }

    /* ---------- Glassmorphism utility ---------- */
    .sf-glass {
        background: rgba(255,255,255,0.70);
        backdrop-filter: blur(16px);
        border: 2px solid rgba(255,255,255,0.50);
        border-radius: 24px;
    }
</style>
"""

DARK_CSS = """
<style>
    /* ============================================================
       SUSTAINFLOW AI — DARK MODE OVERRIDES
       ============================================================ */

    [data-testid="stApp"] {
        background:
            radial-gradient(1200px 700px at 5% -10%, rgba(46,139,87,0.16), transparent 60%),
            radial-gradient(1000px 600px at 108% 0%, rgba(29,107,77,0.18), transparent 55%),
            linear-gradient(165deg, #071a12 0%, #0b2318 55%, #103523 100%);
        color: #ffffff;
    }

    .sf-card {
        background: #0e2a1d;
        border-color: rgba(74,214,149,0.16);
        border-left-color: #2e8b57;
        box-shadow: 0 8px 26px rgba(0,0,0,0.35);
    }
    .sf-card h3, .sf-result-caption { color: #ffffff; font-weight: 900; }
    .sf-muted, .sf-hint { color: #e7f7ec; font-weight: 800; }
    .sf-section { color: #ffffff; font-weight: 900; }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] { color: #ffffff; font-weight: 800; }

    [data-testid="stSelectbox"] label,
    [data-testid="stMultiselect"] label,
    [data-testid="stSlider"] label { color: #e7f7ec; font-weight: 900; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-baseweb="select"] > div {
        background: #0b2318 !important;
        border-color: rgba(74,214,149,0.40) !important;
        color: #ffffff !important;
        font-weight: 900 !important;
    }
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="menu"] {
        background: #0b2318 !important;
    }
    [data-baseweb="menu"] [data-baseweb="option"] * { color: #ffffff !important; font-weight: 800 !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.10);
        border-color: rgba(255,255,255,0.20);
    }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-weight: 900; }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(46,139,87,0.18); color: #ffffff; }

    [data-testid="stDataFrame"] { background: #0e2a1d; }
    [data-testid="stTableStyledCell"] { background: #0e2a1d; color: #ffffff; font-weight: 800; }
    [data-testid="stTableStyledHeaderCell"] { background: #0e4a36 !important; color: #ffffff !important; font-weight: 900; }
</style>
"""

NAV = ["🏠 Home Dashboard", "📉 Module 1 · Food Loss Prediction",
       "📦 Module 2 · Demand Forecast (Units)",
       "🛑 Module 3 · Spoilage Detection", "💡 Module 4 · Recommendation Engine"]

CURRENCIES = {
    "$ USD": ("$", "USD", 1.0),
    "₹ INR": ("₹", "INR", 84.0),
    "€ EUR": ("€", "EUR", 0.92),
    "£ GBP": ("£", "GBP", 0.79),
    "₨ PKR": ("₨", "PKR", 278.0),
}


def current_currency():
    return CURRENCIES.get(st.session_state.get("currency", "$ USD"), CURRENCIES["$ USD"])


def fmt_money(x, nd=0):
    symbol, _code, rate = current_currency()
    return f"{symbol}{float(x) * rate:,.{nd}f}"


def hero(title, subtitle, tag=None):
    st.markdown(
        f"""<div class="sf-hero">
              {f'<div class="sf-hero-tag">{tag}</div>' if tag else ''}
              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>""",
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(f'<div class="sf-section">{label}</div>', unsafe_allow_html=True)


def guide(title, steps):
    lis = "".join(f"<li>{s}</li>" for s in steps)
    st.markdown(
        f'<div class="sf-card" style="padding:16px 24px;">'
        f'<div style="font-weight:800;font-size:15px;margin-bottom:6px;">🧭 {title}</div>'
        f'<ul style="margin:0;padding-left:20px;font-size:13.5px;line-height:1.9;opacity:0.94;">{lis}</ul></div>',
        unsafe_allow_html=True,
    )


def tab_hint(text):
    st.caption(f"💡 {text}")


def show_image(path, caption=None):
    """Render a Results image only if it exists (graceful otherwise)."""
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.info(f"'{os.path.basename(path)}' not generated yet — run the matching "
                "training/prediction script to create it.")


def kpi_cols(cols, items, icons=None):
    with cols.container():
        grid = st.columns(len(items))
        for i, (col, item) in enumerate(zip(grid, items)):
            label, value = item
            icon = f'<div class="sf-kpi-icon">{icons[i]}</div>' if icons else ""
            col.markdown(
                f"""<div class="sf-kpi">
                      {icon}
                      <div class="sf-kpi-value">{value}</div>
                      <div class="sf-kpi-label">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_encoders(path):
    return joblib.load(path)


@st.cache_data
def load_fao_data():
    df = pd.read_csv(DATA_FAO)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


@st.cache_data
def load_demand_data():
    df = pd.read_csv(DATA_DEMAND)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    for col in ["country", "commodity"]:
        df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data
def load_spoilage_data():
    df = pd.read_csv(DATA_SPOILAGE)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    for col in ["country", "commodity", "food_supply_stage"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# ============================================================
# HELPERS
# ============================================================

def read_text(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    return "File not found."


def parse_metrics(text):
    rows = []
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "=", "MODEL SELECTION", "Why:")):
            continue
        if ":" in line and section:
            key, _, val = line.partition(":")
            rows.append({"section": section, "metric": key.strip(), "value": val.strip()})
        elif not line[0].isdigit() and ":" not in line:
            section = line
    return pd.DataFrame(rows)


def result_card(pred, caption, level_out, low, med, color_for):
    klass = color_for(level_out)
    st.markdown(
        f"""<div class="sf-result-{klass}">
              <div style="font-size:14px;opacity:0.85;">{caption}</div>
              <div class="sf-result-value">{pred:,.2f}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="text-align:center;margin-top:8px;">
              <span class="sf-risk-badge">{level_out}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def risk_color_fao(level):
    return {"LOW": "low", "MEDIUM": "med", "HIGH": "high"}[level]


def risk_color_demand(level):
    return {"LOW": "low", "MODERATE": "med", "HIGH": "high", "VERY HIGH": "high"}[level]


# ============================================================
# LOCAL FACTORS (what influenced THIS prediction)
# ============================================================

def encode_value(encoder_map, category, value):
    mapping = encoder_map.get(category, {})
    if value in mapping:
        return mapping[value]
    for key, code in mapping.items():
        if key.lower() == value.lower():
            return code
    return 0


def clean_feature_name(name):
    """Turn sklearn/pipeline feature names (cat__country, country_encoded,
    lag_1, dayofweek, ...) into human-readable labels."""
    n = str(name)
    n = re.sub(r"^(cat|num|bin)__", "", n)
    n = n.replace("_encoded", "")
    n = re.sub(r"^lag_(\d+)$", r"sales_lag_\1", n)
    n = n.replace("rolling_mean_3", "rolling_mean_3d")
    n = n.replace("dayofweek", "day_of_week")
    n = re.sub(r"_+", " ", n).strip()
    return n.title()


# ============================================================
# CASCADING DATASET FILTERS (dropdowns only offer valid combinations)
# ============================================================

def cascade_options(df, filters, column):
    """Values of `column` still available given `filters` = [(col, value), ...].
    Returns [] when no rows match (e.g. a parent dropdown isn't picked yet)."""
    sub = df
    for col, val in filters:
        if val is None or (isinstance(val, str) and not val.strip()):
            return []
        if col not in sub.columns:
            return []
        sub = sub[sub[col] == val]
    if sub.empty:
        return []
    opts = sub[column].dropna().unique().tolist()
    if column == "year":
        return sorted(int(v) for v in opts)
    return sorted((str(v).strip() for v in opts), key=str.lower)


def pick(options, label, key):
    """Dependent selectbox: only offers valid options and resets safely if the
    previously stored value is no longer in the (shrinking) options list."""
    if not options:
        st.session_state.pop(key, None)
        return None
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.selectbox(label, options, key=key)


def loo_factors_fao(pipeline, input_df, df):
    """Fallback local attribution (leave-one-out) for Module 1."""
    cols = ["commodity", "country", "food_supply_stage", "year"]
    ref = {}
    for c in cols:
        ref[c] = int(df[c].median()) if c == "year" else df[c].mode()[0]
    base = float(np.expm1(pipeline.predict(pd.DataFrame([ref])[cols])[0]))
    rows = []
    for c in cols:
        mod = input_df.copy()
        mod[c] = ref[c]
        pred_v = float(np.expm1(pipeline.predict(mod)[0]))
        rows.append({"feature": clean_feature_name(c), "shap_value": pred_v - base, "absolute_shap_value": abs(pred_v - base)})
    out = pd.DataFrame(rows).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
    out["shap_value"] = out["shap_value"].round(4)
    out["absolute_shap_value"] = out["absolute_shap_value"].round(4)
    return out


def loo_factors_demand(model, model_input_df, df, encoder_map):
    """Fallback local attribution (leave-one-out) for Module 2."""
    cols = list(model_input_df.columns)
    med_demand = float(df["demand"].median())
    ref = {}
    for c in cols:
        if c in ("lag_1", "lag_2", "lag_3", "rolling_mean_3"):
            ref[c] = med_demand
        elif c == "year":
            ref[c] = int(df[c].median())
        elif c == "month":
            ref[c] = int(df["month"].median()) if "month" in df else 1
        elif c == "day":
            ref[c] = int(df["day"].median()) if "day" in df else 1
        elif c == "dayofweek":
            ref[c] = int(df["dayofweek"].median()) if "dayofweek" in df else 0
        elif c == "weekend":
            ref[c] = int(df["weekend"].mode()[0]) if "weekend" in df else 0
        elif c == "country_encoded":
            ref[c] = encode_value(encoder_map, "country", df["country"].mode()[0])
        elif c == "commodity_encoded":
            ref[c] = encode_value(encoder_map, "commodity", df["commodity"].mode()[0])
    base = float(model.predict(pd.DataFrame([ref])[cols])[0])
    rows = []
    for c in cols:
        mod = model_input_df.copy()
        mod[c] = ref[c]
        pred_v = float(model.predict(mod)[0])
        rows.append({"feature": clean_feature_name(c), "shap_value": pred_v - base, "absolute_shap_value": abs(pred_v - base)})
    out = pd.DataFrame(rows).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
    out["shap_value"] = out["shap_value"].round(2)
    out["absolute_shap_value"] = out["absolute_shap_value"].round(2)
    return out


def shap_factors_fao(pipeline, input_df, df):
    """SHAP values for a single Module-1 prediction; falls back to LOO."""
    if not SHAP_AVAILABLE:
        return loo_factors_fao(pipeline, input_df, df), "loo"
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        X_t = preprocessor.transform(input_df)
        X_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
        feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X_dense, check_additivity=False))
        if values.ndim == 3:
            sample = values[0, :, 0]
        elif values.ndim == 2:
            sample = values[0]
        else:
            sample = values
        n = min(len(feature_names), len(sample))
        out = pd.DataFrame({
            "feature": [clean_feature_name(x) for x in feature_names[:n]],
            "shap_value": sample[:n],
            "absolute_shap_value": np.abs(sample[:n]),
        }).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
        return out, "shap"
    except Exception:
        return loo_factors_fao(pipeline, input_df, df), "loo"


def shap_factors_demand(model, model_input_df, df, encoder_map):
    """SHAP values for a single Module-2 prediction; falls back to LOO."""
    if not SHAP_AVAILABLE:
        return loo_factors_demand(model, model_input_df, df, encoder_map), "loo"
    try:
        feature_names = list(model_input_df.columns)
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(model_input_df, check_additivity=False))
        if values.ndim == 3:
            sample = values[0, :, 0]
        elif values.ndim == 2:
            sample = values[0]
        else:
            sample = values
        n = min(len(feature_names), len(sample))
        out = pd.DataFrame({
            "feature": [clean_feature_name(x) for x in feature_names[:n]],
            "shap_value": sample[:n],
            "absolute_shap_value": np.abs(sample[:n]),
        }).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
        return out, "shap"
    except Exception:
        return loo_factors_demand(model, model_input_df, df, encoder_map), "loo"


def loo_factors_spoilage(model, input_df, df):
    """Fallback local attribution (leave-one-out) for Module 3."""
    cols = SPOIL_FEATURES
    ref = {}
    for c in cols:
        if c in SPOIL_MODE_COLS:
            ref[c] = int(df[c].mode()[0])
        else:
            ref[c] = float(df[c].median())
    base = float(model.predict_proba(pd.DataFrame([ref])[cols])[:, 1][0])
    rows = []
    for c in cols:
        mod = input_df.copy()
        mod[c] = ref[c]
        pred_v = float(model.predict_proba(mod)[:, 1][0])
        rows.append({"feature": clean_feature_name(c), "shap_value": pred_v - base,
                     "absolute_shap_value": abs(pred_v - base)})
    out = pd.DataFrame(rows).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
    out["shap_value"] = out["shap_value"].round(4)
    out["absolute_shap_value"] = out["absolute_shap_value"].round(4)
    return out


def shap_factors_spoilage(model, input_df, df):
    """SHAP values for a single Module-3 prediction; falls back to LOO."""
    if not SHAP_AVAILABLE:
        return loo_factors_spoilage(model, input_df, df), "loo"
    try:
        feature_names = list(input_df.columns)
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(input_df, check_additivity=False)
        if isinstance(raw, list):
            sample = raw[1][0]
        elif raw.ndim == 3:
            sample = raw[0, :, 1]
        elif raw.ndim == 2:
            sample = raw[0]
        else:
            sample = raw
        n = min(len(feature_names), len(sample))
        out = pd.DataFrame({
            "feature": [clean_feature_name(x) for x in feature_names[:n]],
            "shap_value": sample[:n],
            "absolute_shap_value": np.abs(sample[:n]),
        }).sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)
        return out, "shap"
    except Exception:
        return loo_factors_spoilage(model, input_df, df), "loo"


def render_local_shap(factors, note, method="shap"):
    """Show the 'what influenced this prediction' chart + table."""
    if factors is None or factors.empty:
        st.info("No factor analysis could be generated.")
        return

    if method == "loo":
        note = ("SHAP library blocked by Windows policy — showing local attribution: "
                "how much each factor shifts the prediction vs its typical value. " + (note or ""))
    if note:
        st.markdown(f'<p class="sf-hint">{note}</p>', unsafe_allow_html=True)

    top = factors.head(10).sort_values("shap_value")

    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("shap_value:Q", title="SHAP contribution"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.shap_value > 0,
                alt.value("#1d6b4d"),
                alt.value("#b71c1c"),
            ),
            tooltip=["feature:N", "shap_value:Q"],
        )
        .properties(title="What influenced this prediction", height=300)
    )
    st.altair_chart(chart, width="stretch")

    table = factors.copy()
    table["impact"] = np.where(
        table["shap_value"] >= 0,
        "Increases prediction",
        "Decreases prediction",
    )
    table = table.rename(columns={
        "feature": "Factor",
        "shap_value": "SHAP value",
        "impact": "Effect",
    })[["Factor", "SHAP value", "Effect"]].head(8)
    st.dataframe(table, hide_index=True, width="stretch")


# ============================================================
# PDF REPORT BUILDER (for the Download button)
# ============================================================

@st.cache_data(show_spinner=False)
def factors_pdf_bytes(module_title, inputs, prediction_text, level, factors):
    """Build a clean single-page PDF report (inputs + result + factor table/chart).
    `inputs` is an iterable of (label, value) pairs."""
    buf = io.BytesIO()
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(0.10, 0.965, "SustainFlow AI", fontsize=17, fontweight="bold",
             color="#1d6b4d")
    fig.text(0.10, 0.93, module_title, fontsize=13, color="#444444")

    inp_block = "\n".join(f"{k}: {v}" for k, v in inputs)
    ax_info = fig.add_axes([0.10, 0.76, 0.80, 0.14], frame_on=False)
    ax_info.axis("off")
    ax_info.text(0.0, 1.0, inp_block, fontsize=10, family="monospace",
                 va="top", color="#333333")
    ax_info.text(0.62, 1.0, f"Prediction:  {prediction_text}\n\n"
                            f"Risk level:  {level}",
                 fontsize=12, va="top", color="#333333",
                 fontweight="bold")

    top = factors.head(10).sort_values("shap_value")
    ax_chart = fig.add_axes([0.10, 0.18, 0.46, 0.52])
    colors = ["#1d6b4d" if v >= 0 else "#b71c1c" for v in top["shap_value"]]
    ax_chart.barh(top["feature"], top["shap_value"], color=colors, height=0.6)
    ax_chart.set_title("What influenced this prediction (SHAP contribution)",
                       fontsize=11, loc="left", color="#333333")
    ax_chart.axvline(0, color="#666666", linewidth=0.8)
    for s in ("top", "right"):
        ax_chart.spines[s].set_visible(False)
    ax_chart.tick_params(labelsize=9)
    ax_chart.set_xlabel("SHAP contribution", fontsize=9, color="#555555")

    show = factors.head(8).copy()
    show["impact"] = np.where(show["shap_value"] >= 0,
                              "Increases", "Decreases")
    cells = show[["feature", "shap_value", "impact"]].to_numpy()
    ax_table = fig.add_axes([0.60, 0.26, 0.42, 0.44], frame_on=False)
    ax_table.axis("off")
    out = ax_table.table(
        cellText=cells,
        colLabels=["Factor", "SHAP value", "Effect"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.42, 0.28, 0.30],
    )
    out.scale(1.0, 1.4)
    for (r, c), cell in out.get_celld().items():
        cell.set_fontsize(8)
        if r == 0:
            cell.set_facecolor("#e8f2ec")
            cell.set_text_props(fontweight="bold", color="#13503d")
    fig.text(0.10, 0.10, "SUSTAINFLOW AI · Sustainable Food Supply Chains",
             fontsize=8, color="#9aa8a0")

    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def download_report(module_title, inputs, prediction_text, level, factors, fname,
                    key_suffix=""):
    try:
        data = factors_pdf_bytes(module_title, tuple(inputs.items()),
                                 prediction_text, level, factors)
    except Exception:
        st.info("Report could not be generated.")
        return
    st.download_button(
        "⬇ Download SHAP report (PDF)",
        data=data,
        file_name=fname,
        mime="application/pdf",
        key=f"dl_{key_suffix or module_title}",
        type="primary",
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="stSidebarWelcome">Welcome to SustainFlow AI</div>
            <div class="stSidebarBrand">
                🌱 SustainFlow AI
                <small>Sustainable Food Supply Chains</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Navigate — one click opens a module")
        current = st.session_state.get("nav_page", NAV[0])
        for i, target in enumerate(NAV):
            active = target == current
            if st.button(
                target,
                key=f"nav_sb_{i}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state["nav_page"] = target

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.caption("Appearance")
        if st.button("🌗 Toggle Theme", key="theme_toggle", use_container_width=True):
            st.session_state["theme"] = "Dark" if st.session_state.get("theme", "Light") == "Light" else "Light"
        page = st.session_state.get("nav_page", NAV[0])
        return page


# ============================================================
# PAGE: HOME
# ============================================================

def page_home():
    hero(
        "SustainFlow AI",
        "Smart food-waste intelligence that predicts loss, forecasts demand, detects spoilage risk, and guides prevention — so less food goes to waste across the supply chain.",
        tag="AI · Explainable · Food Waste Reduction",
    )

    section("About SustainFlow AI")
    st.markdown(
        '<div class="sf-card">'
        '<p class="sf-muted" style="font-size:20px; font-weight:700; line-height:1.9;">'
        '<b>SustainFlow AI</b> is an AI-powered decision-support dashboard built to tackle food waste across the '
        'agricultural supply chain. It brings together machine learning, explainable AI, and interactive visualizations '
        'so businesses, retailers, and supply-chain managers can see where food loss happens, why it happens, and what '
        'to do about it — before good food goes to waste.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Why It Is Used")
    st.markdown(
        '<div class="sf-card">'
        '<p class="sf-muted" style="font-size:20px; font-weight:700; line-height:1.9;">'
        'Roughly one-third of all food produced globally is lost or wasted every year. SustainFlow AI is used to:<br><br>'
        '<b>• Predict expected food-loss percentage</b> for any commodity, country, and supply stage<br>'
        '<b>• Forecast future demand</b> in units so stock is ordered correctly<br>'
        '<b>• Detect spoilage risk</b> before shipments deteriorate<br>'
        '<b>• Get AI prevention actions</b> — markdown, reroute, reorder & FIFO — with projected dollar savings<br><br>'
        'The goal is simple: catch waste early, save money, and keep more food on tables.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("How It Works")
    st.markdown(
        '<div class="sf-card">'
        '<p class="sf-muted" style="font-size:20px; font-weight:700; line-height:1.9;">'
        'SustainFlow AI follows a 4-step pipeline using the same food items (201 commodities, 159 countries) across every module:<br><br>'
        '<b>Step 1 — Food Loss Prediction:</b> enter a commodity, country, supply stage and year to get the expected loss %.<br>'
        '<b>Step 2 — Demand Forecast:</b> predict how many units will be needed using lag and rolling-window features.<br>'
        '<b>Step 3 — Spoilage Detection:</b> check whether a shipment is at risk of spoiling.<br>'
        '<b>Step 4 — AI Recommendation Engine:</b> receive an auto-diagnosis and tailored prevention actions with estimated savings.<br><br>'
        'Every prediction includes explainability (SHAP / local attribution) so you can see exactly which factors drove the result.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("Technologies Used")
    tech = [
        ("Python 3.12", "Core programming language"),
        ("Streamlit 1.64", "Interactive web dashboard framework"),
        ("scikit-learn 1.9", "Machine learning models (Random Forest, Gradient Boosting)"),
        ("LightGBM 4.7", "High-performance demand forecasting"),
        ("SHAP 0.52", "Explainable AI — what influenced each prediction"),
        ("pandas / numpy", "Data processing and numerical computation"),
        ("joblib", "Saving and loading trained models"),
        ("Altair / Matplotlib", "Charts, graphs and PDF reports"),
    ]
    cols = st.columns(2)
    for i, (tool, desc) in enumerate(tech):
        with cols[i % 2]:
            st.markdown(
                f'<div class="sf-card" style="padding:18px 22px; margin-bottom:12px;">'
                f'<div style="font-weight:900; font-size:20px; color:#047857;">{tool}</div>'
                f'<div class="sf-muted" style="font-size:19px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    section("How It Can Help")
    st.markdown(
        '<div class="sf-card">'
        '<p class="sf-muted" style="font-size:20px; font-weight:700; line-height:1.9;">'
        '<b>For businesses and retailers:</b> reduce spoilage, recover lost revenue, and optimize ordering.<br>'
        '<b>For supply-chain managers:</b> identify high-risk stages and commodities before loss occurs.<br>'
        '<b>For sustainability teams:</b> measure and cut food waste with data-backed actions.<br>'
        '<b>For everyone:</b> make smarter, faster decisions with transparent, explainable AI — not a black box.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    section("The 4 Modules")
    st.markdown(
        '<p class="sf-muted" style="font-size:19px; font-weight:700; line-height:1.8;">'
        'Each module works on the same food items so you can follow one item from loss prediction all the way to prevention advice.'
        '</p>',
        unsafe_allow_html=True,
    )

    module_cards = [
        ("📉 Module 1 · Food Loss Prediction",
         "Random Forest predicts expected food-loss % for any commodity, country, supply stage and year — with SHAP explainability.",
         "FAO dataset · 25,950 records · 201 commodities · 159 countries",
         NAV[1], "nav_m1"),
        ("📦 Module 2 · Demand Forecast (Units)",
         "LightGBM forecasts future demand in units using lag history and rolling-window features — outperforming a linear baseline.",
         "Demand dataset · monthly records · R² 0.94",
         NAV[2], "nav_m2"),
        ("🛑 Module 3 · Spoilage Detection",
         "Gradient Boosting detects shelf-life fragility and spoilage risk % (LOW → CRITICAL) from operational signals — leakage-free.",
         "Spoilage dataset · leakage-free · ROC-AUC 0.76",
         NAV[3], "nav_m3"),
        ("💡 Module 4 · AI Recommendation Engine",
         "AI advisor reads loss, demand and spoilage risk, auto-diagnoses the problem and generates prevention actions with projected savings.",
         "Decision matrix · markdown, reroute, reorder & FIFO",
         NAV[4], "nav_m4"),
    ]
    mc = st.columns(2)
    for i, (title, desc, meta, target, key) in enumerate(module_cards):
        with mc[i % 2]:
            st.markdown(
                f'<div class="sf-card">'
                f'<div style="font-weight:900; font-size:20px; color:#0f172a; margin-bottom:8px;">{title}</div>'
                f'<div class="sf-muted" style="font-size:18px; line-height:1.7; margin-bottom:10px;">{desc}</div>'
                f'<div class="sf-chip" style="font-size:13px; margin-bottom:12px;">{meta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title.split('·')[1].strip()}", key=key, use_container_width=True):
                st.session_state["nav_page"] = target
                st.rerun()

    section("Developed By")
    st.markdown(
        '<div class="sf-card" style="text-align:center; padding:30px 24px;">'
        '<div style="font-weight:900; font-size:24px; color:#0f172a; margin-bottom:14px;">SustainFlow AI</div>'
        '<div style="font-size:19px; font-weight:700; color:#1e5e4a; margin-bottom:6px;">Built with ❤️ to reduce food waste across supply chains</div>'
        '<div style="font-size:18px; font-weight:900; color:#0f172a; margin-top:16px; margin-bottom:10px;">Developed by</div>'
        '<div style="display:flex; justify-content:center; gap:28px; flex-wrap:wrap;">'
        '<div style="font-size:18px; font-weight:800; color:#0f172a;">Sanath Shetty</div>'
        '<div style="font-size:18px; font-weight:800; color:#0f172a;">Kshiti Shetty</div>'
        '<div style="font-size:18px; font-weight:800; color:#0f172a;">Misha Shetty</div>'
        '<div style="font-size:18px; font-weight:800; color:#0f172a;">Gauthami</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: MODULE 1
# ============================================================

def page_module1():
    hero(
        "Module 1 · Food Loss Prediction",
        "Random Forest predicts the expected food-loss percentage for any commodity, country, supply stage and year — with SHAP explainability.",
        tag="Random Forest · SHAP · FAO 25,950 rows",
    )

    guide("What to do here",
          ["🎯 Predict — choose commodity, country, stage & year → get the expected loss %, why (SHAP) and a PDF report.",
           "📊 Model Performance — model accuracy proof (RMSE, R², comparison charts).",
           "🔍 Explainability — reopen the reasoning of your last prediction."])

    df = load_fao_data()
    commodities = sorted(df["commodity"].dropna().astype(str).unique(), key=str.lower)
    countries = sorted(df["country"].dropna().astype(str).unique(), key=str.lower)
    stages = sorted(df["food_supply_stage"].dropna().astype(str).unique(), key=str.lower)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    tab_predict, tab_metrics, tab_xai = st.tabs(["🎯 Predict", "📊 Model Performance", "🔍 Explainability"])

    with tab_predict:
        tab_hint("Pick commodity → country → supply stage → year. Only combinations present in the FAO dataset are offered, so every choice matches real records.")
        st.markdown('<div class="sf-card sf-card-m1 sf-anim"><h3 style="font-size:26px;">📦 Select shipment details</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            commodity = pick(commodities, "Commodity", "m1_commodity")
        with col2:
            country = pick(cascade_options(df, [("commodity", commodity)], "country"), "Country", "m1_country")
        with col3:
            stage = pick(cascade_options(df, [("commodity", commodity), ("country", country)], "food_supply_stage"), "Supply Stage", "m1_stage")
        with col4:
            year = pick(cascade_options(df, [("commodity", commodity), ("country", country), ("food_supply_stage", stage)], "year"), "Year", "m1_year")
        st.markdown(
            f'<div class="sf-fade-in" style="margin-top:16px; font-size:19px; font-weight:800; color:#0f172a;">'
            f'✅ {len(df[(df["commodity"] == commodity) & (df["country"] == country) & (df["food_supply_stage"] == stage) & (df["year"] == year)])} matching record(s) in the FAO dataset</div>',
            unsafe_allow_html=True,
        )
        submitted = st.button("🌟 Predict Food Loss", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            pipeline = load_model(MODEL_FAO)
            input_df = pd.DataFrame([{
                "commodity": commodity,
                "country": country,
                "food_supply_stage": stage,
                "year": year,
            }])
            pred_log = pipeline.predict(input_df)[0]
            pred = float(np.expm1(pred_log))
            level = "LOW" if pred < 5 else ("MEDIUM" if pred < 15 else "HIGH")
            klass = risk_color_fao(level)

            section("Result")
            r1, r2 = st.columns([2, 1])
            with r1:
                result_card(pred, "Predicted Food Loss (%) ", level, 5, 15, risk_color_fao)
            with r2:
                st.markdown(
                    f"""<div class="sf-card sf-card-m1 sf-anim-2"><h3 style="font-size:24px;">📥 Input</h3>
                        <p class="sf-muted" style="font-size:18px;"><b>{commodity}</b><br>{country}<br>{stage}<br>{year}</p></div>""",
                    unsafe_allow_html=True,
                )

            section("What influenced this prediction")
            factors, method = shap_factors_fao(pipeline, input_df, df)
            render_local_shap(
                factors,
                "Positive values push loss higher · negative values pull it lower.",
                method,
            )
            report_inputs = {"Commodity": commodity, "Country": country,
                             "Supply Stage": stage, "Year": year}
            download_report(
                "Module 1 · Food Loss Prediction",
                report_inputs,
                f"{pred:,.2f}%",
                level,
                factors,
                "module1_food_loss_report.pdf",
                key_suffix="m1_predict",
            )
            st.session_state["m1_factors"] = factors
            st.session_state["m1_method"] = method
            st.session_state["m1_inputs"] = report_inputs
            st.session_state["m1_pred"] = f"{pred:,.2f}%"
            st.session_state["m1_level"] = level

            section("Historical loss for this combination")
            combo_data = df[
                (df["commodity"] == commodity) &
                (df["country"] == country) &
                (df["food_supply_stage"] == stage) &
                (df["year"] == year)
            ][["commodity", "country", "food_supply_stage", "year", "loss_percentage"]]
            if len(combo_data) > 0:
                st.success(f"✅ This combination exists in the FAO dataset — {len(combo_data)} record(s) found.")
                st.dataframe(combo_data, hide_index=True, width="stretch")
            else:
                st.info("❌ This combination does not exist in the FAO dataset.")

    with tab_metrics:
        tab_hint("Evidence that the model works — comparison chart (RMSE / R²) and the metrics table.")
        section("Model comparison")
        g1, g2 = st.columns(2)
        g1.image(os.path.join(RESULTS_DIR, "fao_model_comparison.png"),
                 caption="RMSE / R² comparison", width="stretch")
        g2.image(os.path.join(RESULTS_DIR, "fao_actual_vs_predicted.png"),
                 caption="Actual vs Predicted", width="stretch")

        section("Metrics summary")
        fao_metrics = parse_metrics(read_text(METRICS_FAO))
        if not fao_metrics.empty:
            st.dataframe(
                fao_metrics[~fao_metrics["section"].str.contains("RATIONALE", na=False)],
                hide_index=True, width="stretch",
            )

    with tab_xai:
        tab_hint("The SHAP reasons behind your last prediction — run Predict first to unlock your own result.")
        section("Local SHAP analysis · last prediction")
        m1_factors = st.session_state.get("m1_factors")
        if m1_factors is not None and not m1_factors.empty:
            render_local_shap(
                m1_factors,
                f"Last prediction: {st.session_state.get('m1_inputs')} → "
                f"{st.session_state.get('m1_pred')} ({st.session_state.get('m1_level')})",
                st.session_state.get("m1_method", "shap"),
            )
            download_report(
                "Module 1 · Food Loss Prediction",
                st.session_state.get("m1_inputs", {}),
                st.session_state.get("m1_pred", ""),
                st.session_state.get("m1_level", ""),
                m1_factors,
                "module1_food_loss_report.pdf",
                key_suffix="m1_xai",
            )
        else:
            st.info("Run a prediction on the Predict tab first — your SHAP graph will appear here.")

        section("Global SHAP importance")
        g1, g2 = st.columns(2)
        g1.image(os.path.join(RESULTS_DIR, "shap_global_importance.png"),
                 caption="Global SHAP importance", width="stretch")
        show_image(os.path.join(RESULTS_DIR, "shap_explanation.png"),
                   "Latest prediction SHAP")


# ============================================================
# PAGE: MODULE 2
# ============================================================

def page_module2():
    hero(
        "Module 2 · Demand Forecasting",
        "LightGBM forecasts product-level demand from lag history and time components — outperforming the linear baseline 2×.",
        tag="LightGBM · Lag Features · R² 0.94",
    )

    guide("What to do here",
          ["🎯 Predict — choose country, commodity & year → forecast demand in units + what drives it (SHAP).",
           "📊 Model Performance — proof LightGBM beats the simple linear baseline.",
           "🔍 Explainability — review the reasoning of your last demand forecast."])

    df = load_demand_data()
    countries = sorted(df["country"].dropna().astype(str).unique(), key=str.lower)
    commodities = sorted(df["commodity"].dropna().astype(str).unique(), key=str.lower)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    tab_predict, tab_metrics, tab_xai = st.tabs(["🎯 Predict", "📊 Model Performance", "🔍 Explainability"])

    with tab_predict:
        tab_hint("Pick country → commodity → year. Only combinations present in the demand dataset are offered, so every choice matches real records.")
        st.markdown('<div class="sf-card sf-card-m2 sf-anim"><h3 style="font-size:26px;">📦 Select product & period</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            country = pick(countries, "Country", "m2_country")
        with c2:
            commodity = pick(cascade_options(df, [("country", country)], "commodity"), "Commodity", "m2_commodity")
        with c3:
            year = pick(cascade_options(df, [("country", country), ("commodity", commodity)], "year"), "Year", "m2_year")
        st.markdown(
            f'<div class="sf-fade-in" style="margin-top:16px; font-size:19px; font-weight:800; color:#0f172a;">'
            f'✅ {len(df[(df["country"] == country) & (df["commodity"] == commodity) & (df["year"] == year)])} matching record(s) in the demand dataset</div>',
            unsafe_allow_html=True,
        )
        submitted = st.button("🌟 Predict Demand", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            model = load_model(MODEL_DEMAND)
            encoder_map = load_encoders(ENCODERS)

            combo_data = df[
                (df["country"] == country) &
                (df["commodity"] == commodity) &
                (df["year"] == year)
            ].sort_values("month")

            if len(combo_data) >= 3:
                lag_1 = float(combo_data.iloc[-1]["demand"])
                lag_2 = float(combo_data.iloc[-2]["demand"])
                lag_3 = float(combo_data.iloc[-3]["demand"])
                rolling = (lag_1 + lag_2 + lag_3) / 3
            elif len(combo_data) == 2:
                lag_1 = float(combo_data.iloc[-1]["demand"])
                lag_2 = float(combo_data.iloc[-2]["demand"])
                lag_3 = 0.0
                rolling = (lag_1 + lag_2) / 2
            elif len(combo_data) == 1:
                lag_1 = float(combo_data.iloc[-1]["demand"])
                lag_2, lag_3 = 0.0, 0.0
                rolling = lag_1
            else:
                lag_1 = lag_2 = lag_3 = rolling = 0.0

            def encode(category, value):
                mapping = encoder_map.get(category, {})
                if value in mapping:
                    return mapping[value]
                for key, code in mapping.items():
                    if key.lower() == value.lower():
                        return code
                return 0

            feature_names = ["country_encoded", "commodity_encoded", "year", "month",
                             "day", "dayofweek", "weekend", "lag_1", "lag_2", "lag_3", "rolling_mean_3"]
            month = int(combo_data.iloc[-1]["month"]) if len(combo_data) > 0 else 1
            day = int(combo_data.iloc[-1].get("day", 1)) if len(combo_data) > 0 else 1
            dayofweek = int(combo_data.iloc[-1].get("dayofweek", 0)) if len(combo_data) > 0 else 0
            weekend = int(combo_data.iloc[-1].get("weekend", 0)) if len(combo_data) > 0 else 0

            model_input = pd.DataFrame([{
                "country_encoded": encode("country", country),
                "commodity_encoded": encode("commodity", commodity),
                "year": int(year),
                "month": month,
                "day": day,
                "dayofweek": dayofweek,
                "weekend": weekend,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "rolling_mean_3": rolling,
            }])[feature_names]

            pred = float(model.predict(model_input)[0])
            level = "LOW" if pred < 500 else ("MODERATE" if pred < 1000 else ("HIGH" if pred < 2000 else "VERY HIGH"))

            section("Result")
            r1, r2 = st.columns([2, 1])
            with r1:
                result_card(pred, "Predicted Demand (units)", level, 500, 1000, risk_color_demand)
            with r2:
                st.markdown(
                    f"""<div class="sf-card"><h3>Input</h3>
                        <p class="sf-muted"><b>{commodity}</b><br>{country}<br>{year}<br>
                        lag_1: {lag_1:,.0f} · lag_2: {lag_2:,.0f}<br>rolling_3: {rolling:,.0f}</p></div>""",
                    unsafe_allow_html=True,
                )

            section("What influenced this prediction")
            factors, method = shap_factors_demand(model, model_input, df, encoder_map)
            render_local_shap(
                factors,
                "Positive values push demand higher · negative values pull it lower.",
                method,
            )
            report_inputs = {"Commodity": commodity, "Country": country, "Year": year,
                             "lag_1": f"{lag_1:,.0f}", "lag_2": f"{lag_2:,.0f}",
                             "rolling_3": f"{rolling:,.0f}"}
            download_report(
                "Module 2 · Demand Forecast",
                report_inputs,
                f"{pred:,.2f} units",
                level,
                factors,
                "module2_demand_report.pdf",
                key_suffix="m2_predict",
            )
            st.session_state["m2_factors"] = factors
            st.session_state["m2_method"] = method
            st.session_state["m2_inputs"] = report_inputs
            st.session_state["m2_pred"] = f"{pred:,.2f} units"
            st.session_state["m2_level"] = level

            section("Recent demand history")
            if len(combo_data) > 0:
                st.success(f"✅ This combination exists in the demand dataset — {len(combo_data)} record(s) found.")
                st.dataframe(combo_data[["month", "day", "demand"]], hide_index=True, width="stretch")
            else:
                st.info("❌ This combination does not exist in the demand dataset.")

    with tab_metrics:
        tab_hint("Shows the model is accurate (RMSE / R²) and clearly better than a simple linear model.")
        section("Model comparison")
        g1, g2 = st.columns(2)
        g1.image(os.path.join(RESULTS_DIR, "demand_model_comparison.png"),
                 caption="RMSE / R² comparison", width="stretch")
        g2.image(os.path.join(RESULTS_DIR, "demand_actual_vs_predicted.png"),
                 caption="Actual vs Predicted", width="stretch")

        section("Metrics summary")
        demand_metrics = parse_metrics(read_text(METRICS_DEMAND))
        if not demand_metrics.empty:
            st.dataframe(
                demand_metrics[~demand_metrics["section"].str.contains("RATIONALE", na=False)],
                hide_index=True, width="stretch",
            )

    with tab_xai:
        tab_hint("The reasons behind your last demand forecast — run Predict first to unlock your own result.")
        section("Local SHAP analysis · last prediction")
        m2_factors = st.session_state.get("m2_factors")
        if m2_factors is not None and not m2_factors.empty:
            render_local_shap(
                m2_factors,
                f"Last prediction: {st.session_state.get('m2_inputs')} → "
                f"{st.session_state.get('m2_pred')} ({st.session_state.get('m2_level')})",
                st.session_state.get("m2_method", "shap"),
            )
            download_report(
                "Module 2 · Demand Forecast",
                st.session_state.get("m2_inputs", {}),
                st.session_state.get("m2_pred", ""),
                st.session_state.get("m2_level", ""),
                m2_factors,
                "module2_demand_report.pdf",
                key_suffix="m2_xai",
            )
        else:
            st.info("Run a prediction on the Predict tab first — your SHAP graph will appear here.")

        section("Demand SHAP explanation")
        st.image(os.path.join(RESULTS_DIR, "demand_shap_explanation.png"),
                 caption="Demand SHAP explanation", width="stretch")


# ============================================================
# PAGE: MODULE 3
# ============================================================

SPOIL_FEATURES = ["loss_percentage", "avg_demand_log", "volatility_ratio",
                  "peak_month", "stage_code", "year"]

SPOIL_MODE_COLS = {"peak_month", "stage_code", "year"}


def spoilage_risk_level(score):
    if score >= 70.0:
        return "CRITICAL"
    if score >= 50.0:
        return "HIGH"
    if score >= 30.0:
        return "MEDIUM"
    return "LOW"


def risk_color_spoilage(level):
    return {"LOW": "low", "MEDIUM": "med", "HIGH": "high", "CRITICAL": "high"}[level]


def page_module3():
    hero(
        "Module 3 · Spoilage Detection",
        "A Gradient Boosting model predicts shelf-life fragility for the same food items as Module 1 & 2 — from loss %, demand, volatility, supply stage and season, trained against an external shelf-life reference (leakage-free; commodity identity and perishability are withheld so it learns a generalizable pattern).",
        tag="Gradient Boosting · Leakage-Free · Accuracy 73.2% · ROC-AUC 0.76",
    )

    guide("What to do here",
          ["🎯 Predict — the dropdowns share the SAME items as Module 1 & 2 (commodity, country, stage, year).",
           "Pick a real combination, press Detect Spoilage Risk → get risk %, why, and its spoilage record.",
           "📊 Model Performance & 🔍 Explainability — accuracy proof + what drives a risk prediction."])

    df = load_spoilage_data()
    commodities = sorted(df["commodity"].dropna().astype(str).unique(), key=str.lower)
    countries = sorted(df["country"].dropna().astype(str).unique(), key=str.lower)
    stages = sorted(df["food_supply_stage"].dropna().astype(str).unique(), key=str.lower)
    years = sorted(df["year"].dropna().astype(int).unique().tolist())

    tab_predict, tab_metrics, tab_xai = st.tabs(["🎯 Predict", "📊 Model Performance", "🔍 Explainability"])

    with tab_predict:
        tab_hint("Pick commodity → country → supply stage → year. Only combinations present in the spoilage dataset are offered, so every choice matches real records.")
        st.markdown('<div class="sf-card sf-card-m3 sf-anim"><h3 style="font-size:26px;">🛑 Select shipment details</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            commodity = pick(commodities, "Commodity", "m3_commodity")
        with c2:
            country = pick(cascade_options(df, [("commodity", commodity)], "country"), "Country", "m3_country")
        c3, c4 = st.columns(2)
        with c3:
            stage = pick(cascade_options(df, [("commodity", commodity), ("country", country)], "food_supply_stage"), "Supply Stage", "m3_stage")
        with c4:
            year = pick(cascade_options(df, [("commodity", commodity), ("country", country), ("food_supply_stage", stage)], "year"), "Year", "m3_year")
        st.markdown(
            f'<div class="sf-fade-in" style="margin-top:16px; font-size:19px; font-weight:800; color:#0f172a;">'
            f'✅ {len(df[(df["commodity"] == commodity) & (df["country"] == country) & (df["food_supply_stage"] == stage) & (df["year"] == year)])} matching record(s) in the spoilage dataset</div>',
            unsafe_allow_html=True,
        )
        submitted = st.button("🌟 Detect Spoilage Risk", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            model = load_model(MODEL_SPOILAGE)
            combo = df[
                (df["commodity"] == commodity) &
                (df["country"] == country) &
                (df["food_supply_stage"] == stage) &
                (df["year"] == year)
            ]
            if len(combo) == 0:
                st.info("❌ This combination does not exist in the spoilage dataset.")
            else:
                row = combo.iloc[0]
                input_df = pd.DataFrame([{f: float(row[f]) for f in SPOIL_FEATURES}])
                prob = float(model.predict_proba(input_df)[:, 1][0]) * 100.0
                level = spoilage_risk_level(prob)

                section("Result")
                r1, r2 = st.columns([2, 1])
                with r1:
                    result_card(prob, "Predicted Spoilage Risk (%) ", level, 30, 50, risk_color_spoilage)
                with r2:
                    st.markdown(
                        f"""<div class="sf-card sf-card-m3 sf-anim-2"><h3 style="font-size:24px;">📥 Input</h3>
                        <p class="sf-muted" style="font-size:18px;"><b>{commodity}</b><br>{country}<br>{stage}<br>{year}<br>
                        perishable: {'yes' if int(row['perishable']) else 'no'}<br>
                        FAO loss: {float(row['loss_percentage']):,.2f}%<br>
                        avg demand: {float(row['avg_demand']):,.0f} units/mo<br>
                        volatility: {float(row['volatility_ratio']):.2f}<br>
                        peak season: month {int(row['peak_month'])}</p></div>""",
                        unsafe_allow_html=True,
                    )

                section("What influenced this prediction")
                factors, method = shap_factors_spoilage(model, input_df, df)
                render_local_shap(
                    factors,
                    "Positive values push spoilage risk higher · negative values pull it lower.",
                    method,
                )
                report_inputs = {
                    "Commodity": commodity, "Country": country, "Supply Stage": stage, "Year": str(year),
                    "Perishable": "yes" if int(row["perishable"]) else "no",
                    "FAO Loss %": f"{float(row['loss_percentage']):.2f}",
                    "Avg Demand (units/mo)": f"{float(row['avg_demand']):,.0f}",
                    "Volatility": f"{float(row['volatility_ratio']):.2f}",
                    "Peak Month": str(int(row["peak_month"])),
                }
                download_report(
                    "Module 3 · Spoilage Detection",
                    report_inputs,
                    f"{prob:,.2f}%",
                    level,
                    factors,
                    "module3_spoilage_report.pdf",
                    key_suffix="m3_predict",
                )
                st.session_state["m3_factors"] = factors
                st.session_state["m3_method"] = method
                st.session_state["m3_inputs"] = report_inputs
                st.session_state["m3_pred"] = f"{prob:,.2f}%"
                st.session_state["m3_level"] = level

                section("Spoilage record for this combination")
                st.success(f"✅ This combination exists in the spoilage dataset — {len(combo)} record(s) found.")
                st.dataframe(
                    combo[["commodity", "country", "food_supply_stage", "year",
                           "perishable", "loss_percentage", "avg_demand",
                           "volatility_ratio", "spoilage_risk_score", "high_spoilage_risk"]],
                    hide_index=True, width="stretch",
                )

    with tab_metrics:
        tab_hint("The model predicts shelf-life fragility from operational signals (loss %, demand, volatility, stage, season) while withholding commodity identity and perishability — so no target leakage. ROC-AUC 0.76 shows a genuine generalizable signal; this tab shows which signals matter most.")
        section("Metrics summary")
        spoilage_metrics = parse_metrics(read_text(METRICS_SPOILAGE))
        if not spoilage_metrics.empty:
            st.dataframe(
                spoilage_metrics[~spoilage_metrics["section"].str.contains("RATIONAL", na=False)],
                hide_index=True, width="stretch",
            )

        section("Global feature importance")
        model = load_model(MODEL_SPOILAGE)
        imp = pd.DataFrame({
            "Feature": [clean_feature_name(f) for f in SPOIL_FEATURES],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True)
        chart = (
            alt.Chart(imp)
            .mark_bar(color="#1d6b4d")
            .encode(
                x=alt.X("importance:Q", title="Feature importance"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature:N", alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(title="How each signal drives spoilage risk", height=280)
        )
        st.altair_chart(chart, width="stretch")

    with tab_xai:
        tab_hint("The factors that pushed your last spoilage prediction up or down — run Predict first.")
        section("Local SHAP analysis · last prediction")
        m3_factors = st.session_state.get("m3_factors")
        if m3_factors is not None and not m3_factors.empty:
            render_local_shap(
                m3_factors,
                f"Last prediction: {st.session_state.get('m3_inputs')} → "
                f"{st.session_state.get('m3_pred')} ({st.session_state.get('m3_level')})",
                st.session_state.get("m3_method", "shap"),
            )
            download_report(
                "Module 3 · Spoilage Detection",
                st.session_state.get("m3_inputs", {}),
                st.session_state.get("m3_pred", ""),
                st.session_state.get("m3_level", ""),
                m3_factors,
                "module3_spoilage_report.pdf",
                key_suffix="m3_xai",
            )
        else:
            st.info("Run a prediction on the Predict tab first — your SHAP graph will appear here.")


# ============================================================
# PAGE: MODULE 4
# ============================================================

def estimate_unit_cost(commodity):
    c = str(commodity).lower()
    table = [("meat", 8.50), ("fish", 12.00), ("poultry", 6.50), ("milk", 3.20),
             ("tomatoes", 2.50), ("cucumbers", 2.60), ("bananas", 2.30), ("apples", 2.40),
             ("cherries", 3.50), ("cauliflower", 2.20), ("broccoli", 2.40), ("potatoes", 1.80),
             ("grapes", 3.00), ("rice", 2.00), ("maize", 1.60), ("eggs", 2.20)]
    for key, cost in table:
        if key in c:
            return cost
    return 4.00


def generate_recommendation(row):
    level = row["spoilage_risk_level"]
    score = float(row["spoilage_risk_score"])
    commodity = row["commodity"]
    market = f"{row['country']} ({row['food_supply_stage']})"
    units = float(row["projected_spoilage_units"])
    loss_usd = float(row["projected_spoilage_loss_usd"])
    if level == "CRITICAL":
        return {"action_type": "DYNAMIC_MARKDOWN",
                "title": f"Apply dynamic markdown on {commodity}",
                "details": f"CRITICAL spoilage risk ({score:.0f}%) for {commodity} in {market}. Mark down now to clear inventory before expiry.",
                "expected_waste_reduction_pct": 65,
                "financial_savings_usd": round(loss_usd * 0.65, 2)}
    if level == "HIGH":
        return {"action_type": "ROUTE & ROTATE",
                "title": f"Reroute & rotate {commodity} stock",
                "details": f"High spoilage risk ({score:.0f}%) for {commodity} in {market}. Reallocate {int(units * 0.5)} excess units to faster-selling channels.",
                "expected_waste_reduction_pct": 50,
                "financial_savings_usd": round(loss_usd * 0.50, 2)}
    if level == "MEDIUM":
        return {"action_type": "ORDER_CALIBRATION",
                "title": f"Trim reorder quantity for {commodity} by 20%",
                "details": f"Moderate spoilage risk ({score:.0f}%) for {commodity} in {market}. Cut the next reorder quantity to avoid overstock.",
                "expected_waste_reduction_pct": 35,
                "financial_savings_usd": round(loss_usd * 0.35, 2)}
    return {"action_type": "FIFO_MONITORING",
            "title": f"Standard FIFO rotation for {commodity}",
            "details": f"Low spoilage risk ({score:.0f}%) for {commodity} in {market}. Maintain FIFO shelf rotation and cold-chain temperature checks.",
            "expected_waste_reduction_pct": 15,
            "financial_savings_usd": round(loss_usd * 0.15, 2)}


@st.cache_data(show_spinner="Building Module 4 decision matrix…")
def build_decision_matrix():
    """Chain Module 1 (loss %) + Module 2 (demand) + Module 3 (spoilage risk)
    into an actionable decision matrix for the same food items."""
    df = load_spoilage_data().copy()
    spoilage_model = load_model(MODEL_SPOILAGE)

    df["predicted_loss_percentage"] = df["loss_percentage"].clip(lower=0).round(2)
    df["predicted_spoilage_prob"] = (
        spoilage_model.predict_proba(df[SPOIL_FEATURES])[:, 1] * 100.0
    )
    df["spoilage_risk_score"] = df["predicted_spoilage_prob"].round(2)
    df["spoilage_risk_level"] = df["spoilage_risk_score"].apply(spoilage_risk_level)

    df["estimated_monthly_demand"] = df["avg_demand"].round(0)
    df["projected_spoilage_units"] = (
        df["estimated_monthly_demand"] * (df["spoilage_risk_score"] / 100.0) * 0.35
    ).round(1)
    df["unit_cost"] = df["commodity"].apply(estimate_unit_cost)
    df["projected_spoilage_loss_usd"] = (
        df["projected_spoilage_units"] * df["unit_cost"]
    ).round(2)

    rec = df.apply(generate_recommendation, axis=1)
    df["action_type"] = [r["action_type"] for r in rec]
    df["recommendation_title"] = [r["title"] for r in rec]
    df["recommendation_details"] = [r["details"] for r in rec]
    df["expected_waste_reduction_pct"] = [r["expected_waste_reduction_pct"] for r in rec]
    df["financial_savings_usd"] = [r["financial_savings_usd"] for r in rec]
    return df.sort_values("spoilage_risk_score", ascending=False).reset_index(drop=True)


# ============================================================
# MODULE 4 · AI ADVISOR (diagnostic reasoning engine)
# ============================================================

def advisor_thresholds(matrix):
    return {
        "loss_p50": float(matrix["predicted_loss_percentage"].median()),
        "loss_p75": float(np.percentile(matrix["predicted_loss_percentage"], 75)),
        "demand_p25": float(np.percentile(matrix["estimated_monthly_demand"], 25)),
        "demand_p50": float(matrix["estimated_monthly_demand"].median()),
        "demand_p75": float(np.percentile(matrix["estimated_monthly_demand"], 75)),
    }


def item_diagnosis(r, t):
    """AI reasoning: read Mod 1 loss + Mod 2 demand + Mod 3 spoilage for one
    item, detect issues and auto-generate tailored prevention actions."""
    level = r["spoilage_risk_level"]
    score = float(r["spoilage_risk_score"])
    loss = float(r["predicted_loss_percentage"])
    demand = float(r["estimated_monthly_demand"])
    commodity, country, stage = r["commodity"], r["country"], r["food_supply_stage"]
    year = int(r["year"])

    issues = []
    if level in ("CRITICAL", "HIGH"):
        issues.append(("URGENT" if level == "CRITICAL" else "HIGH", "🛑",
                       f"Spoilage risk is {level} ({score:.0f}%)",
                       f"{commodity} at the '{stage}' stage is flagged as high risk; "
                       f"FAO loss here already averages {loss:.1f}%.",
                       ["Apply a markdown now to clear stock before expiry",
                        "Cut the next order quantity by 25–40%",
                        "Ship/store this batch first (first-in-first-out)",
                        "Verify temperature & humidity logs at the storage point"]))
    if loss >= t["loss_p75"]:
        issues.append(("HIGH", "📉",
                       f"Loss is above typical ({loss:.1f}% vs median {t['loss_p50']:.1f}%)",
                       f"At the '{stage}' stage this commodity damages more than most items.",
                       ["Audit handling & packaging at this stage",
                        "Reduce dwell time before the next processing step",
                        "Add temperature logging before transfer"]))
    if demand <= t["demand_p25"]:
        issues.append(("MEDIUM", "📦",
                       f"Demand is low ({demand:,.0f} vs typical {t['demand_p50']:,.0f} units/mo)",
                       "Weak selling means stock sits longer, which raises spoilage exposure.",
                       ["Reduce procurement by 30–40% for the next cycle",
                        "Bundle this item with a high-demand commodity",
                        "Run a targeted promotion to speed up turnover"]))
    elif demand >= t["demand_p75"]:
        issues.append(("MEDIUM", "🚀",
                       f"Demand is strong ({demand:,.0f} vs typical {t['demand_p50']:,.0f} units/mo)",
                       "Fast sales mean the risk is stock-out, not spoilage.",
                       ["Raise order quantity ~25% and add buffer stock",
                        "Secure supply early and stagger shipments",
                        "Prioritise fresh rotation to avoid shortages"]))

    priority = "LOW"
    actions = ["Maintain standard FIFO rotation",
               "Keep routine cold-chain temperature checks",
               "Review monthly — no urgency detected"]
    if issues:
        ranks = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        priority = min((i[0] for i in issues), key=lambda x: ranks[x])
        actions, seen = [], set()
        for _, _, _, _, act in issues:
            for a in act:
                if a not in seen:
                    seen.add(a)
                    actions.append(a)
        actions = actions[:6]
        top = issues[0]
        summary = (f"{commodity} in {country} ({stage}, {year}): {top[3]} "
                   f"{'Only ' if len(issues)==1 else ''}" +
                   " ".join(f"{i[2].split(' (')[0]}." for i in issues[1:] if i[2]))
    else:
        summary = (f"{commodity} in {country} ({stage}, {year}) shows no critical signals. "
                   "Continue with standard FIFO rotation and routine cold-chain checks.")

    return {
        "commodity": commodity, "country": country, "stage": stage, "year": year,
        "risk_level": level, "risk_score": score,
        "loss_pct": loss, "demand": demand,
        "perishable": int(r["perishable"]),
        "priority": priority, "summary": summary,
        "issues": issues, "actions": actions,
        "loss_usd": float(r["projected_spoilage_loss_usd"]),
        "savings_usd": float(r["financial_savings_usd"]),
        "recommendation_title": r["recommendation_title"],
    }


def page_module4():
    hero(
        "Module 4 · AI Recommendation Engine",
        "An AI advisor reads Module 1 (loss), Module 2 (demand) and Module 3 (spoilage) for any food item — diagnoses the problem and auto-generates the best ways to prevent it.",
        tag="AI Advisor · Auto-Diagnosis · Prevention Actions",
    )

    guide("What to do here",
          ["🧠 AI Advisor — pick any item → the AI reads Modules 1–3, tells you the problem and auto-writes prevention steps.",
           "📋 Decision Matrix — the full risk-ranked table with recommended action & savings.",
           "🎛 Simulator — test the markdown what-if."])

    cur_col1, cur_col2 = st.columns([2.2, 7.8])
    with cur_col1:
        st.selectbox(
            "💱 Currency",
            list(CURRENCIES.keys()),
            key="currency",
            index=list(CURRENCIES.keys()).index(st.session_state.get("currency", "$ USD")),
        )
    with cur_col2:
        st.markdown(
            '<div class="sf-muted sf-anim" style="padding-top:32px; font-size:20px;">'
            'Every money figure on this page — projected losses, potential recovery & savings — is shown '
            'in the currency you pick here.</div>',
            unsafe_allow_html=True,
        )

    matrix = None
    try:
        matrix = build_decision_matrix()
    except Exception:
        st.error("The decision matrix could not be built — make sure all models are trained "
                 "(run Codes/train_fao_model.py, Codes/train_demand_model.py and "
                 "Codes/train_spoilage_model.py first).")
        return

    critical = int((matrix["spoilage_risk_level"] == "CRITICAL").sum())
    high = int((matrix["spoilage_risk_level"] == "HIGH").sum())
    t = advisor_thresholds(matrix)
    n_low_demand = int((matrix["estimated_monthly_demand"] <= t["demand_p25"]).sum())
    n_high_loss = int((matrix["predicted_loss_percentage"] >= t["loss_p75"]).sum())
    total_loss = float(matrix["projected_spoilage_loss_usd"].sum())
    _cur_symbol, _cur_code, _cur_rate = current_currency()

    kpi_cols(
        st,
        [("CRITICAL + HIGH RISK", f"{critical + high}"),
         ("ABOVE-AVG LOSS ITEMS", f"{n_high_loss}"),
         ("LOW-DEMAND ITEMS", f"{n_low_demand}"),
         ("PROJECTED SPOILAGE LOSS", fmt_money(total_loss, 0))],
        icons=["🛑", "📉", "📦", "💰"],
    )

    tab_advisor, tab_matrix, tab_sim = st.tabs(
        ["🧠 AI Advisor", "📋 Decision Matrix", "🎛 Simulator"]
    )

    commodities = sorted(matrix["commodity"].dropna().astype(str).unique(), key=str.lower)
    countries = sorted(matrix["country"].dropna().astype(str).unique(), key=str.lower)
    stages = sorted(matrix["food_supply_stage"].dropna().astype(str).unique(), key=str.lower)
    years = sorted(matrix["year"].dropna().astype(int).unique().tolist())

    # ---------------- AI ADVISOR ----------------
    with tab_advisor:
        tab_hint("Pick commodity → country → supply stage → year. Only combinations present in the dataset are offered, so every choice matches real records.")
        st.markdown(
            '<div class="sf-card sf-anim"><h3 style="font-size:24px;">Select an item and let the AI analyse it</h3>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            commodity = pick(commodities, "Commodity", "m4_commodity")
        with c2:
            country = pick(cascade_options(matrix, [("commodity", commodity)], "country"), "Country", "m4_country")
        c3, c4 = st.columns(2)
        with c3:
            stage = pick(cascade_options(matrix, [("commodity", commodity), ("country", country)], "food_supply_stage"), "Supply Stage", "m4_stage")
        with c4:
            year = pick(cascade_options(matrix, [("commodity", commodity), ("country", country), ("food_supply_stage", stage)], "year"), "Year", "m4_year")
        st.markdown(
            f'<div class="sf-caption sf-fade-in" style="font-size:18px; font-weight:800; margin-top:16px;">'
            f'✅ {len(matrix[(matrix["commodity"] == commodity) & (matrix["country"] == country) & (matrix["food_supply_stage"] == stage) & (matrix["year"] == year)])} matching record(s) in the dataset</div>',
            unsafe_allow_html=True,
        )
        submitted = st.button("🧠 Analyse with AI", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            row = matrix[
                (matrix["commodity"] == commodity) &
                (matrix["country"] == country) &
                (matrix["food_supply_stage"] == stage) &
                (matrix["year"] == year)
            ]
            if len(row) == 0:
                st.info("❌ This combination does not exist in the spoilage dataset.")
            else:
                diag = item_diagnosis(row.iloc[0], t)

                section("AI Diagnosis")
                st.markdown(
                    f"""<div class="sf-card sf-bounce"><h3 style="font-size:24px;">🤖 {''.join(f'{i[1]} {i[2]}' for i in diag['issues'][:1])}</h3>
                    <p style="font-size:21px; font-weight:800; line-height:1.8;">{diag['summary']}</p>
                    <p class="sf-hint" style="font-size:19px;">Item: <b>{diag['commodity']}</b> · <b>{diag['country']}</b> · <b>{diag['stage']}</b> · <b>{diag['year']}</b> ·
                    priority: <b>{diag['priority']}</b></p></div>""",
                    unsafe_allow_html=True,
                )

                r = row.iloc[0]
                score_now = float(r["spoilage_risk_score"])
                level_now = str(r["spoilage_risk_level"])
                loss_now = float(r["predicted_loss_percentage"])
                demand_now = float(r["estimated_monthly_demand"])
                med_loss = t["loss_p50"]
                hi_loss = t["loss_p75"]
                med_dem = t["demand_p50"]
                lo_dem = t["demand_p25"]
                rules_fired = " · ".join(f"{i[1]} {i[2]}" for i in diag["issues"]) or "None — only routine care"
                verdict = f"priority <b>{diag['priority']}</b> — recommended action: <b>{diag['recommendation_title']}</b>"
                actions_note = " · ".join(diag["actions"])

                def _ad_badge(n):
                    return (f'<span style="background:linear-gradient(120deg,#0f3d2c,#1d6b4d); color:#ffffff; '
                            f'padding:4px 14px; border-radius:20px; font-size:14px; font-weight:900; '
                            f'letter-spacing:0.6px;">STEP {n}</span>')

                section("How the AI analysis works — step by step")
                st.markdown(
                    f"""<div class="sf-card sf-anim-2" style="line-height:1.9; font-size:20px;">
                    <p>{_ad_badge(1)} <b>The AI scores the danger</b><br>
                    Module 3's trained model reads <b>{commodity}</b> ({country}, {stage}, {year}) — loss <b>{loss_now:.1f}%</b>,
                    demand <b>{demand_now:,.0f}</b> units/mo, perishability, volatility and season — and answers:
                    <b>spoilage risk = {score_now:.0f}% → {level_now}</b>.</p>
                    <p>{_ad_badge(2)} <b>It compares to every item in the data</b><br>
                    Across the full dataset, the typical median loss is <b>{med_loss:.1f}%</b> and typical demand is
                    <b>{med_dem:,.0f}</b> units/mo. Anything above <b>{hi_loss:.1f}%</b> loss is a heavy-waster level;
                    demand below <b>{lo_dem:,.0f}</b> units/mo counts as a slow seller.</p>
                    <p>{_ad_badge(3)} <b>Decision rules that fired for this item</b><br>{rules_fired}</p>
                    <p>{_ad_badge(4)} <b>Verdict + action</b><br>{verdict}<br>
                    <span class="sf-muted" style="font-size:19px;">Prevent by: {actions_note}</span></p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if diag["issues"]:
                    section("Detected issues")
                    for idx, (lvl, icon, title, why, acts) in enumerate(diag["issues"]):
                        klass = "high" if lvl in ("URGENT", "HIGH") else "med"
                        anim_class = ["sf-anim-2", "sf-anim-3", "sf-anim-4"][idx % 3]
                        with st.container():
                            st.markdown(
                                f"""<div class="sf-result-{klass} {anim_class}" style="text-align:left;
                                    padding:24px 28px; margin-bottom:16px;">
                                    <div style="font-weight:900; font-size:20px;">{icon} {title}</div>
                                    <div style="font-size:19px; font-weight:700; margin:8px 0;">{why}</div>
                                    <div style="font-size:17px; font-weight:800;">
                                        <b>Prevent by:</b> {" · ".join(acts)}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                def _rule_row(act, tag):
                    return (f'<div style="display:flex; justify-content:space-between; align-items:center; '
                            f'gap:14px; padding:12px 0; border-bottom:2px solid rgba(128,128,128,0.25);">'
                            f'<span style="font-size:18px; font-weight:800;">• {act}</span>'
                            f'<span style="flex-shrink:0; background:linear-gradient(120deg,#0f3d2c,#1d6b4d); '
                            f'color:#ffffff; font-size:13px; font-weight:900; padding:5px 14px; '
                            f'border-radius:20px; white-space:nowrap;">{tag}</span></div>')

                seen_acts, rule_rows = set(), ""
                for lvl, icon, title, why, acts in diag["issues"]:
                    for a in acts:
                        if a not in seen_acts:
                            seen_acts.add(a)
                            rule_rows += _rule_row(a, f"{icon} {title.split(' (')[0]}")
                for a in diag["actions"]:
                    if a not in seen_acts:
                        seen_acts.add(a)
                        rule_rows += _rule_row(a, "🛡 routine (default)")

                section("What to do next (auto-generated)")
                st.markdown(
                    f"""<div class="sf-card sf-anim-3"><h3 style="font-size:24px;">📋 Recommended actions
                    <small style="font-weight:700; font-size:17px;"> — the badge on the right shows which rule each action came from</small></h3>
                    {rule_rows}
                    <p class="sf-muted" style="margin:12px 0 0; font-size:20px;"><b>{diag['recommendation_title']}</b><br>
                    Expected waste reduction: <b>{int(row.iloc[0]['expected_waste_reduction_pct'])}%</b> ·
                    recoverable: <b>{fmt_money(diag['savings_usd'], 2)}</b> of <b>{fmt_money(diag['loss_usd'], 2)}</b> at risk.</p></div>""",
                    unsafe_allow_html=True,
                )

                hi_dem = t["demand_p75"]
                risk_fired = level_now
                loss_fired = f"loss {loss_now:.1f}% ≥ {hi_loss:.1f}% (top 25%)" if loss_now >= hi_loss else f"loss {loss_now:.1f}% < {hi_loss:.1f}% — passed"
                dem_fired = (f"demand {demand_now:,.0f} ≤ {lo_dem:,.0f} (slow seller)" if demand_now <= lo_dem
                             else f"demand {demand_now:,.0f} ≥ {hi_dem:,.0f} (fast seller)" if demand_now >= hi_dem
                             else f"demand {demand_now:,.0f} in normal range — passed")
                st.markdown(
                    f"""<div class="sf-card sf-anim-4"><h3 style="font-size:24px;">🧾 Why these actions — the 3 checks</h3>
                    <p style="font-size:19px; font-weight:700; line-height:1.9; margin:0;">
                    🛑 <b>Risk check</b> → <b>{score_now:.0f}%</b> = {risk_fired} → chooses the main action above.<br>
                    📉 <b>Loss check</b> → {loss_fired} → {'adds the loss actions' if loss_now >= hi_loss else 'no loss actions added'}.<br>
                    📦 <b>Demand check</b> → {dem_fired} → {'adds the demand actions' if demand_now <= lo_dem or demand_now >= hi_dem else 'no demand actions added'}.<br>
                    <span class="sf-hint" style="font-size:18px;">The list above = actions from the checks that fired (deduplicated, max 6).</span></p></div>""",
                    unsafe_allow_html=True,
                )

    # ---------------- DECISION MATRIX ----------------
    with tab_matrix:
        tab_hint("Every item ranked by spoilage risk, with the recommended action and savings (currency set above).")
        section("Spoilage risk distribution")
        g1, g2 = st.columns([3, 2])
        by_comm = (matrix.groupby("commodity")["projected_spoilage_loss_usd"].sum()
                   .sort_values(ascending=False).head(15) * _cur_rate)
        with g1:
            chart = (
                alt.Chart(by_comm.reset_index())
                .mark_bar(color="#b71c1c")
                .encode(
                    x=alt.X("projected_spoilage_loss_usd:Q", title=f"{_cur_symbol} {_cur_code}"),
                    y=alt.Y("commodity:N", sort="-x", title=None),
                    tooltip=["commodity:N", alt.Tooltip("projected_spoilage_loss_usd:Q", format=",.2f")],
                )
                .properties(title="Top at-risk commodities", height=340)
            )
            st.altair_chart(chart, width="stretch")
        dist = matrix["spoilage_risk_level"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
        with g2:
            pie = (
                alt.Chart(dist.reset_index().rename(columns={"index": "Level", "spoilage_risk_level": "Count"}))
                .mark_arc(innerRadius=45)
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=alt.Color("Level:N", scale=alt.Scale(
                        domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        range=["#c62828", "#f57c00", "#eab308", "#15803d"])),
                    tooltip=["Level:N", "Count:Q"],
                )
                .properties(title="Spoilage risk distribution", height=300)
            )
            st.altair_chart(pie, width="stretch")

        section("Decision matrix — every item, ranked by risk")
        disp = matrix[["commodity", "country", "food_supply_stage", "year",
                       "predicted_loss_percentage", "estimated_monthly_demand",
                       "spoilage_risk_score", "spoilage_risk_level",
                       "recommendation_title", "financial_savings_usd"]].head(100).copy()
        disp = disp.rename(columns={
            "commodity": "Commodity", "country": "Country", "food_supply_stage": "Stage",
            "year": "Year", "predicted_loss_percentage": "Mod 1 Loss %",
            "estimated_monthly_demand": "Demand (units)", "spoilage_risk_score": "Risk Score",
            "spoilage_risk_level": "Level", "recommendation_title": "Recommended Action",
            "financial_savings_usd": "Savings",
        })
        disp["Savings"] = (disp["Savings"] * _cur_rate).round(2)
        st.dataframe(disp, hide_index=True, width="stretch")

        section(f"Export full decision matrix ({_cur_code})")
        export = matrix.copy()
        export["financial_savings_usd"] = (export["financial_savings_usd"] * _cur_rate).round(2)
        export["projected_spoilage_loss_usd"] = (export["projected_spoilage_loss_usd"] * _cur_rate).round(2)
        export = export.rename(columns={
            "financial_savings_usd": f"Savings ({_cur_code})",
            "projected_spoilage_loss_usd": f"Spoilage Loss ({_cur_code})",
        })
        st.download_button(
            "⬇ Download decision matrix (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="sustainflow_decision_matrix.csv",
            mime="text/csv",
            type="primary",
        )

    # ---------------- SIMULATOR ----------------
    with tab_sim:
        section("🎛 Markdown Simulator — what does this do?")
        st.markdown(
            """<div class="sf-card sf-anim"><h3 style="font-size:24px;">🤔 What is this simulator?</h3>
            <p class="sf-muted" style="font-size:20px;">Food at <b>CRITICAL / HIGH spoilage risk</b> will spoil and be thrown away soon.
            One proven rescue method is a <b>markdown</b> — a temporary price cut to make customers buy the food
            <i>before</i> it expires (e.g. "30% off ripe bananas").</p>
            <p class="sf-muted" style="font-size:20px;">This tab lets you test that idea: <b>drag the slider</b> to choose how big a
            discount (markdown %) you apply to ALL at-risk food at once, and it instantly shows how much money
            you save vs. what you still lose.</p>
            <p class="sf-hint" style="font-size:19px;">Formula: 🛑 at-risk value × (markdown % ÷ 100) = ♻️ recovered savings.
            The rest (at-risk value − recovered) is the money still lost because even discounted food sells out
            too late to be fully saved.</p></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="sf-card sf-anim-2"><h3 style="font-size:24px;">🎚 Drag this slider = choose your discount</h3>
            <p class="sf-muted" style="font-size:20px;">For example: <b>40% markdown</b> means "cut the price of all at-risk items by
            40% to clear the stock fast". Watch the ♻️ RECOVERED card grow as you drag it right.
            Real shops often start at 10% off, then increase the discount the closer the food gets to expiring.</p></div>""",
            unsafe_allow_html=True,
        )
        discount = st.slider("Markdown discount rate (how many % off the normal price)", 10, 70, 40, 5)
        st.markdown(
            f'<div class="sf-caption sf-fade-in" style="font-size:18px; font-weight:800; margin-top:12px;">'
            f'🖱 Drag the slider and watch how much more money gets recovered at every discount level.</div>',
            unsafe_allow_html=True,
        )
        at_risk = matrix[matrix["spoilage_risk_level"].isin(["CRITICAL", "HIGH"])]
        at_risk_loss = float(at_risk["projected_spoilage_loss_usd"].sum())
        recovered = at_risk_loss * (discount / 100.0)
        remaining = at_risk_loss - recovered
        kpi_cols(
            st,
            [("AT-RISK INVENTORY VALUE", fmt_money(at_risk_loss, 0)),
             (f"RECOVERED AT {discount}% MARKDOWN", fmt_money(recovered, 0)),
             ("REMAINING LOSS", fmt_money(remaining, 0)),
             ("AFFECTED RECORDS", f"{len(at_risk):,}")],
            icons=["📦", "♻️", "⚠️", "📋"],
        )
        st.markdown(
            f'<div class="sf-hint" style="font-size:18px;">CRITICAL + HIGH risk items respond best to dynamic markdowns (65% & 50% recovery in the baseline matrix).</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# APP ENTRY
# ============================================================

st.set_page_config(page_title="SustainFlow AI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

page = render_sidebar()

theme = st.session_state.get("theme", "Light")
st.markdown(CSS + (DARK_CSS if theme == "Dark" else ""), unsafe_allow_html=True)

if page == NAV[0]:
    page_home()
elif page == NAV[1]:
    page_module1()
elif page == NAV[2]:
    page_module2()
elif page == NAV[3]:
    page_module3()
else:
    page_module4()