"""Custom CSS. Only stable Streamlit test-ids / class names are targeted."""
from __future__ import annotations

import html
import re

import streamlit as st

CSS = """
<style>
:root{
  --qf-bg:#0b1020; --qf-surface:#121a30; --qf-surface-2:#18223d; --qf-border:#26324f;
  --qf-text:#e6e9f2; --qf-muted:#9aa6c4; --qf-accent:#818cf8; --qf-accent-2:#6366f1;
  --qf-ok:#34d399; --qf-warn:#fbbf24; --qf-bad:#f87171; --qf-radius:14px;
}
html, body, [data-testid="stAppViewContainer"], .stApp{
  font-family: Inter, "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp{ background: radial-gradient(1200px 600px at 85% -10%, rgba(99,102,241,.16), transparent 60%), var(--qf-bg); }
.block-container{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
h1,h2,h3{ letter-spacing:-.015em; }
header[data-testid="stHeader"]{ background: transparent; }

/* Sidebar */
[data-testid="stSidebar"]{ background: #0d1428; border-right:1px solid var(--qf-border); }
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }
.qf-brand{ display:flex; align-items:center; gap:.65rem; margin:.1rem 0 1.1rem; }
.qf-logo{ width:36px; height:36px; border-radius:10px; display:grid; place-items:center; font-weight:800; color:#fff;
  background: linear-gradient(135deg,#6366f1,#8b5cf6); box-shadow:0 6px 18px rgba(99,102,241,.4); }
.qf-brand b{ font-size:1.12rem; color:var(--qf-text); } .qf-brand small{ display:block; color:var(--qf-muted); font-size:.72rem; margin-top:-2px; }
[data-testid="stSidebar"] [role="radiogroup"]{ gap:.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  padding:.55rem .75rem; border-radius:10px; border:1px solid transparent; width:100%; cursor:pointer; transition:background .15s, border-color .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:rgba(129,140,248,.10); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:rgba(129,140,248,.18); border-color:rgba(129,140,248,.45); }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child{ display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label p{ font-weight:600; font-size:.95rem; }

/* Cards */
.qf-card{ background:var(--qf-surface); border:1px solid var(--qf-border); border-radius:var(--qf-radius); padding:1.1rem 1.25rem; margin-bottom:.9rem; }
.qf-card h4{ margin:0 0 .35rem; font-size:1rem; }
.qf-muted{ color:var(--qf-muted); font-size:.88rem; }
.qf-hero{ background: linear-gradient(135deg, rgba(99,102,241,.28), rgba(139,92,246,.12) 60%, rgba(18,26,48,.9));
  border:1px solid rgba(129,140,248,.35); border-radius:18px; padding:1.6rem 1.7rem; margin-bottom:1.2rem; }
.qf-hero h1{ margin:0 0 .35rem; font-size:1.9rem; } .qf-hero p{ color:var(--qf-muted); margin:0; max-width:60ch; }
.qf-metric{ background:var(--qf-surface); border:1px solid var(--qf-border); border-radius:var(--qf-radius); padding:1rem 1.15rem; height:100%; }
.qf-metric .l{ color:var(--qf-muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
.qf-metric .v{ font-size:1.95rem; font-weight:750; margin-top:.2rem; color:var(--qf-text); line-height:1.15; }
.qf-metric .s{ color:var(--qf-muted); font-size:.8rem; margin-top:.15rem; }

/* Badges / pills */
.qf-badge{ display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.03em; border:1px solid; }
.qf-easy{ color:#34d399; border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.10); }
.qf-medium{ color:#fbbf24; border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.10); }
.qf-hard{ color:#f87171; border-color:rgba(248,113,113,.5); background:rgba(248,113,113,.10); }
.qf-topic{ color:var(--qf-accent); border-color:rgba(129,140,248,.45); background:rgba(129,140,248,.10); }
.qf-pill{ display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem; border-radius:999px; font-weight:650; font-size:.82rem; border:1px solid; }
.qf-pill i{ width:8px; height:8px; border-radius:50%; display:inline-block; }
.qf-pill.on{ color:var(--qf-ok); border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.09); } .qf-pill.on i{ background:var(--qf-ok); box-shadow:0 0 8px var(--qf-ok); }
.qf-pill.off{ color:var(--qf-bad); border-color:rgba(248,113,113,.5); background:rgba(248,113,113,.09); } .qf-pill.off i{ background:var(--qf-bad); }
.qf-pill.warn{ color:var(--qf-warn); border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.09); } .qf-pill.warn i{ background:var(--qf-warn); }

/* Question rendering */
.qf-qhead{ display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-bottom:.55rem; }
.qf-qnum{ font-weight:800; color:var(--qf-accent); }
.qf-qtext{ font-size:1.08rem; font-weight:600; line-height:1.5; margin:.1rem 0 .7rem; }
.qf-opt{ display:flex; gap:.7rem; padding:.55rem .8rem; border:1px solid var(--qf-border); border-radius:10px; margin:.3rem 0; background:var(--qf-surface-2); }
.qf-opt b{ min-width:1.4rem; color:var(--qf-accent); }
.qf-opt.correct{ border-color:rgba(52,211,153,.7); background:rgba(52,211,153,.10); } .qf-opt.correct b{ color:var(--qf-ok); }
.qf-opt.wrong{ border-color:rgba(248,113,113,.7); background:rgba(248,113,113,.10); } .qf-opt.wrong b{ color:var(--qf-bad); }
.qf-expl{ border-left:3px solid var(--qf-accent-2); padding:.35rem .8rem; margin:.7rem 0 .2rem; color:#cfd6ea; background:rgba(99,102,241,.07); border-radius:0 8px 8px 0; font-size:.93rem; }
.qf-src{ border-left:3px solid #475569; padding:.3rem .8rem; margin-top:.5rem; color:var(--qf-muted); font-size:.85rem; font-style:italic; }
.qf-activity{ display:flex; justify-content:space-between; gap:1rem; padding:.55rem 0; border-bottom:1px solid var(--qf-border); font-size:.92rem; }
.qf-activity:last-child{ border-bottom:none; }

/* Inputs & buttons */
.stButton > button, .stDownloadButton > button{ border-radius:10px; font-weight:650; border:1px solid var(--qf-border); transition: transform .12s, border-color .15s, box-shadow .15s; }
.stButton > button:hover, .stDownloadButton > button:hover{ border-color:var(--qf-accent); transform:translateY(-1px); }
.stButton > button[kind="primary"]{ background:linear-gradient(135deg,#6366f1,#7c3aed); border:none; color:#fff; box-shadow:0 6px 18px rgba(99,102,241,.35); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, input:focus-visible, textarea:focus-visible{ outline:2px solid var(--qf-accent) !important; outline-offset:2px; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div{ border-radius:10px !important; }
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:var(--qf-radius); }
[data-testid="stTabs"] button{ font-weight:650; }


/* Option cards (quiz mode) */
[class*="st-key-qopts"] .stButton > button{ justify-content:flex-start; text-align:left; padding:.75rem 1rem; border-radius:12px;
  background:var(--qf-surface-2); font-weight:550; min-height:3rem; }
[class*="st-key-qopts"] .stButton > button:hover{ border-color:var(--qf-accent); background:rgba(129,140,248,.12); transform:translateX(3px); }
[class*="st-key-qopts"] .stButton > button[kind="primary"]{ background:rgba(99,102,241,.28); border:1px solid var(--qf-accent); box-shadow:none; }
[class*="st-key-qopts"] .stButton > button > div, [class*="st-key-qopts"] .stButton > button [data-testid="stMarkdownContainer"]{
  width:100%; justify-content:flex-start; text-align:left; }
[class*="st-key-qopts"] .stButton > button p{ text-align:left; width:100%; }

/* Answer reveal animations (instant feedback) */
@keyframes qf-pop{ 0%{ transform:scale(.96); opacity:.4 } 55%{ transform:scale(1.035) } 100%{ transform:scale(1); opacity:1 } }
@keyframes qf-glow{ 0%{ box-shadow:0 0 0 0 rgba(52,211,153,.65) } 100%{ box-shadow:0 0 0 16px rgba(52,211,153,0) } }
@keyframes qf-shake{ 0%,100%{ transform:translateX(0) } 20%{ transform:translateX(-7px) } 40%{ transform:translateX(7px) } 60%{ transform:translateX(-5px) } 80%{ transform:translateX(4px) } }
@keyframes qf-slide{ from{ transform:translateY(10px); opacity:0 } to{ transform:translateY(0); opacity:1 } }
@keyframes qf-icon{ 0%{ transform:scale(0) rotate(-45deg) } 70%{ transform:scale(1.35) rotate(8deg) } 100%{ transform:scale(1) rotate(0) } }
.qf-anim .qf-opt.correct{ animation: qf-pop .5s ease-out, qf-glow 1.1s ease-out .1s; }
.qf-anim .qf-opt.wrong{ animation: qf-shake .5s ease-in-out; }
.qf-anim .qf-expl{ animation: qf-slide .5s ease-out .35s both; }
.qf-verdict{ display:flex; align-items:center; gap:.7rem; padding:.75rem 1rem; border-radius:12px; margin:.8rem 0 .2rem; font-weight:650;
  animation: qf-slide .4s ease-out .15s both; }
.qf-verdict .qf-ico{ width:1.9rem; height:1.9rem; border-radius:50%; display:grid; place-items:center; font-size:1.05rem; color:#0b1020;
  animation: qf-icon .55s cubic-bezier(.2,.9,.3,1.3) .2s both; }
.qf-verdict.ok{ background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.55); color:#a7f3d0; } .qf-verdict.ok .qf-ico{ background:var(--qf-ok); }
.qf-verdict.bad{ background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.55); color:#fecaca; } .qf-verdict.bad .qf-ico{ background:var(--qf-bad); }


/* Step indicator (Generate page) */
.qf-steps{ display:flex; gap:.5rem; margin:.2rem 0 1rem; flex-wrap:wrap; }
.qf-step{ display:flex; align-items:center; gap:.5rem; padding:.35rem .8rem .35rem .4rem; border-radius:999px; font-size:.85rem; font-weight:650;
  border:1px solid var(--qf-border); color:var(--qf-muted); background:var(--qf-surface); }
.qf-step span{ width:1.35rem; height:1.35rem; border-radius:50%; display:grid; place-items:center; font-size:.72rem; background:var(--qf-surface-2); }
.qf-step.active{ color:var(--qf-text); border-color:var(--qf-accent); } .qf-step.active span{ background:var(--qf-accent-2); color:#fff; }
.qf-step.done{ color:var(--qf-ok); border-color:rgba(52,211,153,.4); } .qf-step.done span{ background:rgba(52,211,153,.2); color:var(--qf-ok); }
.qf-live{ animation: qf-slide .45s ease-out both; }

/* Responsive */
@media (max-width: 900px){
  .block-container{ padding: 1.2rem 1rem 3rem; }
  .qf-hero h1{ font-size:1.5rem; } .qf-metric .v{ font-size:1.55rem; }
}
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
"""



LIGHT_CSS = """
<style>
:root{
  --qf-bg:#f5f6fb; --qf-surface:#ffffff; --qf-surface-2:#f0f2fa; --qf-border:#dcdff0; --qf-text:#1b1f3b; --qf-muted:#5b6385;
  --qf-accent:#4f46e5; --qf-accent-2:#4f46e5; --qf-ok:#059669; --qf-warn:#b45309; --qf-bad:#dc2626;
}
.stApp{ background: radial-gradient(1200px 600px at 85% -10%, rgba(99,102,241,.10), transparent 60%), var(--qf-bg); }
[data-testid="stSidebar"]{ background:#ffffff; }
.qf-hero{ background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(139,92,246,.07) 60%, #ffffff); border-color:rgba(99,102,241,.3); }
.qf-expl{ color:#2a2f55; background:rgba(99,102,241,.06); }
.qf-src{ border-left-color:#c3c8e0; }
.qf-easy{ color:#047857; } .qf-medium{ color:#b45309; } .qf-hard{ color:#b91c1c; } .qf-topic{ color:#4338ca; }
.qf-pill.on{ color:#047857; } .qf-pill.off{ color:#b91c1c; } .qf-pill.warn{ color:#b45309; }
.qf-verdict.ok{ color:#065f46; } .qf-verdict.bad{ color:#991b1b; }
.qf-opt.correct b{ color:#047857; } .qf-opt.wrong b{ color:#b91c1c; }
[class*="st-key-qopts"] .stButton > button[kind="primary"]{ background:rgba(79,70,229,.12); }
.qf-step.done{ color:#047857; } .qf-step.done span{ color:#047857; }
</style>
"""


def is_light() -> bool:
    """True when the user picked Streamlit's light theme (menu -> Settings) or their system is light."""
    try:
        return st.context.theme.get("type") == "light"
    except Exception:  # older Streamlit or no browser context (tests)
        return False


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if is_light():
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def esc(text: object) -> str:
    """HTML-escape untrusted text (model output / uploaded material) before it goes into custom HTML."""
    return html.escape(str(text), quote=True)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")


def md_escape(text: str) -> str:
    """Neutralise markdown / LaTeX / emoji-shortcode / image syntax in untrusted labels."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def badge(difficulty: str) -> str:
    cls = {"Easy": "qf-easy", "Medium": "qf-medium", "Hard": "qf-hard"}.get(difficulty, "qf-medium")
    return f'<span class="qf-badge {cls}">{esc(difficulty)}</span>'
