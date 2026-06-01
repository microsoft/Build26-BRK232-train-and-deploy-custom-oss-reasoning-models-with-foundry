"""
Streamlit dashboard for inspecting slime rollout samples.

Usage:
    streamlit run dashboard.py -- --log-dir rollout_logs

Or set ROLLOUT_LOG_DIR env var.
"""

from __future__ import annotations

import argparse
import ast
import base64
import difflib
import hashlib
import html
import io
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import orjson  # type: ignore[import-not-found]
    _loads = orjson.loads  # 3-10x faster than stdlib json
    _ParseError = (json.JSONDecodeError, ValueError, orjson.JSONDecodeError)
except ImportError:
    _loads = json.loads
    _ParseError = (json.JSONDecodeError, ValueError)

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISPLAY_FLOAT_DECIMALS = 3
DIAGNOSTIC_FLOAT_DECIMALS = 4

FOUNDRY_PAGE_ICON_SVG = """<svg height="20" viewBox="0 0 32 32" width="20" xmlns="http://www.w3.org/2000/svg" role="presentation"><path clip-rule="evenodd" d="M20.4052 2C20.3713 2.04989 20.3403 2.10356 20.3119 2.15906C20.1753 2.42519 20.0629 2.80022 19.9685 3.2499C19.7794 4.15205 19.6545 5.3972 19.5714 6.7798C19.405 9.54716 19.405 12.8938 19.405 15.213V24.4338L19.4049 24.4698C19.3854 27.5153 16.8918 29.9806 13.8112 29.9999L13.7749 30H3.57642C3.18062 30 2.9073 29.6141 3.04346 29.2496C4.56004 25.1917 6.6982 19.4832 8.50404 14.6901C9.40697 12.2934 10.2268 10.1257 10.8442 8.50763C11.4636 6.88453 11.876 5.82419 11.9665 5.63239C12.2132 5.10978 12.6147 4.1951 13.1873 3.40856C13.7637 2.61659 14.4808 2.00001 15.3445 2H20.4052ZM29.2769 10.1842C29.4966 10.1842 29.6747 10.3603 29.6747 10.5775V17.6706L29.6745 17.7148C29.6504 19.5836 28.1106 21.0913 26.2147 21.0913H21.668C21.6778 21.0796 21.6872 21.0676 21.6966 21.0552C21.8605 20.8367 21.9531 20.526 21.9587 20.134L21.9589 20.0958V14.0817C21.9589 11.9291 23.7238 10.1842 25.9011 10.1842H29.2769ZM21.2532 2.14424C21.5631 2.14425 21.8986 2.38926 22.2468 2.88783C22.5881 3.37681 22.9111 4.06635 23.2065 4.85721C23.7783 6.3875 24.2354 8.26487 24.5265 9.71512C22.6354 10.2861 21.2595 12.0248 21.2595 14.0817V20.0782L21.2594 20.0921C21.2575 20.2355 21.2263 20.4039 21.1685 20.5329C21.1042 20.6758 21.0375 20.7121 20.9938 20.7121C20.9575 20.7121 20.8869 20.6826 20.7852 20.5652C20.6894 20.4549 20.5915 20.2961 20.4975 20.1117C20.3151 19.7539 20.1614 19.3273 20.0739 19.0482V15.213C20.0739 8.68733 20.3039 5.39271 20.5834 3.73209C20.7239 2.89797 20.8739 2.49601 20.9998 2.30459C21.0605 2.21243 21.1101 2.17748 21.1426 2.16241C21.1755 2.14714 21.207 2.14424 21.2532 2.14424Z" fill="#000000" fill-rule="evenodd"></path></svg>"""
FOUNDRY_PAGE_ICON = "data:image/svg+xml;base64," + base64.b64encode(
    FOUNDRY_PAGE_ICON_SVG.encode("utf-8")
).decode("ascii")

st.set_page_config(page_title="Foundry Rollout Browser", page_icon=FOUNDRY_PAGE_ICON, layout="wide")


FOUNDRY_THEME_CSS = """
<style>
:root {
    --foundry-bg: #f7f8fb;
    --foundry-surface: #ffffff;
    --foundry-surface-subtle: #faf9f8;
    --foundry-surface-muted: #f3f2f1;
    --foundry-stroke: #e1dfdd;
    --foundry-stroke-strong: #c8c6c4;
    --foundry-text: #242424;
    --foundry-text-muted: #616161;
    --foundry-text-subtle: #8a8886;
    --foundry-brand: #0f6cbd;
    --foundry-brand-hover: #115ea3;
    --foundry-brand-pressed: #0f548c;
    --foundry-teal: #00a3a3;
    --foundry-purple: #5b5fc7;
    --foundry-success: #107c10;
    --foundry-warning: #f7630c;
    --foundry-danger: #c50f1f;
    --foundry-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(15, 108, 189, 0.10), transparent 26rem),
        radial-gradient(circle at top right, rgba(91, 95, 199, 0.08), transparent 24rem),
        linear-gradient(180deg, #fbfbfd 0%, var(--foundry-bg) 38%, #ffffff 100%);
    color: var(--foundry-text);
    font-family: "Aptos", "Segoe UI Variable Text", "Segoe UI", sans-serif;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {
    color: var(--foundry-text);
    font-family: "Aptos Display", "Segoe UI Variable Display", "Segoe UI", sans-serif;
    letter-spacing: 0;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.62) 0%, rgba(248, 247, 246, 0.88) 45%, #f3f2f1 100%),
        radial-gradient(circle at 0 0, rgba(15, 108, 189, 0.10), transparent 12rem);
    border-right: 1px solid var(--foundry-stroke);
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 1rem 1rem 1.25rem;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(200, 198, 196, 0.72);
    margin: 0.9rem 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--foundry-text);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--foundry-text-muted);
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
    text-transform: uppercase;
}

[data-testid="stSidebar"] .stButton > button {
    border-radius: 6px;
    min-height: 2.25rem;
    width: 100%;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: rgba(255, 255, 255, 0.95);
    border-color: var(--foundry-stroke) !important;
    border-radius: 6px;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    background: rgba(255, 255, 255, 0.70);
    border: 1px solid rgba(225, 223, 221, 0.92);
    border-radius: 8px;
    padding: 0.35rem 0.45rem;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(225, 223, 221, 0.96);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    overflow: hidden;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: var(--foundry-text);
    font-weight: 600;
    min-height: 2.35rem;
}

.foundry-sidebar-brand {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(225, 223, 221, 0.96);
    border-radius: 10px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.05);
    overflow: hidden;
    padding: 0.85rem 0.85rem 0.9rem;
    position: relative;
}

.foundry-sidebar-brand::before {
    background: linear-gradient(90deg, #242424 0%, rgba(15, 108, 189, 0.45) 48%, rgba(0, 163, 163, 0.36) 100%);
    content: "";
    height: 3px;
    left: 0;
    position: absolute;
    right: 0;
    top: 0;
}

.foundry-brand-row {
    align-items: center;
    display: flex;
    gap: 0.75rem;
}

.foundry-product-mark {
    align-items: center;
    background: transparent;
    border: 0;
    color: #000000;
    display: inline-flex;
    flex: 0 0 auto;
    font-size: 1rem;
    height: 2rem;
    justify-content: center;
    position: relative;
    width: 2rem;
}

.foundry-brand-name {
    color: var(--foundry-text);
    font-size: 0.98rem;
    font-weight: 600;
    line-height: 1.15;
}

.foundry-brand-subtitle {
    color: var(--foundry-text-muted);
    font-size: 0.78rem;
    line-height: 1.25;
    margin-top: 0.1rem;
}

.foundry-source-card {
    background: rgba(250, 249, 248, 0.84);
    border: 1px solid var(--foundry-stroke);
    border-radius: 6px;
    margin-top: 0.85rem;
    padding: 0.55rem 0.6rem;
}

.foundry-source-label {
    color: var(--foundry-text-subtle);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1;
    margin-bottom: 0.34rem;
    text-transform: uppercase;
}

.foundry-source {
    color: var(--foundry-text-muted);
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 0.75rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.foundry-file-stats {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
}

.foundry-refresh-form {
    display: inline-flex;
    margin: 0;
}

.foundry-refresh-chip {
    align-items: center;
    appearance: none;
    background: #ffffff;
    border: 1px solid var(--foundry-stroke);
    border-radius: 999px;
    color: var(--foundry-text);
    cursor: pointer;
    display: inline-flex;
    flex: 0 0 auto;
    font-family: inherit;
    font-size: 0.86rem;
    height: 1.95rem;
    justify-content: center;
    line-height: 1;
    padding: 0;
    text-decoration: none;
    width: 1.95rem;
}

.foundry-refresh-chip:hover {
    background: var(--foundry-surface-subtle);
    border-color: var(--foundry-stroke-strong);
    color: var(--foundry-text);
    text-decoration: none;
}

.foundry-file-stats span,
.foundry-meta-pill {
    align-items: center;
    background: #ffffff;
    border: 1px solid var(--foundry-stroke);
    border-radius: 999px;
    color: var(--foundry-text-muted);
    display: inline-flex;
    font-size: 0.75rem;
    font-weight: 500;
    gap: 0.35rem;
    line-height: 1;
    padding: 0.35rem 0.55rem;
}

.foundry-file-stats strong {
    color: var(--foundry-text);
    font-weight: 700;
}

.foundry-sidebar-section-title {
    color: var(--foundry-text-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin: 0.2rem 0 0.45rem;
    text-transform: uppercase;
}

.foundry-sidebar-control-label {
    color: var(--foundry-text);
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.5rem 0 0.35rem;
}

.foundry-rollout-summary {
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(225, 223, 221, 0.96);
    border-radius: 8px;
    display: grid;
    gap: 0.4rem;
    grid-template-columns: 1fr 1fr;
    margin-top: 0.65rem;
    padding: 0.65rem 0.7rem;
}

.foundry-rollout-summary div {
    min-width: 0;
}

.foundry-rollout-summary span {
    color: var(--foundry-text-subtle) !important;
    display: block;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    margin-bottom: 0.24rem;
    text-transform: uppercase;
}

.foundry-rollout-summary strong {
    color: var(--foundry-text);
    display: block;
    font-size: 0.9rem;
    font-weight: 650;
    line-height: 1.15;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.foundry-app-header {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--foundry-stroke);
    border-radius: 8px;
    box-shadow: var(--foundry-shadow);
    margin-bottom: 1rem;
    overflow: hidden;
    padding: 1.15rem 1.25rem;
    position: relative;
}

.foundry-app-header::before {
    background: linear-gradient(90deg, var(--foundry-brand), var(--foundry-purple), var(--foundry-teal));
    content: "";
    height: 3px;
    left: 0;
    position: absolute;
    right: 0;
    top: 0;
}

.foundry-app-header h1 {
    color: var(--foundry-text);
    font-size: clamp(1.45rem, 2.3vw, 2.15rem);
    font-weight: 600;
    letter-spacing: 0;
    line-height: 1.12;
    margin: 0;
}

.foundry-app-header p {
    color: var(--foundry-text-muted);
    font-size: 0.95rem;
    line-height: 1.45;
    margin: 0.5rem 0 0;
    max-width: 68rem;
}

.foundry-header-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}

.foundry-meta-pill strong {
    color: var(--foundry-text);
    font-weight: 600;
}

.foundry-icon {
    color: currentColor;
    display: inline-block;
    fill: none;
    flex: 0 0 auto;
    height: 1em;
    line-height: 1;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 1.75;
    vertical-align: -0.14em;
    width: 1em;
}

.foundry-product-mark .foundry-icon {
    height: 1.1rem;
    position: relative;
    stroke-width: 1.9;
    width: 1.1rem;
    z-index: 1;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--foundry-stroke);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    padding: 0.85rem 1rem;
}

[data-testid="stMetricLabel"] {
    color: var(--foundry-text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--foundry-text);
    font-size: 1.55rem;
    font-weight: 600;
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    border-radius: 4px;
    font-weight: 600;
}

.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background: #ffffff;
    border: 1px solid var(--foundry-stroke-strong);
    color: var(--foundry-text);
}

.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background: #f5f5f5;
    border-color: var(--foundry-brand);
    color: var(--foundry-brand-hover);
}

[data-testid="stBaseButton-primary"] {
    background: var(--foundry-brand);
    border: 1px solid var(--foundry-brand);
    color: #ffffff;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--foundry-stroke);
    border-radius: 0;
    display: flex;
    gap: 0.125rem;
    overflow-x: auto;
    padding: 0 0.125rem;
}

.stTabs [data-baseweb="tab"] {
    align-items: center;
    border: 1px solid transparent;
    border-bottom: 0;
    border-radius: 6px 6px 0 0;
    color: var(--foundry-text-muted);
    display: inline-flex;
    font-weight: 600;
    min-height: 2.45rem;
    min-width: max-content;
    padding: 0.55rem 0.95rem;
    white-space: nowrap;
}

.stTabs [aria-selected="true"][data-baseweb="tab"] {
    background: #ffffff;
    border-color: var(--foundry-stroke);
    box-shadow: 0 -1px 0 #ffffff inset;
    color: var(--foundry-text);
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [role="tabpanel"] {
    margin-top: 0.85rem;
    padding-top: 0.15rem;
}

.stTabs [role="tabpanel"] [data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

.stTabs [role="tabpanel"] [data-testid="stVerticalBlock"] {
    gap: 0.9rem;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stExpander"],
[data-testid="stAlert"] {
    border-radius: 8px;
}

.foundry-tool-table-wrap {
    border: 1px solid var(--foundry-stroke);
    border-radius: 8px;
    overflow-x: auto;
}

.foundry-tool-table {
    border-collapse: collapse;
    min-width: 64rem;
    table-layout: fixed;
    width: 100%;
}

.foundry-tool-table th,
.foundry-tool-table td {
    border-bottom: 1px solid var(--foundry-stroke);
    border-right: 1px solid var(--foundry-stroke);
    color: var(--foundry-text);
    font-size: 0.86rem;
    line-height: 1.35;
    padding: 0.68rem 0.75rem;
    text-align: left;
    vertical-align: middle;
}

.foundry-tool-table th {
    background: #f8f8fa;
    color: var(--foundry-text-muted);
    font-weight: 500;
}

.foundry-tool-table tr:last-child td {
    border-bottom: 0;
}

.foundry-tool-table th:last-child,
.foundry-tool-table td:last-child {
    border-right: 0;
}

.foundry-tool-table-step {
    text-align: right !important;
}

.foundry-tool-name {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 0.85rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.foundry-tool-status {
    white-space: nowrap;
}

.foundry-tool-status-ok {
    color: var(--foundry-success) !important;
}

.foundry-tool-status-different,
.foundry-tool-status-missing-output,
.foundry-tool-status-unexpected-output {
    color: var(--foundry-warning) !important;
}

.foundry-tool-args {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 0.78rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid var(--foundry-stroke);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--foundry-stroke);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

[data-testid="stCodeBlock"],
pre,
code {
    border-radius: 6px;
    font-family: "Cascadia Mono", "Consolas", monospace;
}

hr {
    border-color: var(--foundry-stroke);
}
</style>
"""


def apply_foundry_theme():
    st.markdown(FOUNDRY_THEME_CSS, unsafe_allow_html=True)


def _h(value) -> str:
    return html.escape(str(value), quote=True)


FLUENT_ICON_SVGS = {
    "FoundryMark": """
<svg class="foundry-icon" height="20" viewBox="0 0 32 32" width="20" xmlns="http://www.w3.org/2000/svg" role="presentation" aria-hidden="true">
    <path clip-rule="evenodd" d="M20.4052 2C20.3713 2.04989 20.3403 2.10356 20.3119 2.15906C20.1753 2.42519 20.0629 2.80022 19.9685 3.2499C19.7794 4.15205 19.6545 5.3972 19.5714 6.7798C19.405 9.54716 19.405 12.8938 19.405 15.213V24.4338L19.4049 24.4698C19.3854 27.5153 16.8918 29.9806 13.8112 29.9999L13.7749 30H3.57642C3.18062 30 2.9073 29.6141 3.04346 29.2496C4.56004 25.1917 6.6982 19.4832 8.50404 14.6901C9.40697 12.2934 10.2268 10.1257 10.8442 8.50763C11.4636 6.88453 11.876 5.82419 11.9665 5.63239C12.2132 5.10978 12.6147 4.1951 13.1873 3.40856C13.7637 2.61659 14.4808 2.00001 15.3445 2H20.4052ZM29.2769 10.1842C29.4966 10.1842 29.6747 10.3603 29.6747 10.5775V17.6706L29.6745 17.7148C29.6504 19.5836 28.1106 21.0913 26.2147 21.0913H21.668C21.6778 21.0796 21.6872 21.0676 21.6966 21.0552C21.8605 20.8367 21.9531 20.526 21.9587 20.134L21.9589 20.0958V14.0817C21.9589 11.9291 23.7238 10.1842 25.9011 10.1842H29.2769ZM21.2532 2.14424C21.5631 2.14425 21.8986 2.38926 22.2468 2.88783C22.5881 3.37681 22.9111 4.06635 23.2065 4.85721C23.7783 6.3875 24.2354 8.26487 24.5265 9.71512C22.6354 10.2861 21.2595 12.0248 21.2595 14.0817V20.0782L21.2594 20.0921C21.2575 20.2355 21.2263 20.4039 21.1685 20.5329C21.1042 20.6758 21.0375 20.7121 20.9938 20.7121C20.9575 20.7121 20.8869 20.6826 20.7852 20.5652C20.6894 20.4549 20.5915 20.2961 20.4975 20.1117C20.3151 19.7539 20.1614 19.3273 20.0739 19.0482V15.213C20.0739 8.68733 20.3039 5.39271 20.5834 3.73209C20.7239 2.89797 20.8739 2.49601 20.9998 2.30459C21.0605 2.21243 21.1101 2.17748 21.1426 2.16241C21.1755 2.14714 21.207 2.14424 21.2532 2.14424Z" fill="currentColor" fill-rule="evenodd"></path>
</svg>
""",
    "BuildQueue": """
<svg class="foundry-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M5 6.5h14" />
    <path d="M5 12h14" />
    <path d="M5 17.5h8" />
    <path d="M17 15.5l2 2-2 2" />
</svg>
""",
    "TestBeaker": """
<svg class="foundry-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M9 4h6" />
    <path d="M10 4v5.4L5.8 17a2 2 0 0 0 1.8 3h8.8a2 2 0 0 0 1.8-3L14 9.4V4" />
    <path d="M8.4 15.5h7.2" />
</svg>
""",
    "LineChart": """
<svg class="foundry-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 19h16" />
    <path d="M5.5 15.5 9.5 11l3.5 3 5.5-7" />
    <path d="M18.5 7v4.5" />
    <path d="M14 7h4.5" />
</svg>
""",
}


def fluent_icon(icon_name: str) -> str:
    return FLUENT_ICON_SVGS.get(icon_name, FLUENT_ICON_SVGS["LineChart"])


def render_foundry_sidebar_brand(log_dir: str, train_file_count: int, eval_file_count: int):
    st.markdown(
        f"""
<div class="foundry-sidebar-brand">
    <div class="foundry-brand-row">
        <span class="foundry-product-mark">{fluent_icon("FoundryMark")}</span>
        <div>
            <div class="foundry-brand-name">Foundry Rollout Browser</div>
        </div>
    </div>
    <div class="foundry-source-card">
        <div class="foundry-source-label">Source folder</div>
        <div class="foundry-source" title="{_h(log_dir)}">{_h(log_dir)}</div>
    </div>
    <div class="foundry-file-stats">
        <form action="" class="foundry-refresh-form" method="get" target="_self">
            <button class="foundry-refresh-chip" name="refresh" title="Refresh folder" type="submit" value="1" aria-label="Refresh folder">↻</button>
        </form>
        <span>{fluent_icon("BuildQueue")} <strong>{train_file_count}</strong> train</span>
        <span>{fluent_icon("TestBeaker")} <strong>{eval_file_count}</strong> eval</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_foundry_app_header(title: str, description: str, details: list[tuple[str, str]]):
    detail_html = "".join(
        f'<span class="foundry-meta-pill"><strong>{_h(label)}</strong>{_h(value)}</span>'
        for label, value in details
    )
    st.markdown(
        f"""
<section class="foundry-app-header">
    <h1>{_h(title)}</h1>
    <p>{_h(description)}</p>
    <div class="foundry-header-meta">{detail_html}</div>
</section>
""",
        unsafe_allow_html=True,
    )


apply_foundry_theme()


def default_log_dir() -> str:
    configured = os.environ.get("ROLLOUT_LOG_DIR")
    if configured:
        return configured

    candidates = [
        Path("rollout_logs"),
        Path("../rollout_logs"),
        Path("../../rollout_logs"),
        Path("rollout_samples"),
        Path("../rollout_samples"),
        Path("../../rollout_samples"),
        Path("tau-bench-foundry-command-job/rollout_samples"),
    ]
    try:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir.parents[1] / "rollout_logs")
        candidates.append(script_dir.parents[1] / "rollout_samples")
    except (NameError, IndexError, OSError):
        pass

    for candidate in candidates:
        if candidate.expanduser().exists():
            return str(candidate)
    return "rollout_logs"


def get_log_dir() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=default_log_dir())
    args, _ = parser.parse_known_args()
    return args.log_dir


def normalize_rollout_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    fallback_index = pd.Series(range(len(normalized)), index=normalized.index)

    if "rollout_id" not in normalized:
        normalized["rollout_id"] = 0
    if "sample_idx" not in normalized:
        normalized["sample_idx"] = fallback_index
    else:
        normalized["sample_idx"] = normalized["sample_idx"].where(
            ~normalized["sample_idx"].isna(),
            fallback_index,
        )

    if "group_index" not in normalized:
        normalized["group_index"] = normalized["sample_idx"]
    else:
        normalized["group_index"] = normalized["group_index"].where(
            ~normalized["group_index"].isna(),
            normalized["sample_idx"],
        )

    if "prompt" not in normalized:
        normalized["prompt"] = ""
    if "response" not in normalized:
        normalized["response"] = ""
    if "label" not in normalized:
        normalized["label"] = ""
    if "truncated" not in normalized:
        normalized["truncated"] = False
    if "response_length" not in normalized:
        normalized["response_length"] = normalized["response"].fillna("").astype(str).str.len()

    return normalized


def parse_structured_value(value):
    if isinstance(value, (dict, list, tuple)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{(":
        return value

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def get_row_metadata(row) -> dict:
    value = row.get("metadata", {}) if hasattr(row, "get") else {}
    parsed = parse_structured_value(value)
    return parsed if isinstance(parsed, dict) else {}


def format_compact_value(value) -> str:
    parsed = parse_structured_value(value)
    if isinstance(parsed, dict):
        return ", ".join(f"{key}: {format_compact_value(val)}" for key, val in parsed.items())
    if isinstance(parsed, (list, tuple)):
        return ", ".join(format_compact_value(item) for item in parsed)
    return "" if parsed is None else str(parsed)


def metadata_context_text(row) -> str:
    metadata = get_row_metadata(row)
    if not metadata:
        return ""

    parts = []
    scenario_id = metadata.get("scenario_id")
    order_id = metadata.get("order_id")
    target_items = metadata.get("target_items")
    expected_resolution = metadata.get("expected_resolution")

    if scenario_id:
        parts.append(f"Scenario {scenario_id}")
    if order_id:
        parts.append(f"Order {order_id}")
    if target_items:
        parts.append(f"Items {format_compact_value(target_items)}")
    if expected_resolution:
        parts.append(str(expected_resolution))

    return " | ".join(parts)


def extract_row_prompt_text(row) -> str:
    prompt_text = extract_prompt_text(row.get("prompt", "") if hasattr(row, "get") else "")
    prompt_text = "" if prompt_text is None else str(prompt_text).strip()
    if prompt_text:
        return prompt_text

    context_text = metadata_context_text(row)
    if context_text:
        return context_text

    sample_idx = row.get("sample_idx", "?") if hasattr(row, "get") else "?"
    return f"Sample {sample_idx}"


def render_rollout_metadata(row):
    metadata = get_row_metadata(row)
    if not metadata:
        return

    fields = [
        ("Scenario", metadata.get("scenario_id")),
        ("Order", metadata.get("order_id")),
        ("Target items", metadata.get("target_items")),
        ("Expected resolution", metadata.get("expected_resolution")),
        ("Expected tools", metadata.get("expected_tools")),
        ("Expected actions", metadata.get("expected_actions")),
        ("Expected amounts", metadata.get("expected_amounts")),
        ("Difficulty", metadata.get("difficulty")),
    ]
    rows = [
        {"Field": label, "Value": format_compact_value(value)}
        for label, value in fields
        if value not in (None, "", [], {})
    ]
    if rows:
        st.markdown("**Scenario metadata**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


@st.cache_data
def load_jsonl_files(file_infos: tuple[tuple[str, int, int], ...]) -> pd.DataFrame:
    if not file_infos:
        return pd.DataFrame()
    rows = []
    skipped = 0
    for f, _, _ in file_infos:
        with open(f, "rb") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(_loads(line))
                    except _ParseError:
                        skipped += 1
    if skipped:
        st.warning(f"Skipped {skipped} malformed lines while loading rollout files.")
    return normalize_rollout_dataframe(pd.DataFrame(rows))


def discover_jsonl_files(directory: str, prefix: str) -> tuple[tuple[str, int, int], ...]:
    root = Path(directory).expanduser()
    if not root.exists():
        return ()

    files = []
    for path in root.rglob(f"{prefix}_*.jsonl"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(files))


_UPLOAD_STATE_KEYS = (
    "_upload_fingerprint",
    "_upload_train_df",
    "_upload_eval_df",
    "_upload_msg",
)


def get_upload_key() -> str:
    return f"rollout_upload_{st.session_state.get('_upload_widget_version', 0)}"


def clear_uploaded_file_state():
    for key in _UPLOAD_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_upload_widget_version"] = st.session_state.get("_upload_widget_version", 0) + 1


def get_uploaded_files_from_state(upload_key: str):
    return st.session_state.get(upload_key) or []


def apply_uploaded_files(uploaded_files):
    if not uploaded_files:
        return None

    fingerprint = hashlib.md5(
        "|".join(sorted(f"{uf.name}:{uf.size}" for uf in uploaded_files)).encode()
    ).hexdigest()

    if st.session_state.get("_upload_fingerprint") != fingerprint:
        def _parse_file_bytes(raw_bytes: bytes, fname: str):
            rows = []
            skipped = 0
            buf = io.BytesIO(raw_bytes)
            for raw_line in buf:
                line = raw_line.strip()
                if line:
                    try:
                        rows.append(_loads(line))
                    except _ParseError:
                        skipped += 1
            is_eval = "eval" in fname.lower()
            return rows, is_eval, skipped

        file_data = [(uf.getvalue(), uf.name) for uf in uploaded_files]
        upload_train_rows: list = []
        upload_eval_rows: list = []
        total_skipped = 0

        with ThreadPoolExecutor(max_workers=min(8, len(file_data))) as executor:
            futures = {
                executor.submit(_parse_file_bytes, raw, name): name
                for raw, name in file_data
            }
            for future in as_completed(futures):
                rows, is_eval, skipped = future.result()
                if is_eval:
                    upload_eval_rows.extend(rows)
                else:
                    upload_train_rows.extend(rows)
                total_skipped += skipped

        st.session_state["_upload_fingerprint"] = fingerprint
        st.session_state["_upload_train_df"] = (
            normalize_rollout_dataframe(pd.DataFrame(upload_train_rows)) if upload_train_rows else pd.DataFrame()
        )
        st.session_state["_upload_eval_df"] = (
            normalize_rollout_dataframe(pd.DataFrame(upload_eval_rows)) if upload_eval_rows else pd.DataFrame()
        )
        st.session_state["_upload_msg"] = (
            f"Loaded {len(upload_train_rows)} train + {len(upload_eval_rows)} eval rows."
            + (f" ({total_skipped} malformed lines skipped)" if total_skipped else "")
        )

    return st.session_state["_upload_train_df"], st.session_state["_upload_eval_df"]


def render_sidebar_bottom(upload_key: str, uploaded_files, sidebar=None):
    sidebar = st.sidebar if sidebar is None else sidebar
    sidebar.divider()
    with sidebar.expander("Upload JSONL files", expanded=False):
        st.file_uploader(
            "Upload rollout JSONL files",
            type=["jsonl", "json", "txt"],
            accept_multiple_files=True,
            key=upload_key,
            help="Upload train_rollout_*.jsonl or eval_rollout_*.jsonl files from your local machine.",
        )
        if uploaded_files:
            st.success(st.session_state.get("_upload_msg", "Uploaded files loaded."))
            if st.button("Use folder files", key="clear_uploaded_files"):
                clear_uploaded_file_state()
                st.rerun()

    with sidebar.expander(f"Environment variables ({len(os.environ)})", expanded=False):
        _SENSITIVE_HINTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "SAS")

        def _mask(name: str, value: str) -> str:
            upper = name.upper()
            if any(h in upper for h in _SENSITIVE_HINTS):
                return "***" if value else ""
            return value

        _env_filter = st.text_input(
            "Filter (substring, case-insensitive)",
            value="",
            key="env_var_filter",
            help="Type to filter by name or value. Common secret-looking vars are masked.",
        )
        _show_masked = st.checkbox(
            "Reveal masked values",
            value=False,
            key="env_var_reveal",
            help="Unmask values for variables whose name contains TOKEN/SECRET/PASSWORD/KEY/CREDENTIAL/SAS.",
        )
        _needle = _env_filter.strip().lower()
        _rows = []
        for _name in sorted(os.environ):
            _raw = os.environ[_name]
            _display = _raw if _show_masked else _mask(_name, _raw)
            if _needle and _needle not in _name.lower() and _needle not in _raw.lower():
                continue
            _rows.append({"name": _name, "value": _display})
        if _rows:
            st.dataframe(
                pd.DataFrame(_rows),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Showing {len(_rows)} of {len(os.environ)} variables.")
        else:
            st.caption("No environment variables match the filter.")


def stop_after_sidebar_bottom(upload_key: str, uploaded_files):
    render_sidebar_bottom(upload_key, uploaded_files)
    st.stop()


CHATML_TURN_RE = re.compile(
    r"<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|(?=<\|im_start\|>)|$)",
    flags=re.DOTALL,
)


def parse_prompt_messages(prompt_str: str) -> list[dict[str, str]]:
    if not isinstance(prompt_str, str):
        prompt_str = "" if prompt_str is None else str(prompt_str)

    try:
        msgs = json.loads(prompt_str)
        if isinstance(msgs, list):
            return [
                {"role": str(m.get("role", "?")), "content": str(m.get("content", ""))}
                for m in msgs
                if isinstance(m, dict)
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    chatml_messages = []
    for match in CHATML_TURN_RE.finditer(prompt_str):
        role = match.group(1).strip()
        content = match.group(2).strip()
        if role:
            chatml_messages.append({"role": role, "content": content})
    return chatml_messages


def extract_prompt_text(prompt_str: str) -> str:
    """Extract the user question from a prompt (JSON chat, ChatML, or plain text)."""
    msgs = parse_prompt_messages(prompt_str)
    if msgs:
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m.get("content", "")
        return msgs[-1].get("content", "")
    return prompt_str


def render_prompt_full(prompt_str: str):
    """Render the full prompt including system/user roles."""
    msgs = parse_prompt_messages(prompt_str)
    if msgs:
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            if role == "assistant" and not content:
                continue
            st.markdown(f"**{role.title()}**")
            st.markdown(content)
        return
    st.markdown(prompt_str)


def extract_reward_score(reward):
    """Return a numeric reward value, handling both plain numbers and dict rewards."""
    if isinstance(reward, (int, float)):
        return reward
    if isinstance(reward, dict) and "score" in reward:
        return reward["score"]
    return None


def is_missing_value(value) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_float(value, decimals: int = DISPLAY_FLOAT_DECIMALS) -> str:
    if is_missing_value(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(value)
    return f"{numeric:.{decimals}f}"


def format_reward_value(reward) -> str:
    score = extract_reward_score(reward)
    if score is not None:
        return format_float(score)
    return "—" if is_missing_value(reward) else str(reward)


def format_percentage(value, decimals: int = 1) -> str:
    if is_missing_value(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(value)
    return f"{numeric:.{decimals}%}"


def dynamic_numeric_domain(values, padding_ratio: float = 0.08, min_padding: float = 0.02) -> list[float] | None:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric_values.empty:
        return None

    min_value = float(numeric_values.min())
    max_value = float(numeric_values.max())
    if min_value == max_value:
        padding = max(abs(min_value) * padding_ratio, min_padding)
    else:
        padding = max((max_value - min_value) * padding_ratio, min_padding)
    return [min_value - padding, max_value + padding]


def reward_color(reward) -> str:
    score = extract_reward_score(reward)
    if score is None:
        return "gray"
    if score >= 1.0:
        return "green"
    if score > 0:
        return "orange"
    return "red"


def extract_reward_prediction(reward) -> str:
    """Return the model answer captured by the reward function, when present."""
    if isinstance(reward, dict):
        pred = reward.get("pred")
        return "" if pred is None else str(pred)
    return ""


def extract_code_block(response: str) -> str:
    if not isinstance(response, str):
        return ""
    match = re.search(r"<code>(.*?)</code>", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else ""


def compact_text(text: str, limit: int = 160) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_generation(text: str) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip().lower()


def pairwise_similarity(values: list[str], max_items: int = 20) -> float | None:
    values = [n for v in values if (n := normalize_generation(v))]
    if len(values) < 2:
        return None
    values = values[:max_items]
    scores = []
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            scores.append(difflib.SequenceMatcher(None, left, right).ratio())
    return sum(scores) / len(scores) if scores else None


def add_generation_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "response" not in enriched:
        enriched["response"] = ""
    if "response_length" not in enriched:
        enriched["response_length"] = None
    if "reward" in enriched:
        enriched["_reward"] = enriched["reward"].apply(extract_reward_score)
        enriched["_prediction"] = enriched["reward"].apply(extract_reward_prediction)
    else:
        enriched["_reward"] = None
        enriched["_prediction"] = ""
    enriched["_code"] = enriched["response"].apply(extract_code_block)
    enriched["_answer"] = enriched.apply(
        lambda row: row["_prediction"] or row["_code"] or row.get("response", ""),
        axis=1,
    )
    enriched["_response_preview"] = enriched["response"].apply(compact_text)
    enriched["_answer_preview"] = enriched["_answer"].apply(lambda value: compact_text(value, 220))
    return enriched


def coerce_sequence(value) -> list:
    parsed = parse_structured_value(value)
    if is_missing_value(parsed) or parsed == "":
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return [parsed]


def format_structured_text(value, limit: int | None = None) -> str:
    parsed = parse_structured_value(value)
    if isinstance(parsed, (dict, list, tuple)):
        text = json.dumps(parsed, indent=2, ensure_ascii=False)
    else:
        text = "" if parsed is None else str(parsed)
    return compact_text(text, limit) if limit else text


def tool_name_from_value(value) -> str:
    parsed = parse_structured_value(value)
    if isinstance(parsed, dict):
        function = parsed.get("function") if isinstance(parsed.get("function"), dict) else {}
        name = parsed.get("name") or function.get("name") or parsed.get("tool_name")
        return "" if name is None else str(name)
    return "" if parsed is None else str(parsed).strip().strip("'\"")


def tool_arguments_from_value(value):
    parsed = parse_structured_value(value)
    if not isinstance(parsed, dict):
        return None
    if "arguments" in parsed:
        return parsed.get("arguments")
    function = parsed.get("function")
    if isinstance(function, dict):
        return function.get("arguments")
    return None


def normalize_tool_call(value) -> dict | None:
    parsed = parse_structured_value(value)
    if not isinstance(parsed, dict):
        name = tool_name_from_value(parsed)
        return {"name": name, "arguments": None, "result": None, "id": ""} if name else None

    name = tool_name_from_value(parsed)
    if not name:
        return None
    return {
        "name": name,
        "arguments": tool_arguments_from_value(parsed),
        "result": parsed.get("result") or parsed.get("content"),
        "id": parsed.get("id") or parsed.get("tool_call_id") or "",
    }


def get_conversation_trace(row) -> list[dict]:
    value = row.get("conversation_trace") if hasattr(row, "get") else None
    if is_missing_value(value) or value == "":
        metadata = get_row_metadata(row)
        value = metadata.get("conversation_trace") or metadata.get("input_messages")

    parsed = parse_structured_value(value)
    if isinstance(parsed, tuple):
        parsed = list(parsed)
    if isinstance(parsed, list):
        messages = []
        for item in parsed:
            item = parse_structured_value(item)
            if isinstance(item, dict):
                messages.append(item)
            else:
                messages.append({"role": "message", "content": "" if item is None else str(item)})
        return messages
    if isinstance(parsed, str):
        return parse_prompt_messages(parsed)
    return []


def get_expected_tool_names(row) -> list[str]:
    metadata = get_row_metadata(row)
    candidates = [metadata.get("expected_tools")]
    if hasattr(row, "get"):
        candidates.append(row.get("expected_tools"))

    for value in candidates:
        tools = coerce_sequence(value)
        if not tools and isinstance(value, str) and "," in value:
            tools = [part.strip() for part in value.split(",")]
        names = [tool_name_from_value(tool) for tool in tools]
        names = [name for name in names if name]
        if names:
            return names
    return []


def get_output_tool_calls(row, trace: list[dict] | None = None) -> list[dict]:
    value = row.get("output_tools") if hasattr(row, "get") else None
    calls = [call for tool in coerce_sequence(value) if (call := normalize_tool_call(tool))]
    if calls:
        return calls

    trace = trace if trace is not None else get_conversation_trace(row)
    trace_calls = []
    for message in trace:
        for tool_call in coerce_sequence(message.get("tool_calls") if isinstance(message, dict) else None):
            call = normalize_tool_call(tool_call)
            if call:
                trace_calls.append(call)
    return trace_calls


def build_tool_comparison_rows(row, trace: list[dict] | None = None) -> list[dict]:
    expected = get_expected_tool_names(row)
    output_calls = get_output_tool_calls(row, trace)
    unmatched_output_indices = list(range(len(output_calls)))
    rows = []
    for expected_name in expected:
        match_index = next(
            (
                output_index
                for output_index in unmatched_output_indices
                if output_calls[output_index].get("name", "") == expected_name
            ),
            None,
        )
        if match_index is None:
            output_call = {}
            output_name = ""
            status = "Missing output"
        else:
            output_call = output_calls[match_index]
            output_name = output_call.get("name", "")
            unmatched_output_indices.remove(match_index)
            status = "OK"
        rows.append({
            "Step": len(rows) + 1,
            "Expected tool": expected_name or "-",
            "Output tool": output_name or "-",
            "Status": status,
            "Output arguments": format_structured_text(output_call.get("arguments"), limit=180) if output_call else "",
        })

    for output_index in unmatched_output_indices:
        output_call = output_calls[output_index]
        output_name = output_call.get("name", "")
        rows.append({
            "Step": len(rows) + 1,
            "Expected tool": "-",
            "Output tool": output_name or "-",
            "Status": "Unexpected output",
            "Output arguments": format_structured_text(output_call.get("arguments"), limit=180) if output_call else "",
        })
    return rows


def render_tool_comparison_table(table: pd.DataFrame):
    status_class = {
        "OK": "foundry-tool-status-ok",
        "Different": "foundry-tool-status-different",
        "Missing output": "foundry-tool-status-missing-output",
        "Unexpected output": "foundry-tool-status-unexpected-output",
    }
    row_html = []
    for row in table.to_dict("records"):
        status = str(row.get("Status", ""))
        row_html.append(
            "<tr>"
            f"<td class=\"foundry-tool-table-step\">{_h(row.get('Step', ''))}</td>"
            f"<td class=\"foundry-tool-name\">{_h(row.get('Expected tool', ''))}</td>"
            f"<td class=\"foundry-tool-name\">{_h(row.get('Output tool', ''))}</td>"
            f"<td class=\"foundry-tool-status {status_class.get(status, '')}\">{_h(status)}</td>"
            f"<td class=\"foundry-tool-args\" title=\"{_h(row.get('Output arguments', ''))}\">{_h(row.get('Output arguments', ''))}</td>"
            "</tr>"
        )

    st.markdown(
        "<div class=\"foundry-tool-table-wrap\">"
        "<table class=\"foundry-tool-table\">"
        "<colgroup>"
        "<col style=\"width: 4.8rem;\" />"
        "<col style=\"width: 14rem;\" />"
        "<col style=\"width: 14rem;\" />"
        "<col style=\"width: 8rem;\" />"
        "<col style=\"width: 23.2rem;\" />"
        "</colgroup>"
        "<thead><tr>"
        "<th>Step</th>"
        "<th>Expected tool</th>"
        "<th>Output tool</th>"
        "<th>Status</th>"
        "<th>Output arguments</th>"
        "</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_tool_comparison(row, trace: list[dict] | None = None):
    rows = build_tool_comparison_rows(row, trace)
    if not rows:
        st.caption("No expected or output tool calls were recorded for this sample.")
        return

    table = pd.DataFrame(rows)
    ok_count = int((table["Status"] == "OK").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected tools", len(get_expected_tool_names(row)))
    c2.metric("Output tools", len(get_output_tool_calls(row, trace)))
    c3.metric("Step matches", f"{ok_count}/{len(table)}")
    render_tool_comparison_table(table)


def render_trace_content(role: str, content):
    text = "" if content is None else str(content)
    parsed = parse_structured_value(content)
    if isinstance(parsed, (dict, list)):
        st.json(parsed, expanded=False)
        return
    if not text.strip():
        st.caption("(empty)")
        return

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    visible_text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if think_blocks:
        if visible_text:
            st.markdown("**Visible response**")
            st.text(visible_text)
        with st.expander("Reasoning", expanded=False):
            for block in think_blocks:
                st.text(block.strip())
        return

    if role in {"system", "user"}:
        st.markdown(text)
    else:
        st.text(text)


def render_trace_tool_calls(message: dict):
    tool_calls = [call for value in coerce_sequence(message.get("tool_calls")) if (call := normalize_tool_call(value))]
    if not tool_calls:
        return

    st.markdown("**Tool calls**")
    st.dataframe(
        pd.DataFrame([
            {
                "ID": call.get("id") or "-",
                "Tool": call.get("name") or "-",
                "Arguments": format_structured_text(call.get("arguments"), limit=220),
            }
            for call in tool_calls
        ]),
        hide_index=True,
        use_container_width=True,
        height=min(260, 40 + 35 * len(tool_calls)),
    )


def render_conversation_trace(
    row,
    key_prefix: str,
    show_system_control: bool = True,
    collapse_system: bool = False,
):
    trace = get_conversation_trace(row)
    if not trace:
        st.caption("No conversation trace was recorded for this sample.")
        return

    show_system = True
    if show_system_control:
        show_system = st.checkbox("Show system messages", value=False, key=f"{key_prefix}_show_system")

    hidden_count = 0
    for index, message in enumerate(trace, start=1):
        role = str(message.get("role", "message")).lower() if isinstance(message, dict) else "message"
        if role == "system" and not show_system:
            hidden_count += 1
            continue

        name = message.get("name") if isinstance(message, dict) else ""
        title = f"{index}. {role.title()}" + (f" / {name}" if name else "")
        if role == "system" and collapse_system:
            with st.expander(title, expanded=False):
                content = message.get("content") if isinstance(message, dict) else message
                render_trace_content(role, content)
                if isinstance(message, dict):
                    render_trace_tool_calls(message)
            continue

        with st.container(border=True):
            st.markdown(f"**{title}**")
            content = message.get("content") if isinstance(message, dict) else message
            render_trace_content(role, content)
            if isinstance(message, dict):
                render_trace_tool_calls(message)

    if hidden_count:
        st.caption(f"Hidden system messages: {hidden_count}")


def render_sample_diagnostics(row, key_prefix: str, show_system_control: bool = True):
    trace = get_conversation_trace(row)
    render_tool_comparison(row, trace)


def sample_id_label(row) -> str:
    sample_id = row.get("sample_idx", "?") if hasattr(row, "get") else "?"
    return f"Sample {sample_id}"


def sample_key_fragment(value) -> str:
    return re.sub(r"\W+", "_", str(value)).strip("_") or "sample"


def render_conversation_trace_browser(samples: pd.DataFrame, key_prefix: str):
    if samples.empty:
        st.info("No samples are available for this prompt.")
        return

    samples = samples.reset_index(drop=True)
    option_indices = list(range(len(samples)))
    picker_col, chat_col = st.columns([0.25, 0.75])

    with picker_col:
        st.markdown("**Sample IDs**")
        selected_index = st.radio(
            "Sample IDs",
            option_indices,
            format_func=lambda index: sample_id_label(samples.iloc[index]),
            label_visibility="collapsed",
            key=f"{key_prefix}_sample_id",
        )

    row = samples.iloc[selected_index]
    sample_id = row.get("sample_idx", selected_index) if hasattr(row, "get") else selected_index
    reward = row.get("reward") if hasattr(row, "get") else None
    with chat_col:
        st.markdown(f"**{sample_id_label(row)}** · Reward {format_reward_value(reward)}")
        st.markdown("**Tool call comparison**")
        trace = get_conversation_trace(row)
        render_tool_comparison(row, trace)
        st.divider()
        render_conversation_trace(
            row,
            f"{key_prefix}_{sample_key_fragment(sample_id)}",
            show_system_control=False,
            collapse_system=True,
        )


def render_rollout_picker(df: pd.DataFrame, key_prefix: str, label: str, sidebar=None):
    """Returns (selected_rollout, rollout_changed).

    rollout_changed is True on the first render and whenever the user picks a
    different rollout.  Callers should reset all dependent widget keys when it
    is True so that stale selections from the previous rollout are not shown.
    """
    sidebar = st.sidebar if sidebar is None else sidebar
    rollout_ids = sorted(df["rollout_id"].dropna().unique())
    if not rollout_ids:
        return None, False

    sidebar.markdown(
        f'<div class="foundry-sidebar-section-title">{_h(label)} rollout</div>',
        unsafe_allow_html=True,
    )
    default_index = len(rollout_ids) - 1
    sidebar.markdown('<div class="foundry-sidebar-control-label">Rollout</div>', unsafe_allow_html=True)
    prev_col, rollout_col, next_col = sidebar.columns([0.18, 0.64, 0.18], gap="small", vertical_alignment="bottom")

    current_rollout = st.session_state.get(f"{key_prefix}_rid", rollout_ids[default_index])
    current_index = rollout_ids.index(current_rollout) if current_rollout in rollout_ids else default_index
    with prev_col:
        st.button(
            "◀",
            key=f"{key_prefix}_prev",
            help="Previous rollout",
            disabled=current_index == 0,
            on_click=lambda: st.session_state.update({f"{key_prefix}_rid": rollout_ids[current_index - 1]}),
        )
    with rollout_col:
        selected_rollout = st.selectbox(
            "Rollout",
            rollout_ids,
            index=default_index,
            key=f"{key_prefix}_rid",
            label_visibility="collapsed",
        )
    with next_col:
        st.button(
            "▶",
            key=f"{key_prefix}_next",
            help="Next rollout",
            disabled=current_index == len(rollout_ids) - 1,
            on_click=lambda: st.session_state.update({f"{key_prefix}_rid": rollout_ids[current_index + 1]}),
        )

    # Detect whether the rollout just changed so callers can reset downstream state.
    _last_key = f"{key_prefix}_last_rollout"
    rollout_changed = st.session_state.get(_last_key) != selected_rollout
    st.session_state[_last_key] = selected_rollout

    rollout_summary = df.groupby("rollout_id").size().rename("samples").reset_index()
    if "reward" in df:
        reward_df = df.copy()
        reward_df["_reward"] = reward_df["reward"].apply(extract_reward_score)
        reward_summary = reward_df.groupby("rollout_id")["_reward"].mean().rename("avg_reward").reset_index()
        rollout_summary = rollout_summary.merge(reward_summary, on="rollout_id", how="left")
    current = rollout_summary[rollout_summary["rollout_id"] == selected_rollout].iloc[0]
    reward_value = current.get("avg_reward")
    reward_text = format_float(reward_value)
    sidebar.markdown(
        f"""
<div class="foundry-rollout-summary">
    <div><span>Samples</span><strong>{int(current['samples'])}</strong></div>
    <div><span>Avg reward</span><strong>{_h(reward_text)}</strong></div>
</div>
""",
        unsafe_allow_html=True,
    )
    return selected_rollout, rollout_changed


# Keys owned by the Train view that must be cleared when the rollout changes.
_TRAIN_DEPENDENT_KEYS = [
    "prompt_table",
    "train_generation_score",
    "train_generation_sort",
    "train_generation_search",
    "train_generation_cols",
    "train_compare_left",
    "train_compare_right",
]

# Keys owned by the Eval view that must be cleared when the rollout changes.
_EVAL_DEPENDENT_KEYS = [
    "e_ds",
    "e_detail",
    "eval_prompt_table",
    "eval_generation_score",
    "eval_generation_sort",
    "eval_generation_search",
    "eval_generation_cols",
    "eval_compare_left",
    "eval_compare_right",
]


def render_variation_summary(samples: pd.DataFrame):
    samples = add_generation_columns(samples)
    responses = samples["response"].fillna("").astype(str).tolist()
    unique_responses = samples["response"].fillna("").map(normalize_generation).nunique()
    unique_answers = samples["_answer"].fillna("").map(normalize_generation).nunique()
    similarity = pairwise_similarity(responses)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generations", len(samples))
    c2.metric("Unique responses", f"{unique_responses}/{len(samples)}")
    c3.metric("Unique answers", f"{unique_answers}/{len(samples)}")
    c4.metric("Avg similarity", format_percentage(similarity) if similarity is not None else "—")

    chart_col, answer_col = st.columns([1, 1])
    with chart_col:
        if samples["_reward"].notna().any():
            reward_points = samples["_reward"].reset_index(drop=True).rename("reward").reset_index()
            reward_points["index"] = reward_points["index"] + 1
            reward_points = reward_points.dropna(subset=["reward"])
            reward_domain = dynamic_numeric_domain(reward_points["reward"])

            reward_chart = {
                "height": 180,
                "mark": {
                    "type": "line",
                    "color": "#242424",
                    "point": {"filled": True, "size": 58, "color": "#242424"},
                    "strokeWidth": 2.25,
                },
                "encoding": {
                    "x": {
                        "field": "index",
                        "type": "quantitative",
                        "title": "Generation index",
                        "axis": {"format": "d", "tickMinStep": 1},
                    },
                    "y": {
                        "field": "reward",
                        "type": "quantitative",
                        "title": "Reward",
                        "scale": {"domain": reward_domain, "nice": False, "zero": False},
                        "axis": {"format": ".3f"},
                    },
                    "tooltip": [
                        {"field": "index", "type": "quantitative", "title": "Index", "format": "d"},
                        {"field": "reward", "type": "quantitative", "title": "Reward", "format": ".3f"},
                    ],
                },
            }
            st.caption("Reward by generation index")
            st.vega_lite_chart(reward_points, reward_chart, use_container_width=True)
        elif "response_length" in samples:
            st.caption("Response length distribution")
            st.bar_chart(samples[["response_length"]], height=180)

    with answer_col:
        answer_counts = (
            samples["_answer_preview"]
            .replace("", "(empty)")
            .value_counts()
            .head(8)
            .rename_axis("answer")
            .reset_index(name="count")
        )
        st.caption("Most common extracted answers")
        st.dataframe(answer_counts, hide_index=True, use_container_width=True, height=220)

    return samples


def render_generation_browser(samples: pd.DataFrame, key_prefix: str):
    samples = add_generation_columns(samples)

    controls = st.columns([1.1, 1.1, 1.4, 1])
    score_values = sorted(
        [score for score in samples["_reward"].dropna().unique()],
        key=lambda value: str(value),
    )
    score_options = ["All"] + [str(score) for score in score_values]
    selected_score = controls[0].selectbox(
        "Score filter",
        score_options,
        key=f"{key_prefix}_score",
        format_func=lambda score: score if score == "All" else format_float(score),
    )
    sort_by = controls[1].selectbox(
        "Sort",
        ["Original order", "Reward high → low", "Reward low → high", "Length high → low", "Length low → high"],
        key=f"{key_prefix}_sort",
    )
    search = controls[2].text_input("Search responses", key=f"{key_prefix}_search", placeholder="Find text/code/prediction")
    columns_per_row = controls[3].selectbox("Card columns", [1, 2, 3], index=1, key=f"{key_prefix}_cols")

    filtered = samples.copy()
    if selected_score != "All":
        filtered = filtered[filtered["_reward"].astype(str) == selected_score]
    if search:
        needle = search.lower()
        filtered = filtered[
            filtered["response"].fillna("").str.lower().str.contains(needle, regex=False)
            | filtered["_answer"].fillna("").str.lower().str.contains(needle, regex=False)
        ]

    if sort_by == "Reward high → low":
        filtered = filtered.sort_values("_reward", ascending=False, na_position="last")
    elif sort_by == "Reward low → high":
        filtered = filtered.sort_values("_reward", ascending=True, na_position="last")
    elif sort_by == "Length high → low" and "response_length" in filtered:
        filtered = filtered.sort_values("response_length", ascending=False, na_position="last")
    elif sort_by == "Length low → high" and "response_length" in filtered:
        filtered = filtered.sort_values("response_length", ascending=True, na_position="last")

    if filtered.empty:
        st.info("No generations match the current filters.")
        return

    summary_table = filtered[["_reward", "response_length", "_answer_preview", "_response_preview"]].copy()
    summary_table.index = range(1, len(summary_table) + 1)
    summary_table.index.name = "#"
    summary_table.columns = ["Score", "Tokens", "Extracted answer", "Response preview"]
    summary_table["Score"] = summary_table["Score"].apply(format_float)
    st.dataframe(summary_table, use_container_width=True, height=min(320, 40 + 35 * len(summary_table)))

    for start in range(0, len(filtered), columns_per_row):
        cols = st.columns(columns_per_row)
        for card_offset, (col, (_, row)) in enumerate(zip(cols, filtered.iloc[start:start + columns_per_row].iterrows())):
            score = extract_reward_score(row.get("reward"))
            color = reward_color(row.get("reward"))
            length = row.get("response_length", "—")
            with col.container(border=True):
                st.markdown(f":{color}[**Score: {format_float(score)}**] · `{length}` tokens")
                if row.get("_answer"):
                    with st.expander("Extracted answer / code", expanded=True):
                        st.code(row["_answer"] if row.get("_code") else str(row["_answer"]))
                with st.expander("Full generation", expanded=False):
                    st.text(row.get("response", ""))
                sample_key = re.sub(r"\W+", "_", str(row.get("sample_idx", f"{start}_{card_offset}")))
                with st.expander("Tool comparison", expanded=False):
                    render_sample_diagnostics(
                        row,
                        f"{key_prefix}_sample_{sample_key}_{start}_{card_offset}",
                        show_system_control=False,
                    )


def render_prompt_group_browser(
    rdf: pd.DataFrame,
    *,
    table_key: str,
    trace_key_prefix: str,
    generation_key_prefix: str,
    compare_left_key: str,
    compare_right_key: str,
    empty_message: str,
    default_selection_caption: str,
):
    if rdf.empty:
        st.info(empty_message)
        return

    if "_reward" not in rdf.columns:
        rdf = rdf.copy()
        rdf["_reward"] = rdf["reward"].apply(extract_reward_score) if "reward" in rdf else pd.NA

    prompts = []
    for gidx, group in rdf.groupby("group_index", sort=True, dropna=False):
        first = group.iloc[0]
        prompt_text = extract_row_prompt_text(first)
        n_correct = int((group["_reward"] == 1.0).sum()) if group["_reward"].notna().any() else 0
        n_total = len(group)
        prompts.append({
            "group_index": gidx,
            "prompt_text": prompt_text,
            "raw_prompt": first.get("prompt", ""),
            "metadata": first.get("metadata", {}),
            "label": first.get("label", ""),
            "n_correct": n_correct,
            "n_total": n_total,
            "avg_reward": group["_reward"].mean() if group["_reward"].notna().any() else None,
            "unique_answers": add_generation_columns(group)["_answer"].fillna("").map(normalize_generation).nunique(),
        })

    st.subheader("Prompts")
    prompt_df = pd.DataFrame(prompts)
    if prompt_df.empty:
        st.info(empty_message)
        return

    prompt_df["reward"] = prompt_df["avg_reward"].apply(format_float)
    prompt_df["prompt_preview"] = prompt_df["prompt_text"].astype(str).str[:120]

    selection = st.dataframe(
        prompt_df[["group_index", "prompt_preview", "reward"]].rename(columns={
            "group_index": "Group",
            "prompt_preview": "Prompt",
            "reward": "Reward",
        }),
        height=min(400, 40 + 35 * len(prompt_df)),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    selected_rows = selection.selection.rows if selection and selection.selection else []
    sel_idx = selected_rows[0] if selected_rows else 0
    if not selected_rows:
        st.caption(default_selection_caption)

    prompt_info = prompts[sel_idx]
    group_samples = rdf[rdf["group_index"] == prompt_info["group_index"]].reset_index(drop=True)

    st.divider()

    n_correct = prompt_info["n_correct"]
    n_total = prompt_info["n_total"]
    st.subheader(f"Prompt #{prompt_info['group_index']} - {n_correct}/{n_total} correct")

    with st.expander("Full prompt", expanded=False):
        if str(prompt_info["raw_prompt"]).strip():
            render_prompt_full(prompt_info["raw_prompt"])
        else:
            st.caption("This rollout row has an empty prompt field; using metadata to identify the scenario.")
        render_rollout_metadata(prompt_info)
        label = prompt_info.get("label")
        try:
            label_is_empty = pd.isna(label)
        except (TypeError, ValueError):
            label_is_empty = False
        if not label_is_empty and str(label).strip():
            st.info(f"**Expected answer:** {label}")

    trace_tab, overview_tab, generations_tab, compare_tab = st.tabs([
        "Conversation trace",
        "Variation overview",
        "Generation browser",
        "Side-by-side compare",
    ])

    with trace_tab:
        trace_key = sample_key_fragment(prompt_info["group_index"])
        render_conversation_trace_browser(group_samples, f"{trace_key_prefix}_{trace_key}")

    with overview_tab:
        render_variation_summary(group_samples)
        st.caption("Use this view to quickly spot whether generations are converging or producing many distinct answers.")

    with generations_tab:
        render_generation_browser(group_samples, generation_key_prefix)

    with compare_tab:
        enriched_samples = add_generation_columns(group_samples)
        n_gens = len(enriched_samples)

        if n_gens < 2:
            st.info("Only one generation for this prompt - nothing to compare.")
        else:
            compare_options = list(range(n_gens))

            def _gen_label(i: int) -> str:
                row = enriched_samples.iloc[i]
                return f"[{i}] score {format_float(row['_reward'])} - {row.get('response_length', '—')} tokens"

            picker_col_l, picker_col_r = st.columns(2)
            left_idx = picker_col_l.selectbox(
                "Left generation",
                compare_options,
                format_func=_gen_label,
                key=compare_left_key,
            )
            right_idx = picker_col_r.selectbox(
                "Right generation",
                compare_options,
                index=min(1, n_gens - 1),
                format_func=_gen_label,
                key=compare_right_key,
            )

            left_row = enriched_samples.iloc[left_idx]
            right_row = enriched_samples.iloc[right_idx]

            left_card, right_card = st.columns(2)

            def _render_compare_card(row: pd.Series, side: str):
                score = extract_reward_score(row.get("reward"))
                color = reward_color(row.get("reward"))
                length = row.get("response_length", "—")
                st.markdown(f":{color}[**Score: {format_float(score)}**] · `{length}` tokens")
                st.markdown(f"**{side} extracted answer**")
                st.code(row["_answer"] or "(empty)")
                st.markdown("**Full generation**")
                st.text(row.get("response", ""))

            with left_card.container(border=True):
                _render_compare_card(left_row, "Left")
            with right_card.container(border=True):
                _render_compare_card(right_row, "Right")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if "refresh" in st.query_params:
    st.cache_data.clear()
    clear_uploaded_file_state()
    st.query_params.clear()
    st.rerun()

log_dir = get_log_dir()
train_file_infos = discover_jsonl_files(log_dir, "train_rollout")
eval_file_infos = discover_jsonl_files(log_dir, "eval_rollout")
train_df = load_jsonl_files(train_file_infos)
eval_df = load_jsonl_files(eval_file_infos)
has_train = not train_df.empty
has_eval = not eval_df.empty

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

sidebar_header = st.sidebar.container()
sidebar_controls = st.sidebar.container()
sidebar_bottom = st.sidebar.container()

with sidebar_header:
    render_foundry_sidebar_brand(log_dir, len(train_file_infos), len(eval_file_infos))

upload_key = get_upload_key()
uploaded_files = get_uploaded_files_from_state(upload_key)
uploaded_dfs = apply_uploaded_files(uploaded_files)
if uploaded_dfs is not None:
    train_df, eval_df = uploaded_dfs
    has_train = not train_df.empty
    has_eval = not eval_df.empty

with sidebar_controls:
    view_options = []
    if has_train:
        view_options.append("Train")
    if has_eval:
        view_options.append("Eval")

    if view_options:
        st.divider()
        current_view = st.session_state.get("view", view_options[0])
        view_index = view_options.index(current_view) if current_view in view_options else 0
        view = st.radio("View", view_options, index=view_index, key="view")
    else:
        view = None

if not has_train and not has_eval:
    if uploaded_files:
        st.warning("Uploaded files could not be parsed. Ensure filenames contain 'train' or 'eval' and contain valid JSONL.")
    else:
        render_foundry_app_header(
            "Foundry Rollout Browser",
            "Upload train or eval rollout JSONL files to inspect generations, rewards, and variation.",
            [("Source", log_dir), ("Train files", str(len(train_file_infos))), ("Eval files", str(len(eval_file_infos)))],
        )
        st.info("Upload rollout JSONL files using the sidebar to get started. "
                f"(No `train_rollout_*.jsonl` or `eval_rollout_*.jsonl` files found under `{log_dir}`)")
    render_sidebar_bottom(upload_key, uploaded_files, sidebar_bottom)
    st.stop()

render_sidebar_bottom(upload_key, uploaded_files, sidebar_bottom)


# ===================================================================
#  TRAIN VIEW
# ===================================================================

if view == "Train":
    if not has_train:
        st.warning("No train rollout files found.")
        st.stop()

    # -- Step 1: Pick a rollout --
    selected_rollout, rollout_changed = render_rollout_picker(train_df, "t", "Train", sidebar_controls)
    if selected_rollout is None:
        st.warning("No train rollouts found.")
        st.stop()

    if rollout_changed:
        for _key in _TRAIN_DEPENDENT_KEYS:
            st.session_state.pop(_key, None)

    rdf = train_df[train_df["rollout_id"] == selected_rollout].copy()
    rdf["_reward"] = rdf["reward"].apply(extract_reward_score)

    # -- Header metrics for this rollout --
    render_foundry_app_header(
        f"Train rollout #{selected_rollout}",
        "Review prompt groups, reward movement, generation diversity, and side-by-side model outputs in the current rollout.",
        [("Source", log_dir), ("Prompts", str(rdf["group_index"].nunique())), ("Samples", str(len(rdf)))],
    )

    # -- Training-run context across all rollouts --
    with st.expander("Training reward trend across all rollouts", expanded=False):
        all_rewards = train_df.copy()
        all_rewards["_reward"] = all_rewards["reward"].apply(extract_reward_score)
        trend = all_rewards.groupby("rollout_id")["_reward"].mean().reset_index()
        trend.columns = ["rollout_id", "mean_reward"]
        if trend["mean_reward"].notna().any():
            reward_trend_chart = {
                "height": 220,
                "mark": {
                    "type": "line",
                    "color": "#242424",
                    "point": {"filled": True, "size": 54, "color": "#242424"},
                    "strokeWidth": 2.25,
                },
                "encoding": {
                    "x": {
                        "field": "rollout_id",
                        "type": "quantitative",
                        "title": "Rollout",
                        "axis": {"format": "d", "tickMinStep": 1},
                    },
                    "y": {
                        "field": "mean_reward",
                        "type": "quantitative",
                        "title": "Mean reward",
                        "scale": {"domain": dynamic_numeric_domain(trend["mean_reward"]), "nice": False, "zero": False},
                        "axis": {"format": ".3f"},
                    },
                    "tooltip": [
                        {"field": "rollout_id", "type": "quantitative", "title": "Rollout", "format": "d"},
                        {"field": "mean_reward", "type": "quantitative", "title": "Mean reward", "format": ".3f"},
                    ],
                },
            }
            st.vega_lite_chart(trend, reward_trend_chart, use_container_width=True)
        else:
            st.caption("No reward values are available for this trend.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts", rdf["group_index"].nunique())
    c2.metric("Samples/Prompt", int(len(rdf) / max(rdf["group_index"].nunique(), 1)))
    c3.metric("Avg Reward", format_float(rdf["_reward"].mean()) if rdf["_reward"].notna().any() else "—")
    c4.metric("Avg Length", f"{rdf['response_length'].mean():.0f}" if "response_length" in rdf else "—")

    st.divider()

    # -- Step 2: Build prompt list --
    prompts = []
    for gidx, group in rdf.groupby("group_index", sort=True, dropna=False):
        first = group.iloc[0]
        prompt_text = extract_row_prompt_text(first)
        n_correct = int((group["_reward"] == 1.0).sum()) if group["_reward"].notna().any() else 0
        n_total = len(group)
        prompts.append({
            "group_index": gidx,
            "prompt_text": prompt_text,
            "raw_prompt": first.get("prompt", ""),
            "metadata": first.get("metadata", {}),
            "label": first.get("label", ""),
            "n_correct": n_correct,
            "n_total": n_total,
            "avg_reward": group["_reward"].mean() if group["_reward"].notna().any() else None,
            "unique_answers": add_generation_columns(group)["_answer"].fillna("").map(normalize_generation).nunique(),
        })

    # -- Prompt table (clickable) --
    st.subheader("Prompts")
    prompt_df = pd.DataFrame(prompts)
    if prompt_df.empty:
        st.info("No prompt groups were found for this rollout.")
        st.stop()

    prompt_df["reward"] = prompt_df["avg_reward"].apply(format_float)
    prompt_df["prompt_preview"] = prompt_df["prompt_text"].str[:120]

    selection = st.dataframe(
        prompt_df[["group_index", "prompt_preview", "reward"]].rename(columns={
            "group_index": "Group",
            "prompt_preview": "Prompt",
            "reward": "Reward",
        }),
        height=min(400, 40 + 35 * len(prompt_df)),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="prompt_table",
    )

    # -- Prompt detail (shown when a row is clicked) --
    selected_rows = selection.selection.rows if selection and selection.selection else []
    sel_idx = selected_rows[0] if selected_rows else 0
    if not selected_rows:
        st.caption("Showing the first prompt by default. Select another row to inspect a different prompt.")

    prompt_info = prompts[sel_idx]
    group_samples = rdf[rdf["group_index"] == prompt_info["group_index"]].reset_index(drop=True)

    st.divider()

    # -- Summary --
    n_correct = prompt_info["n_correct"]
    n_total = prompt_info["n_total"]
    st.subheader(f"Prompt #{prompt_info['group_index']}  —  {n_correct}/{n_total} correct")

    # -- Full prompt --
    with st.expander("Full prompt", expanded=False):
        if str(prompt_info["raw_prompt"]).strip():
            render_prompt_full(prompt_info["raw_prompt"])
        else:
            st.caption("This rollout row has an empty prompt field; using metadata to identify the scenario.")
        render_rollout_metadata(prompt_info)
        if prompt_info["label"]:
            st.info(f"**Expected answer:** {prompt_info['label']}")

    trace_tab, overview_tab, generations_tab, compare_tab = st.tabs([
        "Conversation trace",
        "Variation overview",
        "Generation browser",
        "Side-by-side compare",
    ])

    with trace_tab:
        trace_key = sample_key_fragment(prompt_info["group_index"])
        render_conversation_trace_browser(group_samples, f"train_trace_{trace_key}")

    with overview_tab:
        enriched_samples = render_variation_summary(group_samples)
        st.caption("Use this view to quickly spot whether generations are converging or producing many distinct answers.")

    with generations_tab:
        render_generation_browser(group_samples, "train_generation")

    with compare_tab:
        enriched_samples = add_generation_columns(group_samples)
        n_gens = len(enriched_samples)

        if n_gens < 2:
            st.info("Only one generation for this prompt — nothing to compare.")
        else:
            compare_options = list(range(n_gens))

            def _gen_label(i: int) -> str:
                row = enriched_samples.iloc[i]
                return f"[{i}] score {format_float(row['_reward'])} · {row.get('response_length', '—')} tokens"

            picker_col_l, picker_col_r = st.columns(2)
            left_idx = picker_col_l.selectbox(
                "Left generation",
                compare_options,
                format_func=_gen_label,
                key="train_compare_left",
            )
            right_idx = picker_col_r.selectbox(
                "Right generation",
                compare_options,
                index=min(1, n_gens - 1),
                format_func=_gen_label,
                key="train_compare_right",
            )

            left_row = enriched_samples.iloc[left_idx]
            right_row = enriched_samples.iloc[right_idx]

            left_card, right_card = st.columns(2)

            def _render_compare_card(row: pd.Series, side: str):
                score = extract_reward_score(row.get("reward"))
                color = reward_color(row.get("reward"))
                length = row.get("response_length", "—")
                st.markdown(f":{color}[**Score: {format_float(score)}**] · `{length}` tokens")
                st.markdown(f"**{side} extracted answer**")
                st.code(row["_answer"] or "(empty)")
                st.markdown("**Full generation**")
                st.text(row.get("response", ""))

            with left_card.container(border=True):
                _render_compare_card(left_row, "Left")
            with right_card.container(border=True):
                _render_compare_card(right_row, "Right")


# ===================================================================
#  EVAL VIEW
# ===================================================================

elif view == "Eval":
    if not has_eval:
        st.warning("No eval rollout files found.")
        st.stop()

    # -- Step 1: Pick a rollout --
    selected_rollout, rollout_changed = render_rollout_picker(eval_df, "e", "Eval", sidebar_controls)
    if selected_rollout is None:
        st.warning("No eval rollouts found.")
        st.stop()

    if rollout_changed:
        for _key in _EVAL_DEPENDENT_KEYS:
            st.session_state.pop(_key, None)

    rollout_eval_df = eval_df[eval_df["rollout_id"] == selected_rollout].copy()
    if "dataset" in eval_df.columns:
        datasets = sorted(rollout_eval_df["dataset"].dropna().unique())
        if datasets:
            dataset_key = f"e_ds_{sample_key_fragment(selected_rollout)}"
            if st.session_state.get(dataset_key) not in datasets:
                st.session_state.pop(dataset_key, None)
            selected_dataset = sidebar_controls.selectbox("Dataset", datasets, key=dataset_key)
            edf = rollout_eval_df[rollout_eval_df["dataset"] == selected_dataset].copy()
        else:
            selected_dataset = None
            edf = rollout_eval_df
    else:
        selected_dataset = None
        edf = rollout_eval_df

    edf["_reward"] = edf["reward"].apply(extract_reward_score)

    # -- Header --
    title = f"Eval Rollout #{selected_rollout}"
    if selected_dataset:
        title += f" — {selected_dataset}"
    render_foundry_app_header(
        title,
        "Inspect evaluation samples, accuracy movement, truncation, and generation variation for this rollout.",
        [
            ("Source", log_dir),
            ("Dataset", selected_dataset or "All"),
            ("Prompts", str(edf["group_index"].nunique())),
            ("Samples", str(len(edf))),
        ],
    )

    prompt_count = edf["group_index"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts", prompt_count)
    c2.metric("Samples/Prompt", int(len(edf) / max(prompt_count, 1)))
    accuracy = edf["_reward"].mean() if edf["_reward"].notna().any() else None
    c3.metric("Accuracy", format_percentage(accuracy) if accuracy is not None else "—")
    trunc = edf["truncated"].sum() if "truncated" in edf else 0
    c4.metric("Truncated", int(trunc))

    # -- Accuracy trend --
    with st.expander("Accuracy trend across all eval rollouts", expanded=False):
        all_eval = eval_df.copy()
        all_eval["_reward"] = all_eval["reward"].apply(extract_reward_score)
        if selected_dataset and "dataset" in all_eval.columns:
            all_eval = all_eval[all_eval["dataset"] == selected_dataset]
        trend = all_eval.groupby("rollout_id")["_reward"].mean().reset_index()
        trend.columns = ["rollout_id", "accuracy"]
        if trend["accuracy"].notna().any():
            accuracy_trend_chart = {
                "height": 220,
                "mark": {
                    "type": "line",
                    "color": "#242424",
                    "point": {"filled": True, "size": 54, "color": "#242424"},
                    "strokeWidth": 2.25,
                },
                "encoding": {
                    "x": {
                        "field": "rollout_id",
                        "type": "quantitative",
                        "title": "Rollout",
                        "axis": {"format": "d", "tickMinStep": 1},
                    },
                    "y": {
                        "field": "accuracy",
                        "type": "quantitative",
                        "title": "Accuracy",
                        "scale": {"domain": dynamic_numeric_domain(trend["accuracy"]), "nice": False, "zero": False},
                        "axis": {"format": ".1%"},
                    },
                    "tooltip": [
                        {"field": "rollout_id", "type": "quantitative", "title": "Rollout", "format": "d"},
                        {"field": "accuracy", "type": "quantitative", "title": "Accuracy", "format": ".1%"},
                    ],
                },
            }
            st.vega_lite_chart(trend, accuracy_trend_chart, use_container_width=True)
        else:
            st.caption("No accuracy values are available for this trend.")

    st.divider()
    render_prompt_group_browser(
        edf,
        table_key="eval_prompt_table",
        trace_key_prefix="eval_trace",
        generation_key_prefix="eval_generation",
        compare_left_key="eval_compare_left",
        compare_right_key="eval_compare_right",
        empty_message="No eval prompt groups were found for this rollout.",
        default_selection_caption="Showing the first eval prompt by default. Select another row to inspect a different prompt.",
    )

