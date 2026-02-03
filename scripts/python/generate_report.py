#!/usr/bin/env python3
"""
Generate HTML report for MIDI-grep extraction results.
Creates a single-page report with audio players, comparison charts, and Strudel code.
Styled like Playwright/Jupyter reports.
"""

import argparse
import html
import json
import os
import base64
from pathlib import Path
from datetime import datetime

def encode_audio_base64(path):
    """Encode audio file as base64 data URI."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = path.split('.')[-1]
    mime = 'audio/wav' if ext == 'wav' else f'audio/{ext}'
    return f"data:{mime};base64,{data}"

def encode_image_base64(path):
    """Encode image file as base64 data URI."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = path.split('.')[-1].lower()
    mime = 'image/png' if ext == 'png' else f'image/{ext}'
    return f"data:{mime};base64,{data}"

def extract_info_from_strudel(code):
    """Extract BPM, key, notes from strudel code comments."""
    import re
    info = {}

    # Extract BPM
    bpm_match = re.search(r'BPM:\s*(\d+)', code)
    if bpm_match:
        info['bpm'] = int(bpm_match.group(1))

    # Extract key
    key_match = re.search(r'Key:\s*([A-G][#b]?\s*(?:major|minor)?)', code, re.IGNORECASE)
    if key_match:
        info['key'] = key_match.group(1)

    # Extract notes count
    notes_match = re.search(r'Notes:\s*(\d+)', code)
    if notes_match:
        info['notes'] = int(notes_match.group(1))

    # Extract drum hits
    drums_match = re.search(r'Drums:\s*(\d+)\s*hits', code)
    if drums_match:
        info['drum_hits'] = int(drums_match.group(1))

    # Extract style
    style_match = re.search(r'Style:\s*(\w+)', code)
    if style_match:
        info['style'] = style_match.group(1)

    # Extract genre
    genre_match = re.search(r'Genre:\s*([^\n]+)', code)
    if genre_match:
        info['genre'] = genre_match.group(1).strip()

    return info

def clean_track_name(folder_name):
    """Clean up folder name for display."""
    import re
    # If it's a yt_xxx format, make it more readable
    if folder_name.startswith('yt_'):
        return f"YouTube Track ({folder_name[3:]})"
    # If it has the [xxx] format, it's already using track name
    if '[' in folder_name and ']' in folder_name:
        return re.sub(r'\s*\[[^\]]+\]$', '', folder_name)
    return folder_name

def generate_charts_html(comparison_results):
    """Generate HTML charts from comparison JSON data - pure HTML/CSS, no images."""
    if not comparison_results:
        return '<div class="no-data" style="margin-top: 1rem;">No comparison data available (render audio first)</div>'

    comp = comparison_results.get('comparison', {})
    orig = comparison_results.get('original', {})
    rend = comparison_results.get('rendered', {})

    html_parts = []

    # Overall similarity score
    overall = comp.get('overall_similarity', 0) * 100
    overall_color = '#3fb950' if overall >= 70 else '#d29922' if overall >= 50 else '#f85149'
    html_parts.append(f'''
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; font-weight: bold; color: {overall_color};">{overall:.0f}%</div>
            <div style="color: var(--text-secondary);">Overall Similarity</div>
        </div>
    ''')

    # Similarity scores table
    metrics = [
        ('Timbre (MFCC)', 'mfcc_similarity'),
        ('Harmony (Chroma)', 'chroma_similarity'),
        ('Brightness', 'brightness_similarity'),
        ('Tempo', 'tempo_similarity'),
        ('Frequency Balance', 'frequency_balance_similarity'),
        ('Energy', 'energy_similarity'),
    ]

    rows = ''
    for label, key in metrics:
        val = comp.get(key, 0) * 100
        color = '#3fb950' if val >= 70 else '#d29922' if val >= 50 else '#f85149'
        bar_width = min(val, 100)
        rows += f'''
            <tr>
                <td style="padding: 0.5rem; color: var(--text-secondary);">{label}</td>
                <td style="padding: 0.5rem; width: 60%;">
                    <div style="background: var(--bg-primary); border-radius: 4px; height: 20px; overflow: hidden;">
                        <div style="width: {bar_width}%; height: 100%; background: {color};"></div>
                    </div>
                </td>
                <td style="padding: 0.5rem; text-align: right; font-weight: 600; color: {color};">{val:.0f}%</td>
            </tr>
        '''

    html_parts.append(f'''
        <div class="chart-item" style="padding: 1rem;">
            <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Similarity Scores</h4>
            <table style="width: 100%; border-collapse: collapse;">{rows}</table>
        </div>
    ''')

    # Frequency bands comparison
    bands = [
        ('Sub Bass', 'sub_bass'),
        ('Bass', 'bass'),
        ('Low Mid', 'low_mid'),
        ('Mid', 'mid'),
        ('High Mid', 'high_mid'),
        ('High', 'high'),
    ]

    orig_bands = orig.get('bands', {})
    rend_bands = rend.get('bands', {})

    band_rows = ''
    for label, key in bands:
        orig_val = orig_bands.get(key, 0) * 100
        rend_val = rend_bands.get(key, 0) * 100
        diff = rend_val - orig_val
        diff_str = f'+{diff:.1f}' if diff > 0 else f'{diff:.1f}'
        diff_color = '#3fb950' if abs(diff) < 5 else '#d29922' if abs(diff) < 15 else '#f85149'

        band_rows += f'''
            <tr>
                <td style="padding: 0.4rem; color: var(--text-secondary);">{label}</td>
                <td style="padding: 0.4rem; text-align: right;">{orig_val:.1f}%</td>
                <td style="padding: 0.4rem; text-align: right;">{rend_val:.1f}%</td>
                <td style="padding: 0.4rem; text-align: right; color: {diff_color};">{diff_str}%</td>
            </tr>
        '''

    html_parts.append(f'''
        <div class="chart-item" style="padding: 1rem;">
            <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Frequency Bands</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid var(--border);">
                    <th style="padding: 0.4rem; text-align: left; color: var(--text-secondary);">Band</th>
                    <th style="padding: 0.4rem; text-align: right; color: #58a6ff;">Original</th>
                    <th style="padding: 0.4rem; text-align: right; color: #3fb950;">Rendered</th>
                    <th style="padding: 0.4rem; text-align: right; color: var(--text-secondary);">Diff</th>
                </tr>
                {band_rows}
            </table>
        </div>
    ''')

    # Audio metrics comparison
    orig_spectral = orig.get('spectral', {})
    rend_spectral = rend.get('spectral', {})
    orig_rhythm = orig.get('rhythm', {})
    rend_rhythm = rend.get('rhythm', {})

    metrics_rows = f'''
        <tr>
            <td style="padding: 0.4rem; color: var(--text-secondary);">Tempo</td>
            <td style="padding: 0.4rem; text-align: right;">{orig_rhythm.get('tempo', 0):.1f} BPM</td>
            <td style="padding: 0.4rem; text-align: right;">{rend_rhythm.get('tempo', 0):.1f} BPM</td>
        </tr>
        <tr>
            <td style="padding: 0.4rem; color: var(--text-secondary);">Brightness (Centroid)</td>
            <td style="padding: 0.4rem; text-align: right;">{orig_spectral.get('centroid_mean', 0):.0f} Hz</td>
            <td style="padding: 0.4rem; text-align: right;">{rend_spectral.get('centroid_mean', 0):.0f} Hz</td>
        </tr>
        <tr>
            <td style="padding: 0.4rem; color: var(--text-secondary);">RMS Energy</td>
            <td style="padding: 0.4rem; text-align: right;">{orig_spectral.get('rms_mean', 0):.4f}</td>
            <td style="padding: 0.4rem; text-align: right;">{rend_spectral.get('rms_mean', 0):.4f}</td>
        </tr>
        <tr>
            <td style="padding: 0.4rem; color: var(--text-secondary);">Spectral Flatness</td>
            <td style="padding: 0.4rem; text-align: right;">{orig_spectral.get('flatness_mean', 0):.4f}</td>
            <td style="padding: 0.4rem; text-align: right;">{rend_spectral.get('flatness_mean', 0):.4f}</td>
        </tr>
    '''

    html_parts.append(f'''
        <div class="chart-item" style="padding: 1rem;">
            <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Audio Metrics</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid var(--border);">
                    <th style="padding: 0.4rem; text-align: left; color: var(--text-secondary);">Metric</th>
                    <th style="padding: 0.4rem; text-align: right; color: #58a6ff;">Original</th>
                    <th style="padding: 0.4rem; text-align: right; color: #3fb950;">Rendered</th>
                </tr>
                {metrics_rows}
            </table>
        </div>
    ''')

    # Insights
    insights = comparison_results.get('insights', [])
    if insights:
        insights_html = '<br>'.join(html.escape(i) for i in insights)
        html_parts.append(f'''
            <div class="chart-item full-width" style="padding: 1rem;">
                <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Analysis Insights</h4>
                <pre style="white-space: pre-wrap; font-family: inherit; margin: 0; color: var(--text-secondary); font-size: 0.85rem;">{insights_html}</pre>
            </div>
        ''')

    return f'<div class="charts-grid">{"".join(html_parts)}</div>'

def generate_ai_analysis_card(ai_params):
    """Generate AI Analysis card HTML if params available."""
    if not ai_params:
        return ''

    analysis = ai_params.get('analysis', {})
    suggestions = ai_params.get('suggestions', {})

    if not analysis and not suggestions:
        return ''

    # Get frequency band data
    bands = analysis.get('spectrum', {}).get('band_energy', {})
    dynamics = analysis.get('dynamics', {})
    rhythm = analysis.get('rhythm', {})

    # Format band percentages
    band_bars = ''
    for band_name, value in bands.items():
        pct = value * 100
        label = band_name.replace('_', ' ').title()
        band_bars += f'''
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="width: 80px; font-size: 0.75rem; color: var(--text-secondary);">{label}</span>
                <div style="flex: 1; height: 12px; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden;">
                    <div style="width: {min(pct * 2, 100):.0f}%; height: 100%; background: var(--accent);"></div>
                </div>
                <span style="width: 50px; text-align: right; font-size: 0.75rem; color: var(--text-secondary);">{pct:.1f}%</span>
            </div>'''

    # Get suggested sounds
    sound_chips = ''
    for voice, settings in suggestions.items():
        if voice in ['bass', 'mid', 'high'] and isinstance(settings, dict):
            sound = settings.get('sound', '')
            if sound:
                sound_chips += f'<span class="badge badge-blue" style="margin-right: 0.3rem;">{voice}: {sound}</span>'

    # Dynamics info
    dynamics_info = ''
    if dynamics:
        rms = dynamics.get('rms_mean', 0)
        dr = dynamics.get('dynamic_range_db', 0)
        dynamics_info = f'RMS: {rms:.3f} | Dynamic Range: {dr:.1f}dB'

    return f'''
        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M8.5 1.5a.5.5 0 0 0-1 0V7H2a.5.5 0 0 0 0 1h5.5v5.5a.5.5 0 0 0 1 0V8H14a.5.5 0 0 0 0-1H8.5V1.5z"/></svg>
                AI Analysis
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div>
                    <h4 style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">Frequency Distribution</h4>
                    {band_bars}
                </div>
                <div>
                    <h4 style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem;">Suggested Sounds</h4>
                    <div style="margin-bottom: 1rem;">{sound_chips if sound_chips else '<span style="color: var(--text-secondary);">N/A</span>'}</div>
                    <h4 style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Dynamics</h4>
                    <p style="font-size: 0.85rem; color: var(--text-primary);">{dynamics_info if dynamics_info else 'N/A'}</p>
                </div>
            </div>
        </div>
    '''

def generate_report(cache_dir, version_dir, output_path=None):
    """Generate HTML report for a version."""

    cache_path = Path(cache_dir)
    version_path = Path(version_dir) if version_dir else cache_path

    # Find files
    melodic_path = cache_path / "melodic.wav"
    if not melodic_path.exists():
        melodic_path = cache_path / "piano.wav"
    drums_path = cache_path / "drums.wav"
    vocals_path = cache_path / "vocals.wav"
    bass_path = cache_path / "bass.wav"

    render_path = version_path / "render.wav"
    if not render_path.exists():
        # Try legacy naming
        for f in version_path.glob("render*.wav"):
            render_path = f
            break

    strudel_path = version_path / "output.strudel"
    if not strudel_path.exists():
        strudel_path = version_path / "output_latest.strudel"

    # Individual chart files (new format)
    chart_frequency = version_path / "chart_frequency.png"
    chart_similarity = version_path / "chart_similarity.png"
    chart_spec_orig = version_path / "chart_spectrogram_original.png"
    chart_spec_rend = version_path / "chart_spectrogram_rendered.png"
    chart_chroma_orig = version_path / "chart_chromagram_original.png"
    chart_chroma_rend = version_path / "chart_chromagram_rendered.png"

    # Comparison results JSON (for HTML charts)
    comparison_json_path = version_path / "comparison.json"

    # Legacy combined chart (fallback)
    comparison_path = version_path / "comparison.png"
    if not comparison_path.exists():
        for f in version_path.glob("comparison*.png"):
            comparison_path = f
            break

    metadata_path = version_path / "metadata.json"
    track_meta_path = cache_path / "metadata.json"
    # AI params: check version dir first, then cache root (for old caches)
    ai_params_path = version_path / "ai_params.json"
    if not ai_params_path.exists():
        ai_params_path = cache_path / "ai_params.json"

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    track_info = {}
    if track_meta_path.exists():
        with open(track_meta_path) as f:
            track_info = json.load(f)

    # Load AI params
    ai_params = {}
    if ai_params_path.exists():
        with open(ai_params_path) as f:
            ai_params = json.load(f)

    # Read Strudel code
    strudel_code = ""
    if strudel_path.exists():
        with open(strudel_path) as f:
            strudel_code = f.read()

    # Extract info from strudel comments as fallback
    strudel_info = extract_info_from_strudel(strudel_code) if strudel_code else {}

    # Encode files as base64
    melodic_data = encode_audio_base64(str(melodic_path)) if melodic_path.exists() else None
    drums_data = encode_audio_base64(str(drums_path)) if drums_path.exists() else None
    vocals_data = encode_audio_base64(str(vocals_path)) if vocals_path.exists() else None
    bass_data = encode_audio_base64(str(bass_path)) if bass_path.exists() else None
    render_data = encode_audio_base64(str(render_path)) if render_path.exists() else None

    # Encode chart images
    chart_frequency_data = encode_image_base64(str(chart_frequency)) if chart_frequency.exists() else None
    chart_similarity_data = encode_image_base64(str(chart_similarity)) if chart_similarity.exists() else None
    chart_spec_orig_data = encode_image_base64(str(chart_spec_orig)) if chart_spec_orig.exists() else None
    chart_spec_rend_data = encode_image_base64(str(chart_spec_rend)) if chart_spec_rend.exists() else None
    chart_chroma_orig_data = encode_image_base64(str(chart_chroma_orig)) if chart_chroma_orig.exists() else None
    chart_chroma_rend_data = encode_image_base64(str(chart_chroma_rend)) if chart_chroma_rend.exists() else None
    has_individual_charts = any([chart_frequency_data, chart_similarity_data])

    # Load comparison results JSON for HTML charts
    comparison_results = None
    if comparison_json_path.exists():
        with open(comparison_json_path) as f:
            comparison_results = json.load(f)

    # Legacy combined chart (fallback if no JSON)
    comparison_data = encode_image_base64(str(comparison_path)) if comparison_path.exists() else None

    # Generate HTML - escape all user-provided text for HTML safety
    track_name_raw = track_info.get('title') or clean_track_name(cache_path.name)
    track_name = html.escape(track_name_raw)
    version = metadata.get('version', 1)

    # Format BPM (round to integer) - use strudel_info as fallback
    bpm_raw = metadata.get('bpm') or strudel_info.get('bpm', 0)
    bpm = round(bpm_raw) if isinstance(bpm_raw, (int, float)) and bpm_raw > 0 else 'N/A'

    key_raw = metadata.get('key') or strudel_info.get('key', 'N/A') or 'N/A'
    key = html.escape(str(key_raw))
    style_raw = metadata.get('style') or strudel_info.get('style', 'N/A') or 'N/A'
    style = html.escape(str(style_raw))
    genre_raw = metadata.get('genre') or strudel_info.get('genre', '') or ''
    genre = html.escape(str(genre_raw))

    notes = metadata.get('notes') or strudel_info.get('notes', 0)
    notes = notes if notes and notes > 0 else 'N/A'

    drum_hits_raw = metadata.get('drum_hits') or strudel_info.get('drum_hits', 0)
    drum_hits = drum_hits_raw if drum_hits_raw and drum_hits_raw > 0 else 'N/A'

    created = metadata.get('created_at', datetime.now().isoformat())

    # AI-detected style if available
    ai_style = ai_params.get('suggestions', {}).get('global', {}).get('style', '')
    if ai_style and style == 'N/A':
        style = html.escape(str(ai_style))

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIDI-grep Report: {track_name}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-orange: #d29922;
            --border: #30363d;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}

        h1 {{
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }}

        .badge-blue {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); }}
        .badge-green {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }}
        .badge-orange {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-orange); }}

        .card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card-title svg {{
            width: 18px;
            height: 18px;
            fill: var(--text-secondary);
        }}

        .audio-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}

        .audio-player {{
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 1rem;
        }}

        .audio-player label {{
            display: block;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }}

        audio {{
            width: 100%;
            height: 40px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }}

        .stat {{
            text-align: center;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }}

        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .code-block {{
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 400px;
            overflow-y: auto;
        }}

        .code-block pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .comparison-img {{
            width: 100%;
            border-radius: 6px;
            margin-top: 1rem;
        }}

        .no-data {{
            color: var(--text-secondary);
            font-style: italic;
            padding: 2rem;
            text-align: center;
        }}

        footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-align: center;
        }}

        .copy-btn {{
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            float: right;
            margin-bottom: 0.5rem;
        }}

        .copy-btn:hover {{
            opacity: 0.9;
        }}

        .playback-controls {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .play-btn {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }}

        .play-btn:hover {{
            background: var(--bg-primary);
            border-color: var(--accent);
        }}

        .play-btn.playing {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }}

        .play-btn svg {{
            width: 14px;
            height: 14px;
            fill: currentColor;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }}

        .chart-item {{
            background: var(--bg-tertiary);
            border-radius: 6px;
            overflow: hidden;
        }}

        .chart-item img {{
            width: 100%;
            display: block;
        }}

        .chart-item.full-width {{
            grid-column: span 2;
        }}

        @media (max-width: 800px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .chart-item.full-width {{
                grid-column: span 1;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{track_name}</h1>
            <div class="subtitle">
                <span class="badge badge-blue">v{version}</span>
                <span class="badge badge-green">{bpm} BPM</span>
                <span class="badge badge-orange">{key}</span>
                <span class="badge badge-blue">{style}</span>
            </div>
        </header>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/></svg>
                Audio Comparison
            </div>
            <div class="playback-controls">
                <button class="play-btn" onclick="toggleSync('drums-melodic')" id="btn-drums-melodic">
                    <svg viewBox="0 0 16 16" class="play-icon"><path d="M6.79 5.093A.5.5 0 0 0 6 5.5v5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5z"/></svg>
                    <svg viewBox="0 0 16 16" class="stop-icon" style="display:none"><path d="M5 3.5h1.5v9H5v-9zm4.5 0H11v9H9.5v-9z"/></svg>
                    Drums + Melodic
                </button>
                <button class="play-btn" onclick="toggleSync('all-stems')" id="btn-all-stems">
                    <svg viewBox="0 0 16 16" class="play-icon"><path d="M6.79 5.093A.5.5 0 0 0 6 5.5v5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5z"/></svg>
                    <svg viewBox="0 0 16 16" class="stop-icon" style="display:none"><path d="M5 3.5h1.5v9H5v-9zm4.5 0H11v9H9.5v-9z"/></svg>
                    All Stems
                </button>
                <button class="play-btn" onclick="stopAll()">
                    <svg viewBox="0 0 16 16"><path d="M5 3.5A1.5 1.5 0 0 1 6.5 2h3A1.5 1.5 0 0 1 11 3.5v9A1.5 1.5 0 0 1 9.5 14h-3A1.5 1.5 0 0 1 5 12.5v-9z"/></svg>
                    Stop All
                </button>
            </div>
            <div class="audio-section">
                <div class="audio-player">
                    <label>Melodic</label>
                    {f'<audio controls src="{melodic_data}" id="audio-melodic"></audio>' if melodic_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Drums</label>
                    {f'<audio controls src="{drums_data}" id="audio-drums"></audio>' if drums_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Vocals</label>
                    {f'<audio controls src="{vocals_data}" id="audio-vocals"></audio>' if vocals_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Bass</label>
                    {f'<audio controls src="{bass_data}" id="audio-bass"></audio>' if bass_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Strudel Render</label>
                    {f'<audio controls src="{render_data}" id="audio-render"></audio>' if render_data else '<div class="no-data">Not available</div>'}
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"/></svg>
                Analysis
            </div>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-value">{bpm}</div>
                    <div class="stat-label">BPM</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{key}</div>
                    <div class="stat-label">Key</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{notes}</div>
                    <div class="stat-label">Notes</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{drum_hits}</div>
                    <div class="stat-label">Drum Hits</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{style if style != 'N/A' else (genre if genre else 'N/A')}</div>
                    <div class="stat-label">Style</div>
                </div>
            </div>
            {generate_charts_html(comparison_results)}
        </div>

        {generate_ai_analysis_card(ai_params)}

        {f"""<div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Z"/></svg>
                Visual Comparison Charts
            </div>
            <div class="charts-grid">
                {f'<div class="chart-item"><img src="{chart_frequency_data}" alt="Frequency Bands"/></div>' if chart_frequency_data else ''}
                {f'<div class="chart-item"><img src="{chart_similarity_data}" alt="Similarity Scores"/></div>' if chart_similarity_data else ''}
                {f'<div class="chart-item"><img src="{chart_spec_orig_data}" alt="Original Spectrogram"/></div>' if chart_spec_orig_data else ''}
                {f'<div class="chart-item"><img src="{chart_spec_rend_data}" alt="Rendered Spectrogram"/></div>' if chart_spec_rend_data else ''}
                {f'<div class="chart-item"><img src="{chart_chroma_orig_data}" alt="Original Chromagram"/></div>' if chart_chroma_orig_data else ''}
                {f'<div class="chart-item"><img src="{chart_chroma_rend_data}" alt="Rendered Chromagram"/></div>' if chart_chroma_rend_data else ''}
            </div>
        </div>""" if has_individual_charts else ''}

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25Zm7.47 3.97a.75.75 0 0 1 1.06 0l2 2a.75.75 0 0 1 0 1.06l-2 2a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l.94-.94-.94-.94a.75.75 0 0 1 0-1.06Zm-4.44 0a.75.75 0 0 1 1.06 0l.94.94.94-.94a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-2 2a.75.75 0 0 1-1.06 0l-2-2a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018Z"/></svg>
                Strudel Code
                <button class="copy-btn" onclick="copyCode()">Copy Code</button>
            </div>
            <div class="code-block">
                <pre id="strudel-code">{html.escape(strudel_code) if strudel_code else 'No code generated'}</pre>
            </div>
        </div>

        <footer>
            Generated by MIDI-grep &bull; {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </footer>
    </div>

    <script>
        function copyCode() {{
            const code = document.getElementById('strudel-code').textContent;
            navigator.clipboard.writeText(code).then(() => {{
                const btn = document.querySelector('.copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy Code', 2000);
            }});
        }}

        // Synchronized playback
        const audioGroups = {{
            'drums-melodic': ['audio-drums', 'audio-melodic'],
            'all-stems': ['audio-drums', 'audio-melodic', 'audio-vocals', 'audio-bass']
        }};

        let activeGroup = null;

        function stopAll() {{
            document.querySelectorAll('audio').forEach(a => {{
                a.pause();
                a.currentTime = 0;
            }});
            document.querySelectorAll('.play-btn').forEach(btn => {{
                btn.classList.remove('playing');
                const playIcon = btn.querySelector('.play-icon');
                const stopIcon = btn.querySelector('.stop-icon');
                if (playIcon) playIcon.style.display = '';
                if (stopIcon) stopIcon.style.display = 'none';
            }});
            activeGroup = null;
        }}

        function toggleSync(group) {{
            const btn = document.getElementById('btn-' + group);
            const isPlaying = btn.classList.contains('playing');

            // Stop everything first
            stopAll();

            if (!isPlaying) {{
                // Start the group
                const ids = audioGroups[group];
                const audios = ids.map(id => document.getElementById(id)).filter(a => a);

                if (audios.length > 0) {{
                    // Sync all to time 0 and play
                    audios.forEach(a => {{
                        a.currentTime = 0;
                    }});

                    // Play all at once
                    Promise.all(audios.map(a => a.play().catch(() => {{}}))).then(() => {{
                        btn.classList.add('playing');
                        const playIcon = btn.querySelector('.play-icon');
                        const stopIcon = btn.querySelector('.stop-icon');
                        if (playIcon) playIcon.style.display = 'none';
                        if (stopIcon) stopIcon.style.display = '';
                        activeGroup = group;
                    }});

                    // Listen for end on any audio
                    audios.forEach(a => {{
                        a.onended = () => {{
                            if (activeGroup === group) stopAll();
                        }};
                    }});
                }}
            }}
        }}

        // Sync seek when user scrubs one audio
        document.querySelectorAll('audio').forEach(audio => {{
            audio.addEventListener('seeked', () => {{
                if (activeGroup) {{
                    const ids = audioGroups[activeGroup];
                    const time = audio.currentTime;
                    ids.forEach(id => {{
                        const a = document.getElementById(id);
                        if (a && a !== audio) {{
                            a.currentTime = time;
                        }}
                    }});
                }}
            }});
        }});
    </script>
</body>
</html>
'''

    # Determine output path
    if output_path is None:
        output_path = version_path / "report.html"

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Report generated: {output_path}")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate HTML report for MIDI-grep results')
    parser.add_argument('cache_dir', help='Path to cache directory (track folder)')
    parser.add_argument('-v', '--version', type=int, help='Version number (default: latest)')
    parser.add_argument('-o', '--output', help='Output HTML path')
    args = parser.parse_args()

    cache_dir = args.cache_dir

    # Find version directory
    version_dir = None
    if args.version:
        version_dir = os.path.join(cache_dir, f"v{args.version:03d}")
    else:
        # Find latest version
        for i in range(999, 0, -1):
            vdir = os.path.join(cache_dir, f"v{i:03d}")
            if os.path.isdir(vdir):
                version_dir = vdir
                break

    generate_report(cache_dir, version_dir, args.output)

if __name__ == '__main__':
    main()
