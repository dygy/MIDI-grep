#!/usr/bin/env python3
"""
Generate HTML report for MIDI-grep extraction results.
Creates a single-page report with audio players, comparison charts, and Strudel code.
Styled like Playwright/Jupyter reports.
"""

import argparse
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

def generate_report(cache_dir, version_dir, output_path=None):
    """Generate HTML report for a version."""

    cache_path = Path(cache_dir)
    version_path = Path(version_dir) if version_dir else cache_path

    # Find files
    melodic_path = cache_path / "melodic.wav"
    if not melodic_path.exists():
        melodic_path = cache_path / "piano.wav"
    drums_path = cache_path / "drums.wav"

    render_path = version_path / "render.wav"
    if not render_path.exists():
        # Try legacy naming
        for f in version_path.glob("render*.wav"):
            render_path = f
            break

    strudel_path = version_path / "output.strudel"
    if not strudel_path.exists():
        strudel_path = version_path / "output_latest.strudel"

    comparison_path = version_path / "comparison.png"
    if not comparison_path.exists():
        for f in version_path.glob("comparison*.png"):
            comparison_path = f
            break

    metadata_path = version_path / "metadata.json"
    track_meta_path = cache_path / "metadata.json"

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    track_info = {}
    if track_meta_path.exists():
        with open(track_meta_path) as f:
            track_info = json.load(f)

    # Read Strudel code
    strudel_code = ""
    if strudel_path.exists():
        with open(strudel_path) as f:
            strudel_code = f.read()

    # Encode files as base64
    melodic_data = encode_audio_base64(str(melodic_path)) if melodic_path.exists() else None
    drums_data = encode_audio_base64(str(drums_path)) if drums_path.exists() else None
    render_data = encode_audio_base64(str(render_path)) if render_path.exists() else None
    comparison_data = encode_image_base64(str(comparison_path)) if comparison_path.exists() else None

    # Generate HTML
    track_name = track_info.get('title', cache_path.name)
    version = metadata.get('version', 1)
    bpm = metadata.get('bpm', 'N/A')
    key = metadata.get('key', 'N/A')
    style = metadata.get('style', 'N/A')
    notes = metadata.get('notes', 'N/A')
    drum_hits = metadata.get('drum_hits', 'N/A')
    created = metadata.get('created_at', datetime.now().isoformat())

    html = f'''<!DOCTYPE html>
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
            <div class="audio-section">
                <div class="audio-player">
                    <label>Original Melodic Stem</label>
                    {f'<audio controls src="{melodic_data}"></audio>' if melodic_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Original Drums Stem</label>
                    {f'<audio controls src="{drums_data}"></audio>' if drums_data else '<div class="no-data">Not available</div>'}
                </div>
                <div class="audio-player">
                    <label>Strudel Render</label>
                    {f'<audio controls src="{render_data}"></audio>' if render_data else '<div class="no-data">Not available</div>'}
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
                    <div class="stat-value">{style}</div>
                    <div class="stat-label">Style</div>
                </div>
            </div>
            {f'<img class="comparison-img" src="{comparison_data}" alt="Frequency Comparison">' if comparison_data else ''}
        </div>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25Zm7.47 3.97a.75.75 0 0 1 1.06 0l2 2a.75.75 0 0 1 0 1.06l-2 2a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l.94-.94-.94-.94a.75.75 0 0 1 0-1.06Zm-4.44 0a.75.75 0 0 1 1.06 0l.94.94.94-.94a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-2 2a.75.75 0 0 1-1.06 0l-2-2a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018Z"/></svg>
                Strudel Code
                <button class="copy-btn" onclick="copyCode()">Copy Code</button>
            </div>
            <div class="code-block">
                <pre id="strudel-code">{strudel_code if strudel_code else 'No code generated'}</pre>
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
    </script>
</body>
</html>
'''

    # Determine output path
    if output_path is None:
        output_path = version_path / "report.html"

    with open(output_path, 'w') as f:
        f.write(html)

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
