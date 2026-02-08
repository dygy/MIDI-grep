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

def generate_audio_player_html(melodic_data, drums_data, vocals_data, bass_data, render_data,
                               render_melodic_data=None, render_drums_data=None, render_bass_data=None):
    """Generate DAW-style arrangement view with horizontal tracks."""

    # Build audio sources for hidden audio elements
    audio_elements = []
    if melodic_data:
        audio_elements.append(f'<audio id="audio-melodic" src="{melodic_data}" preload="auto"></audio>')
    if drums_data:
        audio_elements.append(f'<audio id="audio-drums" src="{drums_data}" preload="auto"></audio>')
    if vocals_data:
        audio_elements.append(f'<audio id="audio-vocals" src="{vocals_data}" preload="auto"></audio>')
    if bass_data:
        audio_elements.append(f'<audio id="audio-bass" src="{bass_data}" preload="auto"></audio>')
    if render_data:
        audio_elements.append(f'<audio id="audio-render" src="{render_data}" preload="auto"></audio>')
    if render_melodic_data:
        audio_elements.append(f'<audio id="audio-render-melodic" src="{render_melodic_data}" preload="auto"></audio>')
    if render_drums_data:
        audio_elements.append(f'<audio id="audio-render-drums" src="{render_drums_data}" preload="auto"></audio>')
    if render_bass_data:
        audio_elements.append(f'<audio id="audio-render-bass" src="{render_bass_data}" preload="auto"></audio>')

    # Build original tracks
    original_tracks = []
    track_configs = [
        ('melodic', 'Melodic', '#3fb950', melodic_data),
        ('drums', 'Drums', '#58a6ff', drums_data),
        ('bass', 'Bass', '#f85149', bass_data),
        ('vocals', 'Vocals', '#d29922', vocals_data),
    ]

    for stem_id, label, color, data in track_configs:
        if data:
            original_tracks.append(f'''
                <div class="daw-track" data-stem="{stem_id}">
                    <div class="track-header" style="--track-color: {color};">
                        <div class="track-name">{label}</div>
                        <button class="track-mute-btn" onclick="toggleStemMute('{stem_id}')" title="Mute/Unmute">M</button>
                    </div>
                    <div class="track-waveform" onclick="seekAudio(event)">
                        <canvas class="track-canvas" data-stem="{stem_id}" data-color="{color}"></canvas>
                    </div>
                </div>
            ''')

    # Build rendered tracks
    rendered_tracks = []
    render_configs = [
        ('render-melodic', 'Melodic', '#3fb950', render_melodic_data),
        ('render-drums', 'Drums', '#58a6ff', render_drums_data),
        ('render-bass', 'Bass', '#f85149', render_bass_data),
    ]

    for stem_id, label, color, data in render_configs:
        if data:
            rendered_tracks.append(f'''
                <div class="daw-track daw-track-rendered" data-stem="{stem_id}">
                    <div class="track-header" style="--track-color: {color};">
                        <div class="track-name">{label}</div>
                        <button class="track-mute-btn" onclick="toggleStemMute('{stem_id}')" title="Mute/Unmute">M</button>
                    </div>
                    <div class="track-waveform" onclick="seekAudio(event)">
                        <canvas class="track-canvas" data-stem="{stem_id}" data-color="{color}"></canvas>
                    </div>
                </div>
            ''')

    return f'''
        <div class="card daw-card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/></svg>
                Arrangement View
            </div>

            <!-- Hidden audio elements -->
            <div style="display:none;">
                {"".join(audio_elements)}
            </div>

            <!-- Transport controls -->
            <div class="daw-transport">
                <div class="transport-center">
                    <span class="transport-time" id="time-current">0:00.0</span>
                    <span class="transport-separator">/</span>
                    <span class="transport-time transport-time-total" id="time-total">0:00.0</span>
                </div>
                <div class="transport-right">
                    <div class="transport-group-btns">
                        <button class="group-btn" onclick="playGroup('original-all')" data-group="original-all">Original</button>
                        <button class="group-btn" onclick="playGroup('render-stems')" data-group="render-stems">Rendered</button>
                        <button class="group-btn group-btn-ab" onclick="playGroup('compare-ab')" data-group="compare-ab">A/B</button>
                    </div>
                </div>
            </div>

            <!-- Timeline header -->
            <div class="daw-timeline">
                <div class="timeline-label"></div>
                <div class="timeline-ruler" id="timeline-ruler">
                    <div class="timeline-playhead" id="playhead"></div>
                </div>
            </div>

            <!-- Track sections - ISOLATED: each has its own play button -->
            <div class="daw-section">
                <div class="section-header">
                    <span>Original Stems</span>
                    <button class="section-play-btn" onclick="playGroup('original-all')" id="btn-original">
                        <span class="play-icon">▶</span> Play
                    </button>
                </div>
                <div class="daw-tracks" id="tracks-original">
                    {"".join(original_tracks)}
                </div>
            </div>

            <div class="daw-section daw-section-rendered">
                <div class="section-header">
                    <span>Rendered Stems</span>
                    <button class="section-play-btn" onclick="playGroup('render-stems')" id="btn-rendered">
                        <span class="play-icon">▶</span> Play
                    </button>
                </div>
                <div class="daw-tracks" id="tracks-rendered">
                    {"".join(rendered_tracks)}
                </div>
            </div>

            <!-- Now playing indicator -->
            <div class="daw-status" id="now-playing">Ready to play</div>
        </div>
    '''


def generate_stem_comparison_html(stem_results, stem_charts):
    """Generate per-stem comparison HTML section."""
    if not stem_results:
        return ''

    html_parts = []
    aggregate = stem_results.get('aggregate', {})
    per_stem = aggregate.get('per_stem', {})
    worst_sections = aggregate.get('worst_sections', [])
    windowed = stem_results.get('windowed', {})

    # Weighted overall score
    weighted_overall = aggregate.get('weighted_overall', 0) * 100
    overall_color = '#3fb950' if weighted_overall >= 70 else '#d29922' if weighted_overall >= 50 else '#f85149'

    html_parts.append(f'''
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 2.5rem; font-weight: bold; color: {overall_color};">{weighted_overall:.1f}%</div>
            <div style="color: var(--text-secondary);">Weighted Per-Stem Similarity</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">
                (Melodic 45% + Drums 30% + Bass 25%)
            </div>
        </div>
    ''')

    # Per-stem breakdown table
    if per_stem:
        stem_rows = ''
        for stem_name, metrics in per_stem.items():
            overall = metrics.get('overall', 0) * 100
            mfcc = metrics.get('mfcc', 0) * 100
            freq_bal = metrics.get('freq_balance', 0) * 100
            energy = metrics.get('energy', 0) * 100

            color = '#3fb950' if overall >= 70 else '#d29922' if overall >= 50 else '#f85149'
            stem_rows += f'''
                <tr>
                    <td style="padding: 0.5rem; font-weight: 600; text-transform: capitalize;">{stem_name}</td>
                    <td style="padding: 0.5rem; text-align: center; color: {color}; font-weight: 600;">{overall:.0f}%</td>
                    <td style="padding: 0.5rem; text-align: center;">{mfcc:.0f}%</td>
                    <td style="padding: 0.5rem; text-align: center;">{freq_bal:.0f}%</td>
                    <td style="padding: 0.5rem; text-align: center;">{energy:.0f}%</td>
                </tr>
            '''

        html_parts.append(f'''
            <div class="chart-item" style="padding: 1rem;">
                <h4 style="margin-bottom: 1rem; color: var(--text-primary);">Per-Stem Breakdown</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid var(--border);">
                        <th style="padding: 0.5rem; text-align: left; color: var(--text-secondary);">Stem</th>
                        <th style="padding: 0.5rem; text-align: center; color: var(--text-secondary);">Overall</th>
                        <th style="padding: 0.5rem; text-align: center; color: var(--text-secondary);">Timbre</th>
                        <th style="padding: 0.5rem; text-align: center; color: var(--text-secondary);">Freq Bal</th>
                        <th style="padding: 0.5rem; text-align: center; color: var(--text-secondary);">Energy</th>
                    </tr>
                    {stem_rows}
                </table>
            </div>
        ''')

    # Worst sections with issues
    if worst_sections:
        section_rows = ''
        for w in worst_sections[:8]:  # Show top 8
            sim = w.get('similarity', 0) * 100
            stem = w.get('stem', 'unknown')
            time_start = w.get('time_start', 0)
            time_end = w.get('time_end', 0)
            issues = w.get('issues', [])

            color = '#f85149' if sim < 40 else '#d29922' if sim < 60 else '#3fb950'
            issues_str = ', '.join(issues[:2]) if issues else 'low similarity'

            section_rows += f'''
                <tr>
                    <td style="padding: 0.4rem; text-transform: capitalize;">{stem}</td>
                    <td style="padding: 0.4rem; text-align: center;">{time_start:.0f}-{time_end:.0f}s</td>
                    <td style="padding: 0.4rem; text-align: center; color: {color}; font-weight: 600;">{sim:.0f}%</td>
                    <td style="padding: 0.4rem; color: var(--text-secondary); font-size: 0.85rem;">{html.escape(issues_str)}</td>
                </tr>
            '''

        html_parts.append(f'''
            <div class="chart-item" style="padding: 1rem;">
                <h4 style="margin-bottom: 1rem; color: var(--text-primary);">⚠️ Sections Needing Improvement</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid var(--border);">
                        <th style="padding: 0.4rem; text-align: left; color: var(--text-secondary);">Stem</th>
                        <th style="padding: 0.4rem; text-align: center; color: var(--text-secondary);">Time</th>
                        <th style="padding: 0.4rem; text-align: center; color: var(--text-secondary);">Score</th>
                        <th style="padding: 0.4rem; text-align: left; color: var(--text-secondary);">Issues</th>
                    </tr>
                    {section_rows}
                </table>
            </div>
        ''')

    # Stem overview chart (if available)
    if stem_charts.get('overview'):
        html_parts.append(f'''
            <div class="chart-item full-width">
                <img src="{stem_charts['overview']}" alt="Stem Overview"/>
                <div class="chart-caption">Per-Stem Similarity Metrics</div>
            </div>
        ''')

    # Temporal charts for each stem
    for stem_name in ['melodic', 'drums', 'bass']:
        key = f'{stem_name}_temporal'
        if stem_charts.get(key):
            html_parts.append(f'''
                <div class="chart-item full-width">
                    <img src="{stem_charts[key]}" alt="{stem_name.title()} Temporal"/>
                    <div class="chart-caption">{stem_name.title()} Stem - Time-Windowed Similarity</div>
                </div>
            ''')

    # Worst sections chart
    if stem_charts.get('worst_sections'):
        html_parts.append(f'''
            <div class="chart-item full-width">
                <img src="{stem_charts['worst_sections']}" alt="Worst Sections"/>
                <div class="chart-caption">Worst Performing Sections</div>
            </div>
        ''')

    return f'''
        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.249.249 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"/></svg>
                Per-Stem Comparison
            </div>
            <div class="charts-grid">
                {"".join(html_parts)}
            </div>
        </div>
    '''


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

    # Rendered stem files (for per-stem comparison)
    render_melodic_path = version_path / "render_melodic.wav"
    render_drums_path = version_path / "render_drums.wav"
    render_bass_path = version_path / "render_bass.wav"

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
    chart_waveform = version_path / "chart_waveform.png"
    chart_onset = version_path / "chart_onset.png"
    chart_mfcc = version_path / "chart_mfcc.png"

    # Comparison results JSON (for HTML charts)
    comparison_json_path = version_path / "comparison.json"

    # Per-stem comparison results JSON and charts
    stem_comparison_json_path = version_path / "stem_comparison.json"
    chart_stem_overview = version_path / "chart_stem_overview.png"
    chart_stem_melodic_temporal = version_path / "chart_stem_melodic_temporal.png"
    chart_stem_drums_temporal = version_path / "chart_stem_drums_temporal.png"
    chart_stem_bass_temporal = version_path / "chart_stem_bass_temporal.png"
    chart_worst_sections = version_path / "chart_worst_sections.png"

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

    # Encode rendered stem audio files
    render_melodic_data = encode_audio_base64(str(render_melodic_path)) if render_melodic_path.exists() else None
    render_drums_data = encode_audio_base64(str(render_drums_path)) if render_drums_path.exists() else None
    render_bass_data = encode_audio_base64(str(render_bass_path)) if render_bass_path.exists() else None

    # Encode chart images
    chart_frequency_data = encode_image_base64(str(chart_frequency)) if chart_frequency.exists() else None
    chart_similarity_data = encode_image_base64(str(chart_similarity)) if chart_similarity.exists() else None
    chart_spec_orig_data = encode_image_base64(str(chart_spec_orig)) if chart_spec_orig.exists() else None
    chart_spec_rend_data = encode_image_base64(str(chart_spec_rend)) if chart_spec_rend.exists() else None
    chart_chroma_orig_data = encode_image_base64(str(chart_chroma_orig)) if chart_chroma_orig.exists() else None
    chart_chroma_rend_data = encode_image_base64(str(chart_chroma_rend)) if chart_chroma_rend.exists() else None
    chart_waveform_data = encode_image_base64(str(chart_waveform)) if chart_waveform.exists() else None
    chart_onset_data = encode_image_base64(str(chart_onset)) if chart_onset.exists() else None
    chart_mfcc_data = encode_image_base64(str(chart_mfcc)) if chart_mfcc.exists() else None
    has_individual_charts = any([chart_frequency_data, chart_similarity_data])

    # Load comparison results JSON for HTML charts
    comparison_results = None
    if comparison_json_path.exists():
        with open(comparison_json_path) as f:
            comparison_results = json.load(f)

    # Load per-stem comparison results
    stem_comparison_results = None
    if stem_comparison_json_path.exists():
        with open(stem_comparison_json_path) as f:
            stem_comparison_results = json.load(f)

    # Encode stem comparison charts
    stem_charts = {}
    if chart_stem_overview.exists():
        stem_charts['overview'] = encode_image_base64(str(chart_stem_overview))
    if chart_stem_melodic_temporal.exists():
        stem_charts['melodic_temporal'] = encode_image_base64(str(chart_stem_melodic_temporal))
    if chart_stem_drums_temporal.exists():
        stem_charts['drums_temporal'] = encode_image_base64(str(chart_stem_drums_temporal))
    if chart_stem_bass_temporal.exists():
        stem_charts['bass_temporal'] = encode_image_base64(str(chart_stem_bass_temporal))
    if chart_worst_sections.exists():
        stem_charts['worst_sections'] = encode_image_base64(str(chart_worst_sections))

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

        /* ========== DAW ARRANGEMENT VIEW ========== */
        .daw-card {{
            background: var(--bg-primary);
            padding: 0;
            overflow: hidden;
        }}

        .daw-card .card-title {{
            padding: 1rem 1.5rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            margin: 0;
        }}

        /* Transport bar */
        .daw-transport {{
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            gap: 1.5rem;
        }}

        .transport-left {{
            display: flex;
            gap: 0.5rem;
        }}

        .transport-btn {{
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
        }}

        .transport-play {{
            background: var(--accent-green);
            color: var(--bg-primary);
        }}

        .transport-play:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(63, 185, 80, 0.4);
        }}

        .transport-play.playing {{
            background: var(--accent-orange);
        }}

        .transport-play.playing .transport-icon {{
            content: '⏸';
        }}

        .transport-stop {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }}

        .transport-stop:hover {{
            background: #f85149;
            color: white;
            border-color: #f85149;
        }}

        .transport-icon {{
            font-size: 1rem;
        }}

        .transport-center {{
            display: flex;
            align-items: baseline;
            gap: 0.25rem;
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }}

        .transport-time {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }}

        .transport-time-total {{
            font-size: 1rem;
            color: var(--text-secondary);
        }}

        .transport-separator {{
            color: var(--text-secondary);
        }}

        .transport-right {{
            margin-left: auto;
        }}

        .transport-group-btns {{
            display: flex;
            gap: 0.5rem;
        }}

        .group-btn {{
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .group-btn:hover {{
            border-color: var(--accent);
        }}

        .group-btn.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }}

        .group-btn-ab {{
            border-color: var(--accent-orange);
        }}

        .group-btn-ab:hover {{
            background: rgba(210, 153, 34, 0.2);
        }}

        /* Timeline */
        .daw-timeline {{
            display: flex;
            height: 24px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }}

        .timeline-label {{
            width: 120px;
            flex-shrink: 0;
            border-right: 1px solid var(--border);
        }}

        .timeline-ruler {{
            flex: 1;
            position: relative;
            background: repeating-linear-gradient(
                90deg,
                var(--border) 0px,
                var(--border) 1px,
                transparent 1px,
                transparent 100px
            );
        }}

        .timeline-playhead {{
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: 100%;
            background: #ff6b6b;
            z-index: 10;
            pointer-events: none;
        }}

        .timeline-playhead::after {{
            content: '';
            position: absolute;
            top: 0;
            left: -4px;
            width: 10px;
            height: 10px;
            background: #ff6b6b;
            clip-path: polygon(50% 100%, 0 0, 100% 0);
        }}

        .timeline-marker {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.65rem;
            color: var(--text-secondary);
            font-family: 'SF Mono', Monaco, monospace;
            pointer-events: none;
        }}

        /* Track sections */
        .daw-section {{
            border-bottom: 1px solid var(--border);
        }}

        .daw-section-rendered {{
            background: linear-gradient(90deg, rgba(63, 185, 80, 0.05) 0%, transparent 120px);
        }}

        .section-header {{
            padding: 0.5rem 1rem;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            font-weight: 600;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-play-btn {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.4rem 1rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            text-transform: none;
            letter-spacing: normal;
        }}

        .section-play-btn:hover {{
            background: var(--accent-hover);
        }}

        .section-play-btn.playing {{
            background: #f85149;
        }}

        .section-play-btn .play-icon {{
            font-size: 0.65rem;
        }}

        .daw-tracks {{
            display: flex;
            flex-direction: column;
        }}

        /* Individual track */
        .daw-track {{
            display: flex;
            height: 60px;
            border-bottom: 1px solid var(--border);
        }}

        .daw-track:last-child {{
            border-bottom: none;
        }}

        .track-header {{
            width: 120px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 0.75rem;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            border-left: 3px solid var(--track-color, var(--accent));
        }}

        .track-name {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .track-controls {{
            display: flex;
            gap: 0.25rem;
        }}

        .track-btn {{
            width: 20px;
            height: 20px;
            border: none;
            border-radius: 3px;
            font-size: 0.6rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.1s ease;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .track-solo.active {{
            background: var(--accent-orange);
            color: var(--bg-primary);
        }}

        .track-mute.active {{
            background: #f85149;
            color: white;
        }}

        .track-mute-btn {{
            width: 24px;
            height: 24px;
            border: none;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            cursor: pointer;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            transition: all 0.15s ease;
        }}

        .track-mute-btn:hover {{
            background: var(--bg-primary);
        }}

        .track-mute-btn.muted {{
            background: #f85149;
            color: white;
        }}

        .track-waveform {{
            flex: 1;
            position: relative;
            background: var(--bg-primary);
            cursor: pointer;
            overflow: hidden;
        }}

        .track-canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* Playhead line that spans all tracks */
        .daw-tracks {{
            position: relative;
        }}

        .tracks-playhead {{
            position: absolute;
            top: 0;
            left: 120px;
            width: calc(100% - 120px);
            height: 100%;
            pointer-events: none;
            z-index: 100;
        }}

        .tracks-playhead::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 2px;
            height: 100%;
            background: #ff6b6b;
            box-shadow: 0 0 8px rgba(255, 107, 107, 0.6);
        }}

        /* Status bar */
        .daw-status {{
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border);
        }}

        /* ========== END DAW VIEW ========== */

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

        .chart-caption {{
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: center;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
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

        {generate_audio_player_html(melodic_data, drums_data, vocals_data, bass_data, render_data,
                                     render_melodic_data, render_drums_data, render_bass_data)}

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

        {generate_stem_comparison_html(stem_comparison_results, stem_charts)}

        {f"""<div class="card">
            <div class="card-title">
                <svg viewBox="0 0 16 16"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Z"/></svg>
                Visual Comparison Charts
            </div>
            <div class="charts-grid">
                {f'<div class="chart-item"><img src="{chart_similarity_data}" alt="Similarity Scores"/><div class="chart-caption">Similarity Scores (with Overall Gauge)</div></div>' if chart_similarity_data else ''}
                {f'<div class="chart-item"><img src="{chart_frequency_data}" alt="Frequency Bands"/><div class="chart-caption">Frequency Bands Distribution</div></div>' if chart_frequency_data else ''}
                {f'<div class="chart-item"><img src="{chart_spec_orig_data}" alt="Original Spectrogram"/><div class="chart-caption">Original - Mel Spectrogram</div></div>' if chart_spec_orig_data else ''}
                {f'<div class="chart-item"><img src="{chart_spec_rend_data}" alt="Rendered Spectrogram"/><div class="chart-caption">Rendered - Mel Spectrogram</div></div>' if chart_spec_rend_data else ''}
                {f'<div class="chart-item"><img src="{chart_chroma_orig_data}" alt="Original Chromagram"/><div class="chart-caption">Original - Chromagram (Pitch Classes)</div></div>' if chart_chroma_orig_data else ''}
                {f'<div class="chart-item"><img src="{chart_chroma_rend_data}" alt="Rendered Chromagram"/><div class="chart-caption">Rendered - Chromagram (Pitch Classes)</div></div>' if chart_chroma_rend_data else ''}
                {f'<div class="chart-item full-width"><img src="{chart_waveform_data}" alt="Waveform Comparison"/><div class="chart-caption">Waveform Comparison (Original vs Rendered)</div></div>' if chart_waveform_data else ''}
                {f'<div class="chart-item full-width"><img src="{chart_onset_data}" alt="Onset Strength"/><div class="chart-caption">Onset Strength (Rhythm/Attack Transients)</div></div>' if chart_onset_data else ''}
                {f'<div class="chart-item full-width"><img src="{chart_mfcc_data}" alt="MFCC Comparison"/><div class="chart-caption">MFCC Comparison (Timbre Fingerprint)</div></div>' if chart_mfcc_data else ''}
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

        // ========== DAW ENGINE ==========
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const waveformData = {{}};  // Store decoded audio data for waveform drawing

        let activeGroup = 'original-all';  // Default to original stems
        let isPlaying = false;
        let abMode = false;
        let abToggleInterval = null;
        let maxDuration = 0;

        // Audio group definitions - ISOLATED: original and rendered NEVER play together
        const audioGroups = {{
            'original-all': ['audio-melodic', 'audio-drums', 'audio-bass', 'audio-vocals'],
            'render-stems': ['audio-render-melodic', 'audio-render-drums', 'audio-render-bass']
        }};

        // Which stems belong to which group (for solo/mute isolation)
        const stemToGroup = {{
            'melodic': 'original-all',
            'drums': 'original-all',
            'bass': 'original-all',
            'vocals': 'original-all',
            'render-melodic': 'render-stems',
            'render-drums': 'render-stems',
            'render-bass': 'render-stems'
        }};

        // Stem state for solo/mute (per group)
        const stemState = {{
            melodic: {{ muted: false, solo: false }},
            drums: {{ muted: false, solo: false }},
            bass: {{ muted: false, solo: false }},
            vocals: {{ muted: false, solo: false }},
            'render-melodic': {{ muted: false, solo: false }},
            'render-drums': {{ muted: false, solo: false }},
            'render-bass': {{ muted: false, solo: false }}
        }};

        // Initialize audio context and decode audio
        async function initAudio() {{
            const audioElements = document.querySelectorAll('audio');
            for (const audio of audioElements) {{
                const id = audio.id;
                if (!id) continue;

                try {{
                    const response = await fetch(audio.src);
                    const arrayBuffer = await response.arrayBuffer();
                    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                    waveformData[id] = audioBuffer;
                    if (audioBuffer.duration > maxDuration) {{
                        maxDuration = audioBuffer.duration;
                    }}
                }} catch (e) {{
                    console.log('Could not decode audio:', id);
                }}
            }}
            drawAllWaveforms();
            updateTotalTime();
            generateTimelineMarkers();
        }}

        // Draw waveform on a single track canvas
        function drawTrackWaveform(canvas, buffer, color) {{
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();

            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = rect.height;

            // Background
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, width, height);

            // Draw waveform
            const data = buffer.getChannelData(0);
            const step = Math.ceil(data.length / width);
            const amp = height * 0.4;

            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();

            for (let i = 0; i < width; i++) {{
                const idx = Math.floor(i * step);
                let min = 1.0, max = -1.0;

                for (let j = 0; j < step; j++) {{
                    const val = data[idx + j] || 0;
                    if (val < min) min = val;
                    if (val > max) max = val;
                }}

                const y1 = height/2 + min * amp;
                const y2 = height/2 + max * amp;

                ctx.moveTo(i, y1);
                ctx.lineTo(i, y2);
            }}

            ctx.stroke();

            // Center line
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.beginPath();
            ctx.moveTo(0, height/2);
            ctx.lineTo(width, height/2);
            ctx.stroke();
        }}

        // Draw all waveforms
        function drawAllWaveforms() {{
            document.querySelectorAll('.track-canvas').forEach(canvas => {{
                const stem = canvas.dataset.stem;
                const color = canvas.dataset.color || '#58a6ff';
                const audioId = 'audio-' + stem;
                const buffer = waveformData[audioId];

                if (buffer) {{
                    drawTrackWaveform(canvas, buffer, color);
                }}
            }});
        }}

        // Generate timeline markers
        function generateTimelineMarkers() {{
            const ruler = document.getElementById('timeline-ruler');
            if (!ruler || maxDuration === 0) return;

            // Calculate marker interval (aim for ~10 markers)
            const interval = Math.ceil(maxDuration / 10);
            let html = '';

            for (let t = 0; t <= maxDuration; t += interval) {{
                const pct = (t / maxDuration) * 100;
                html += `<span class="timeline-marker" style="left: ${{pct}}%;">${{formatTime(t)}}</span>`;
            }}

            ruler.insertAdjacentHTML('afterbegin', html);
        }}

        // Update total time display
        function updateTotalTime() {{
            if (maxDuration > 0) {{
                document.getElementById('time-total').textContent = formatTimeMs(maxDuration);
            }}
        }}

        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}

        function formatTimeMs(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 10);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}.${{ms}}`;
        }}

        // Play current group (NEVER plays both original and rendered together)
        function playAll() {{
            if (isPlaying) {{
                // Pause only the active group
                const groupIds = audioGroups[activeGroup] || [];
                groupIds.forEach(id => {{
                    const audio = document.getElementById(id);
                    if (audio) audio.pause();
                }});
                isPlaying = false;
                return;
            }}

            // Resume audio context if suspended
            if (audioCtx.state === 'suspended') {{
                audioCtx.resume();
            }}

            // Play ONLY the active group (default: original-all)
            const groupIds = audioGroups[activeGroup] || audioGroups['original-all'];
            const audios = groupIds.map(id => document.getElementById(id)).filter(a => a);

            if (audios.length === 0) return;

            // Mute everything first
            document.querySelectorAll('audio').forEach(a => {{ a.volume = 0; a.pause(); a.currentTime = 0; }});

            // Unmute and play only the active group
            audios.forEach(a => {{ a.volume = 1; }});
            updateStemVolumes();

            Promise.all(audios.map(a => a.play().catch(() => {{}}))).then(() => {{
                isPlaying = true;
                updateNowPlaying(activeGroup);
                updateGroupButtons(activeGroup);
                startPlayheadAnimation();
            }});

            audios[0].onended = () => stopAll();
        }}

        // Play a group of audio tracks (ISOLATED - only one group plays)
        function playGroup(group) {{
            // Handle A/B comparison mode
            if (group === 'compare-ab') {{
                startABCompare();
                return;
            }}

            // Stop everything first
            stopAll();

            // Set active group
            activeGroup = group;

            const ids = audioGroups[group] || [];
            const audios = ids.map(id => document.getElementById(id)).filter(a => a);

            if (audios.length === 0) return;

            // Resume audio context if suspended
            if (audioCtx.state === 'suspended') {{
                audioCtx.resume();
            }}

            // Stop and mute ALL audios first
            document.querySelectorAll('audio').forEach(a => {{
                a.pause();
                a.currentTime = 0;
                a.volume = 0;
            }});

            // Unmute and set up ONLY this group
            audios.forEach(a => {{
                a.currentTime = 0;
                a.volume = 1;
            }});
            updateStemVolumes();

            // Play ONLY the selected group (not all)
            Promise.all(audios.map(a => a.play().catch(() => {{}}))).then(() => {{
                isPlaying = true;
                updateNowPlaying(group);
                updateGroupButtons(group);
                startPlayheadAnimation();
            }});

            audios[0].onended = () => stopAll();
        }}

        // A/B comparison - switch between original and rendered every 4 seconds
        function startABCompare() {{
            stopAll();

            let showOriginal = true;
            const origAudios = ['audio-melodic', 'audio-drums', 'audio-bass'].map(id => document.getElementById(id)).filter(a => a);
            const rendAudios = ['audio-render-melodic', 'audio-render-drums', 'audio-render-bass'].map(id => document.getElementById(id)).filter(a => a);

            if (origAudios.length === 0 && rendAudios.length === 0) return;

            const allAudios = [...origAudios, ...rendAudios];
            allAudios.forEach(a => {{ a.currentTime = 0; a.volume = 0; }});

            // Start all but mute rendered
            origAudios.forEach(a => {{ a.volume = 1; }});
            rendAudios.forEach(a => {{ a.volume = 0; }});

            Promise.all(allAudios.map(a => a.play().catch(() => {{}}))).then(() => {{
                activeGroup = 'compare-ab';
                isPlaying = true;
                abMode = true;
                updateNowPlaying('A/B: Original');
                updateGroupButtons('compare-ab');
                startPlayheadAnimation();
            }});

            // Toggle every 4 seconds
            abToggleInterval = setInterval(() => {{
                showOriginal = !showOriginal;
                if (showOriginal) {{
                    origAudios.forEach(a => {{ a.volume = 1; }});
                    rendAudios.forEach(a => {{ a.volume = 0; }});
                    updateNowPlaying('A/B: Original');
                }} else {{
                    origAudios.forEach(a => {{ a.volume = 0; }});
                    rendAudios.forEach(a => {{ a.volume = 1; }});
                    updateNowPlaying('A/B: Rendered');
                }}
            }}, 4000);

            if (origAudios.length > 0) origAudios[0].onended = () => stopAll();
        }}

        // Stop all playback
        function stopAll() {{
            document.querySelectorAll('audio').forEach(a => {{
                a.pause();
                a.currentTime = 0;
                a.volume = 1;
            }});

            if (abToggleInterval) {{
                clearInterval(abToggleInterval);
                abToggleInterval = null;
            }}

            activeGroup = null;
            isPlaying = false;
            abMode = false;

            updateNowPlaying('Ready to play');
            updateGroupButtons(null);
            document.getElementById('time-current').textContent = '0:00.0';
            document.getElementById('playhead').style.left = '0';
        }}

        // Update now playing display
        function updateNowPlaying(group) {{
            const labels = {{
                'original-all': 'Playing: Original Stems',
                'render-stems': 'Playing: Rendered Stems',
                'compare-ab': 'A/B Comparison Mode',
                'all': 'Playing: All Stems'
            }};
            const el = document.getElementById('now-playing');
            if (el) {{
                el.textContent = typeof group === 'string' && group.startsWith('A/B:') ? group : (labels[group] || group || 'Ready to play');
            }}
        }}

        // Update group button states
        function updateGroupButtons(activeGroupName) {{
            document.querySelectorAll('.group-btn[data-group]').forEach(btn => {{
                const group = btn.getAttribute('data-group');
                btn.classList.toggle('active', group === activeGroupName);
            }});

            // Update section play buttons
            const btnOriginal = document.getElementById('btn-original');
            const btnRendered = document.getElementById('btn-rendered');

            if (btnOriginal) {{
                const isPlaying = activeGroupName === 'original-all';
                btnOriginal.classList.toggle('playing', isPlaying);
                btnOriginal.querySelector('.play-icon').textContent = isPlaying ? '⏸' : '▶';
                btnOriginal.lastChild.textContent = isPlaying ? ' Stop' : ' Play';
            }}
            if (btnRendered) {{
                const isPlaying = activeGroupName === 'render-stems';
                btnRendered.classList.toggle('playing', isPlaying);
                btnRendered.querySelector('.play-icon').textContent = isPlaying ? '⏸' : '▶';
                btnRendered.lastChild.textContent = isPlaying ? ' Stop' : ' Play';
            }}
        }}

        // Playhead animation
        let playheadAnimId = null;
        function startPlayheadAnimation() {{
            if (playheadAnimId) cancelAnimationFrame(playheadAnimId);

            function updatePlayhead() {{
                if (!isPlaying) return;

                // Get first playing audio for time
                const audio = Array.from(document.querySelectorAll('audio')).find(a => !a.paused && a.duration);

                if (audio && audio.duration) {{
                    const progress = audio.currentTime / audio.duration;
                    const playhead = document.getElementById('playhead');

                    if (playhead) {{
                        playhead.style.left = `${{progress * 100}}%`;
                    }}

                    document.getElementById('time-current').textContent = formatTimeMs(audio.currentTime);
                }}

                playheadAnimId = requestAnimationFrame(updatePlayhead);
            }}

            updatePlayhead();
        }}

        // Seek on waveform click
        function seekAudio(event) {{
            const wrapper = event.currentTarget;
            const rect = wrapper.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const progress = x / rect.width;

            // Seek all audios
            document.querySelectorAll('audio').forEach(a => {{
                if (a.duration) {{
                    a.currentTime = progress * a.duration;
                }}
            }});

            // Update playhead immediately
            document.getElementById('playhead').style.left = `${{progress * 100}}%`;
            document.getElementById('time-current').textContent = formatTimeMs(progress * maxDuration);
        }}

        // Solo/Mute controls
        function toggleSolo(stem) {{
            const state = stemState[stem];
            if (!state) return;

            state.solo = !state.solo;

            const btn = document.querySelector(`.daw-track[data-stem="${{stem}}"] .track-solo`);
            if (btn) btn.classList.toggle('active', state.solo);

            updateStemVolumes();
        }}

        function toggleMute(stem) {{
            const state = stemState[stem];
            if (!state) return;

            state.muted = !state.muted;

            const btn = document.querySelector(`.daw-track[data-stem="${{stem}}"] .track-mute`);
            if (btn) btn.classList.toggle('active', state.muted);

            updateStemVolumes();
        }}

        // Toggle mute for individual stem
        function toggleStemMute(stem) {{
            const state = stemState[stem];
            if (!state) return;

            state.muted = !state.muted;

            // Update button visual
            const btn = document.querySelector(`.daw-track[data-stem="${{stem}}"] .track-mute-btn`);
            if (btn) {{
                btn.classList.toggle('muted', state.muted);
            }}

            updateStemVolumes();
        }}

        function updateStemVolumes() {{
            // Only apply volumes to stems in the active group
            const activeIds = audioGroups[activeGroup] || [];

            for (const [stem, state] of Object.entries(stemState)) {{
                const audio = document.getElementById('audio-' + stem);
                if (!audio) continue;

                // If this stem is not in the active group, always mute
                const stemGroup = stemToGroup[stem];
                if (stemGroup !== activeGroup) {{
                    audio.volume = 0;
                    continue;
                }}

                // Simple: if muted, volume is 0, otherwise 1
                audio.volume = state.muted ? 0 : 1;
            }}
        }}

        // Initialize on load
        document.addEventListener('DOMContentLoaded', () => {{
            // Wait a bit for audio elements to be ready
            setTimeout(initAudio, 500);

            // Handle resize
            window.addEventListener('resize', () => {{
                drawAllWaveforms();
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
