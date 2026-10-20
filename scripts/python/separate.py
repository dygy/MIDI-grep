#!/usr/bin/env python3
"""
Stem separation using Demucs, written LOSSLESS and sample-aligned.

Modes:
  - piano (default): instrumental stem only (everything except vocals)
  - drums: drums stem only
  - full: melodic / bass / drums / vocals stems

Invariants (tests/test_separate_alignment.py):
  1. Stems are float32 WAV at 44.1 kHz.  There is NO mp3 intermediate: an mp3 round trip
     adds the encoder delay (+25.1 ms = 1105 samples measured on both reference tracks) and
     codec noise, so every stem used to be late against original.wav.
  2. Every stem has exactly the length of the decoded input (no delay, no padding).
  3. The stems sum back to the decoded input at unity gain (mixture consistency, see
     `mixture_consistency`; the vocals stem is never altered by it).  Demucs alone is only ~26-31 dB SNR and per-stem gains drift
     by up to 1.6 dB.

Demucs runs in-process (demucs.apply.apply_model) instead of through its CLI because the CLI
can only write 16-bit/mp3 and rescales the whole file when it clips, which breaks unity gain.

Changing anything in this file changes the stem-cache version (internal/cache/cache.go hashes
separate.py), so cached stems from an older separator are invalidated automatically.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

SAMPLE_RATE = 44100  # demucs native rate; the stems keep it (resampling back would add a filter)

# Quality presets.  `vocal_model` = run that model's vocals SPECIALIST first (a htdemucs_ft bag holds
# one fine-tuned model per source; only the vocals one is run), subtract its vocals from the mix and
# run `model` on the instrumental residual.  Measured on 30 s excerpts of both reference tracks
# (see the commit message): this removes the drum->vocals onset coupling (VAGABUNDO 0.097-0.112 ->
# -0.03, different-song floor -0.017) at ~2x the single-pass cost, where the full htdemucs_ft bag
# costs ~4x and 6-source htdemucs_6s was worse on every metric.
PRESETS = {
    'fast':   {'model': 'htdemucs',    'shifts': 0, 'overlap': 0.25, 'vocal_model': None},
    'normal': {'model': 'htdemucs',    'shifts': 1, 'overlap': 0.25, 'vocal_model': 'htdemucs_ft'},
    'high':   {'model': 'htdemucs_ft', 'shifts': 3, 'overlap': 0.5,  'vocal_model': 'htdemucs_ft'},
    'best':   {'model': 'htdemucs_ft', 'shifts': 5, 'overlap': 0.75, 'vocal_model': 'htdemucs_ft'},
}

# Smoothing window (samples) for the local-power weights of the residual split.
_MC_WINDOW = 4096


def decode_audio(path, sample_rate=SAMPLE_RATE):
    """Decode any input to float32 (channels=2, samples) at `sample_rate` through ffmpeg."""
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', path,
           '-ar', str(sample_rate), '-ac', '2', '-f', 'f32le', '-']
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f'ffmpeg could not decode {path}: {proc.stderr.decode(errors="replace")[:300]}')
    return np.frombuffer(proc.stdout, dtype='<f4').reshape(-1, 2).T.copy()


def mixture_consistency(stems, mix, window=_MC_WINDOW):
    """Make `stems` sum to `mix` exactly.

    The residual (mix - sum(stems)) is shared between the stems in proportion to each
    stem's local power (Wiener-style), so a quiet stem is not given loud residual and
    the stems that carry the energy absorb the model error.  With all stems silent the
    residual is split equally.
    """
    from scipy.ndimage import uniform_filter1d

    names = list(stems)
    arr = np.stack([stems[n] for n in names]).astype(np.float64)       # (S, C, N)
    resid = mix.astype(np.float64) - arr.sum(0)                        # (C, N)
    power = uniform_filter1d(arr ** 2, size=window, axis=-1, mode='nearest').sum(1)  # (S, N)
    total = power.sum(0)
    flat = total < 1e-12
    weights = np.where(flat, 1.0 / len(names), power / np.where(flat, 1.0, total))  # (S, N)
    arr = arr + weights[:, None, :] * resid[None]
    return {n: arr[i].astype(np.float32) for i, n in enumerate(names)}


def _apply(model_name, wav, shifts, overlap, only=None):
    """Run a demucs model over a (2, N) float32 array; returns {source: (2, N)}.

    `only='vocals'` runs just that source's specialist from a htdemucs_ft bag (1 of its 4 models).
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    if only is not None:
        specialist = model.models[model.sources.index(only)]   # bag weights are one-hot per source
        model, names = specialist, [only]
    else:
        names = None
    model.eval()
    x = torch.from_numpy(wav)
    ref = x.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((x - mu) / sd)[None], shifts=shifts, split=True,
                          overlap=overlap, device=_device(), progress=False, num_workers=0)[0]
    out = (out * sd + mu).numpy()
    sources = list(model.sources)
    return {n: out[sources.index(n)] for n in (names or sources)}


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def separate_waveform(wav, preset, apply_fn=_apply):
    """Separate a decoded mix into {'vocals','drums','bass','melodic'} summing to `wav`.

    `apply_fn(model_name, wav, shifts, overlap, only=None) -> {source: array}` is injectable so
    the alignment / unity-gain invariants can be tested without model weights.
    """
    shifts, overlap = preset['shifts'], preset['overlap']
    if preset.get('vocal_model'):
        vocals = apply_fn(preset['vocal_model'], wav, shifts, overlap, only='vocals')['vocals']
        raw = apply_fn(preset['model'], wav - vocals, shifts, overlap)   # drums/bass/other of the residual
    else:
        raw = apply_fn(preset['model'], wav, shifts, overlap)
        vocals = raw['vocals']
    # Models with extra sources (htdemucs_6s: guitar, piano) all belong to the melodic stem.
    melodic = sum(raw[k] for k in ('other', 'guitar', 'piano') if k in raw)
    stems = {'vocals': vocals, 'drums': raw['drums'], 'bass': raw['bass'], 'melodic': melodic}
    for name, arr in stems.items():
        if arr.shape != wav.shape:
            raise RuntimeError(f'{name} stem has shape {arr.shape}, input is {wav.shape}')
    # Unity sum: the model error goes to the INSTRUMENTAL stems only.  Spreading it over the vocals
    # too lowered REG vocals SDR 22.4 -> 18.0 dB on the ground-truth self-test (error that is not
    # singing landed in the vocals stem), the opposite of what the bleed fix is for.
    out = mixture_consistency({k: v for k, v in stems.items() if k != 'vocals'}, wav - vocals)
    out['vocals'] = vocals.astype(np.float32)
    return out


def write_stem(path, stem):
    import soundfile as sf
    sf.write(path, stem.T, SAMPLE_RATE, subtype='FLOAT')
    return path


def main():
    parser = argparse.ArgumentParser(description='Separate audio into stems using Demucs')
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('output_dir', help='Output directory for stems')
    parser.add_argument('--mode', choices=['piano', 'drums', 'full'], default='piano',
                        help='Separation mode: piano (default), drums, or full')
    parser.add_argument('--quality', choices=sorted(PRESETS), default='normal',
                        help='Quality preset (see PRESETS)')
    parser.add_argument('--prefix', default='',
                        help='Prefix for output stem names (e.g., --prefix render produces render_melodic.wav)')
    args = parser.parse_args()

    if not os.path.exists(args.input_audio):
        print(f"Error: Input file not found: {args.input_audio}", file=sys.stderr)
        sys.exit(1)

    preset = PRESETS[args.quality]
    print(f"Separating with model={preset['model']}, vocals-first={preset['vocal_model']}, "
          f"shifts={preset['shifts']}, overlap={preset['overlap']} (lossless float32 wav, unity-sum)")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        wav = decode_audio(args.input_audio)
        stems = separate_waveform(wav, preset)

        def out(name):
            return os.path.join(args.output_dir, f'{args.prefix + "_" if args.prefix else ""}{name}.wav')

        if args.mode == 'piano':
            instrumental = stems['drums'] + stems['bass'] + stems['melodic']
            print(f"Piano/instrumental stem saved to: {write_stem(out('piano'), instrumental)}")
        elif args.mode == 'drums':
            print(f"Drums stem saved to: {write_stem(out('drums'), stems['drums'])}")
        else:
            for name in ('melodic', 'bass', 'drums', 'vocals'):
                print(f"{name.capitalize()} stem saved to: {write_stem(out(name), stems[name])}")
    except Exception as e:  # noqa: BLE001 - CLI boundary, report and exit non-zero
        print(f"Error: Stem separation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
