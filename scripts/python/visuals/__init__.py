"""Hydra visuals for a generated piece (the roadmap's "Hydra: visual pattern suggestions based on audio").

The picture is a real-looking scene that fits the track and moves with the music the piece plays:

1. ``vision``      measures what the track is: title/genre/key/BPM from the track metadata, CLAP zero-shot scene and
                   mood tags on ``instrumental.wav``, its spectral centroid, and the piece's sections.
2. ``hydra_code``  asks a self-hosted Ollama model for the scene prompt and a Hydra chain over that scene, and
                   validates the chain against a whitelist (fallback: a chain built from the same controls).
3. ``scene_image`` paints the scene with a self-hosted diffusion model (sd-turbo by default, diffusers on MPS).
4. ``__main__``    writes ``vNNN/visuals.strudel``: the piece unchanged plus a Hydra block whose controls are
                   queried live from the piece's own bar arrays (kick/snare/hat onsets, bass pitch, lead pitch,
                   chord size, per-bar onset density), so editing the patterns moves the picture.

Audio scoring never sees the Hydra block: ``output.strudel`` is not modified.
"""
