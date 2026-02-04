#!/usr/bin/env python3
"""
RAVE Pipeline CLI.
Command-line interface for the generative sound model pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rave.pipeline import GenerativePipeline
from rave.repository import ModelRepository, LocalModelServer
from rave.timbre_embeddings import TimbreAnalyzer, TimbreIndex


def cmd_process(args):
    """Process stems through the generative pipeline."""
    pipeline = GenerativePipeline(
        models_path=args.models,
        github_repo=args.github,
        similarity_threshold=args.threshold,
        training_mode=args.mode
    )

    result = pipeline.process_track(
        args.stems_dir,
        args.track_id,
        args.output
    )

    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_serve(args):
    """Start local model server."""
    server = LocalModelServer(args.models, args.port)
    print(f"Starting model server on port {args.port}")
    print(f"Models directory: {args.models}")
    print(f"\nUse in Strudel: await samples('http://localhost:{args.port}/<model_id>/')")
    server.serve()
    return 0


def cmd_list(args):
    """List models in repository."""
    repo = ModelRepository(args.models, github_repo=args.github)
    models = repo.list_models()

    if not models:
        print("No models found.")
        return 0

    print(f"\nModels ({len(models)}):")
    print("-" * 60)

    for model_id in models:
        info = repo.get_model(model_id)
        metadata = info.get("metadata", {})
        model_type = metadata.get("type", "unknown")
        source = metadata.get("source_audio", "unknown")

        print(f"\n{model_id}")
        print(f"  Type: {model_type}")
        print(f"  Source: {Path(source).name if source != 'unknown' else 'unknown'}")
        print(f"  URL: {repo.get_strudel_url(model_id)}")

    return 0


def cmd_search(args):
    """Search for similar models."""
    analyzer = TimbreAnalyzer()
    repo = ModelRepository(args.models, github_repo=args.github)

    print(f"Analyzing: {args.audio}")
    embedding = analyzer.extract_embedding(args.audio)

    print(f"Searching for similar models (threshold: {args.threshold:.0%})...")
    matches = repo.find_similar(embedding.tolist(), threshold=args.threshold)

    if not matches:
        print("\nNo matching models found.")
        print("Consider training a new model with: rave-cli train <audio>")
        return 0

    print(f"\nFound {len(matches)} similar model(s):")
    print("-" * 60)

    for model_id, similarity in matches:
        info = repo.get_model(model_id)
        print(f"\n{model_id}")
        print(f"  Similarity: {similarity:.1%}")
        print(f"  URL: {repo.get_strudel_url(model_id)}")

    return 0


def cmd_train(args):
    """Train a new model."""
    from rave.trainer import RAVETrainer, GranularTrainer

    # Use absolute paths
    output_path = Path(args.output).resolve()
    audio_path = Path(args.audio).resolve()

    if args.mode == "rave":
        trainer = RAVETrainer(str(output_path))
        result = trainer.train(
            str(audio_path),
            args.name,
            epochs=args.epochs
        )
    else:
        trainer = GranularTrainer(str(output_path))
        result = trainer.train(
            str(audio_path),
            args.name,
            grain_ms=args.grain_ms
        )

    # Add to repository if requested
    if args.add_to_repo:
        analyzer = TimbreAnalyzer()
        embedding = analyzer.extract_embedding(str(audio_path))

        repo = ModelRepository(str(output_path), github_repo=args.github)
        model_path = output_path / args.name
        repo.add_model(str(model_path), args.name, embedding.tolist(), result)

        if args.github and args.sync:
            print("\nSyncing to GitHub...")
            repo.sync_to_github()

    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_sync(args):
    """Sync models with GitHub."""
    repo = ModelRepository(args.models, github_repo=args.github)

    if args.push:
        print("Pushing to GitHub...")
        repo.sync_to_github()
    elif args.pull:
        print("Pulling from GitHub...")
        repo.sync_from_github()
    else:
        print("Specify --push or --pull")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAVE Generative Sound Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process stems and generate Strudel with trained models
  python -m rave.cli process ./stems --track-id mytrack --output ./output

  # Train a new granular model (fast)
  python -m rave.cli train audio.wav --name my_piano --mode granular

  # Train a RAVE model (high quality, slow)
  python -m rave.cli train audio.wav --name my_synth --mode rave --epochs 500

  # Search for similar existing models
  python -m rave.cli search audio.wav --threshold 0.85

  # Start local server for Strudel
  python -m rave.cli serve --port 5555

  # Sync models to GitHub
  python -m rave.cli sync --github username/repo --push
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process stems through pipeline")
    process_parser.add_argument("stems_dir", help="Directory with separated stems")
    process_parser.add_argument("--track-id", required=True, help="Unique track ID")
    process_parser.add_argument("--output", "-o", default="output", help="Output directory")
    process_parser.add_argument("--models", default="models", help="Models directory")
    process_parser.add_argument("--github", help="GitHub repo (user/repo)")
    process_parser.add_argument("--mode", choices=["granular", "rave"], default="granular",
                               help="Training mode for new models")
    process_parser.add_argument("--threshold", type=float, default=0.88,
                               help="Similarity threshold for reusing models")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start local model server")
    serve_parser.add_argument("--models", default="models", help="Models directory")
    serve_parser.add_argument("--port", type=int, default=5555, help="Server port")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--models", default="models", help="Models directory")
    list_parser.add_argument("--github", help="GitHub repo")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar models")
    search_parser.add_argument("audio", help="Audio file to match")
    search_parser.add_argument("--models", default="models", help="Models directory")
    search_parser.add_argument("--github", help="GitHub repo")
    search_parser.add_argument("--threshold", type=float, default=0.85,
                              help="Minimum similarity threshold")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("audio", help="Source audio file")
    train_parser.add_argument("--name", required=True, help="Model name")
    train_parser.add_argument("--output", default="models", help="Output directory")
    train_parser.add_argument("--mode", choices=["granular", "rave"], default="granular",
                             help="Training mode")
    train_parser.add_argument("--epochs", type=int, default=500, help="RAVE epochs")
    train_parser.add_argument("--grain-ms", type=int, default=100, help="Granular grain size")
    train_parser.add_argument("--add-to-repo", action="store_true",
                             help="Add to model repository")
    train_parser.add_argument("--github", help="GitHub repo")
    train_parser.add_argument("--sync", action="store_true", help="Sync to GitHub after")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync with GitHub")
    sync_parser.add_argument("--models", default="models", help="Models directory")
    sync_parser.add_argument("--github", required=True, help="GitHub repo")
    sync_parser.add_argument("--push", action="store_true", help="Push to GitHub")
    sync_parser.add_argument("--pull", action="store_true", help="Pull from GitHub")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "process": cmd_process,
        "serve": cmd_serve,
        "list": cmd_list,
        "search": cmd_search,
        "train": cmd_train,
        "sync": cmd_sync,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
