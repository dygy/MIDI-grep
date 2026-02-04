#!/usr/bin/env python3
"""
Model Repository Manager.
Handles storage, retrieval, and GitHub synchronization of trained models.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import hashlib


class ModelRepository:
    """Manage a repository of generative sound models."""

    def __init__(self,
                 local_path: str = "models",
                 github_repo: Optional[str] = None,
                 github_branch: str = "main"):
        """
        Initialize model repository.

        Args:
            local_path: Local path for models
            github_repo: GitHub repo (e.g., "username/midi-grep-sounds")
            github_branch: Branch to use
        """
        self.local_path = Path(local_path)
        self.local_path.mkdir(parents=True, exist_ok=True)

        self.github_repo = github_repo
        self.github_branch = github_branch

        self.index_path = self.local_path / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load or create the model index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)

        return {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "models": {}
        }

    def _save_index(self):
        """Save the model index."""
        self.index["updated"] = datetime.now().isoformat()
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_model(self,
                  model_path: str,
                  model_id: str,
                  embedding: List[float],
                  metadata: Dict) -> Dict:
        """
        Add a model to the repository.

        Args:
            model_path: Path to model directory
            model_id: Unique model identifier
            embedding: Timbre embedding vector
            metadata: Model metadata

        Returns:
            Updated model info
        """
        model_path = Path(model_path).resolve()
        dest_path = (self.local_path / model_id).resolve()

        # Only copy if source and destination are different
        if model_path != dest_path:
            if dest_path.exists():
                print(f"Model {model_id} already exists, updating...")
                shutil.rmtree(dest_path)
            shutil.copytree(model_path, dest_path)
        else:
            # Model already in correct location
            if not dest_path.exists():
                raise FileNotFoundError(f"Model not found at {dest_path}")

        self.index["models"][model_id] = {
            "embedding": embedding,
            "metadata": metadata,
            "path": str(dest_path.relative_to(self.local_path.resolve())),
            "added": datetime.now().isoformat()
        }

        self._save_index()
        print(f"Added model: {model_id}")

        return self.index["models"][model_id]

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model info by ID."""
        return self.index["models"].get(model_id)

    def list_models(self) -> List[str]:
        """List all model IDs."""
        return list(self.index["models"].keys())

    def find_similar(self,
                     embedding: List[float],
                     threshold: float = 0.88,
                     top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find models with similar timbre.

        Args:
            embedding: Query embedding
            threshold: Minimum similarity
            top_k: Max results

        Returns:
            List of (model_id, similarity) tuples
        """
        import numpy as np

        query = np.array(embedding)
        query = query / (np.linalg.norm(query) + 1e-8)

        results = []
        for model_id, model_info in self.index["models"].items():
            model_emb = np.array(model_info["embedding"])
            model_emb = model_emb / (np.linalg.norm(model_emb) + 1e-8)

            similarity = float(np.dot(query, model_emb))

            if similarity >= threshold:
                results.append((model_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get local path to model."""
        model_info = self.get_model(model_id)
        if model_info:
            return self.local_path / model_info["path"]
        return None

    def get_strudel_url(self, model_id: str) -> str:
        """
        Get URL for loading model in Strudel.

        Returns local or GitHub URL depending on configuration.
        """
        if self.github_repo:
            return f"https://raw.githubusercontent.com/{self.github_repo}/{self.github_branch}/models/{model_id}/"
        else:
            return f"http://localhost:5555/{model_id}/"

    # GitHub Integration

    def sync_to_github(self):
        """Push local models to GitHub repository."""
        if not self.github_repo:
            print("No GitHub repo configured")
            return False

        print(f"Syncing to GitHub: {self.github_repo}")

        # Check if git repo exists
        if not (self.local_path / ".git").exists():
            self._init_git_repo()

        # Add all changes
        subprocess.run(["git", "add", "."], cwd=self.local_path, check=True)

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", f"Update models {datetime.now().isoformat()}"],
            cwd=self.local_path,
            capture_output=True,
            text=True
        )

        if "nothing to commit" in result.stdout + result.stderr:
            print("No changes to commit")
            return True

        # Push
        subprocess.run(
            ["git", "push", "origin", self.github_branch],
            cwd=self.local_path,
            check=True
        )

        print("Synced to GitHub successfully")
        return True

    def sync_from_github(self):
        """Pull latest models from GitHub."""
        if not self.github_repo:
            print("No GitHub repo configured")
            return False

        if not (self.local_path / ".git").exists():
            # Clone
            subprocess.run([
                "git", "clone",
                f"https://github.com/{self.github_repo}.git",
                str(self.local_path)
            ], check=True)
        else:
            # Pull
            subprocess.run(
                ["git", "pull", "origin", self.github_branch],
                cwd=self.local_path,
                check=True
            )

        # Reload index
        self.index = self._load_index()
        print("Synced from GitHub successfully")
        return True

    def _init_git_repo(self):
        """Initialize git repository."""
        subprocess.run(["git", "init"], cwd=self.local_path, check=True)

        # Create .gitignore
        gitignore = self.local_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n.DS_Store\n")

        # Add remote
        subprocess.run([
            "git", "remote", "add", "origin",
            f"https://github.com/{self.github_repo}.git"
        ], cwd=self.local_path, check=True)


class LocalModelServer:
    """Simple HTTP server for serving models to Strudel."""

    def __init__(self, models_path: str, port: int = 5555):
        self.models_path = Path(models_path)
        self.port = port

    def serve(self):
        """Start the server."""
        import http.server
        import socketserver

        os.chdir(self.models_path)

        class CORSHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET')
                self.send_header('Cache-Control', 'no-store')
                super().end_headers()

        with socketserver.TCPServer(("", self.port), CORSHandler) as httpd:
            print(f"Serving models at http://localhost:{self.port}")
            print(f"Press Ctrl+C to stop")
            httpd.serve_forever()


def generate_model_id(audio_path: str, voice: str = "melodic") -> str:
    """Generate unique model ID from audio file."""
    with open(audio_path, 'rb') as f:
        content_hash = hashlib.md5(f.read()).hexdigest()[:8]

    return f"{voice}_{content_hash}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model repository manager")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--path", default="models", help="Models path")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start local server")
    serve_parser.add_argument("--path", default="models", help="Models path")
    serve_parser.add_argument("--port", type=int, default=5555, help="Port")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync with GitHub")
    sync_parser.add_argument("--path", default="models", help="Models path")
    sync_parser.add_argument("--repo", required=True, help="GitHub repo")
    sync_parser.add_argument("--push", action="store_true", help="Push to GitHub")
    sync_parser.add_argument("--pull", action="store_true", help="Pull from GitHub")

    args = parser.parse_args()

    if args.command == "list":
        repo = ModelRepository(args.path)
        models = repo.list_models()
        print(f"Models ({len(models)}):")
        for model_id in models:
            info = repo.get_model(model_id)
            print(f"  - {model_id}: {info['metadata'].get('type', 'unknown')}")

    elif args.command == "serve":
        server = LocalModelServer(args.path, args.port)
        server.serve()

    elif args.command == "sync":
        repo = ModelRepository(args.path, github_repo=args.repo)
        if args.push:
            repo.sync_to_github()
        if args.pull:
            repo.sync_from_github()
