#!/usr/bin/env python3
"""
Upload a Strudel sample-pack to Cloudflare R2 (or any S3-compatible store) and
return the public base URL that Strudel's ``samples()`` can fetch.

The pack directory is produced by ``build_sample_pack.py`` and contains
``strudel.json`` plus ``drums/``, ``bass/``, ``melodic/``, ``loops/`` WAVs.

Two backends, auto-selected:
  * ``aws``      - ``aws s3 cp --recursive ... --endpoint-url <r2 endpoint>``
                   Fast (one process, parallel). Needs an R2 *S3 API token*
                   (access key id + secret) and the account id.
  * ``wrangler`` - ``npx wrangler r2 object put`` per file. Works with the
                   wrangler OAuth login (``wrangler login``); no S3 keys needed,
                   but slower (one node process per file).

Hosting is interchangeable with the local model server: whatever base URL this
prints, the codegen drops into ``await samples("<base>/strudel.json")``.

Config via flags or environment:
  R2_ACCOUNT_ID         Cloudflare account id (for the S3 endpoint)
  R2_BUCKET             bucket name
  R2_PUBLIC_BASE        public base URL of the bucket (https://pub-xxxx.r2.dev
                        or a custom domain). The printed sample URL is
                        ``<R2_PUBLIC_BASE>/<prefix>``.
  R2_ACCESS_KEY_ID      R2 S3 API token access key id   (aws backend)
  R2_SECRET_ACCESS_KEY  R2 S3 API token secret          (aws backend)

Example:
  python upload_r2.py --pack-dir .../sample_pack --prefix regime-clt
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

# WAV is not reliably in the system mimetypes db; pin the ones we emit so the
# browser fetch + CORS gets a sane Content-Type.
CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".json": "application/json",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
}


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower()) or (
        mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    )


def iter_files(pack_dir: Path) -> list[Path]:
    return sorted(p for p in pack_dir.rglob("*") if p.is_file())


def endpoint_url(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def pick_backend(requested: str, account_id: str, access_key: str, secret: str) -> str:
    if requested != "auto":
        return requested
    if account_id and access_key and secret and have("aws"):
        return "aws"
    if have("npx") or have("wrangler"):
        return "wrangler"
    raise SystemExit(
        "No usable backend: set R2_ACCOUNT_ID + R2_ACCESS_KEY_ID + "
        "R2_SECRET_ACCESS_KEY (aws), or install wrangler/npx."
    )


def upload_aws(
    pack_dir: Path, bucket: str, prefix: str, account_id: str,
    access_key: str, secret: str,
) -> None:
    if not (account_id and access_key and secret):
        raise SystemExit("aws backend needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret,
        "AWS_DEFAULT_REGION": "auto",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    ep = endpoint_url(account_id)
    dest = f"s3://{bucket}/{prefix.strip('/')}/"
    # One recursive copy; set the common content-type for the bulk WAVs.
    cmd = [
        "aws", "s3", "cp", str(pack_dir), dest,
        "--recursive", "--endpoint-url", ep,
        "--exclude", "*.json", "--content-type", "audio/wav",
    ]
    print(f"[aws] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, env=env, check=True)
    # JSON files need their own content-type pass.
    cmd_json = [
        "aws", "s3", "cp", str(pack_dir), dest,
        "--recursive", "--endpoint-url", ep,
        "--exclude", "*", "--include", "*.json",
        "--content-type", "application/json",
    ]
    print(f"[aws] {' '.join(cmd_json)}", file=sys.stderr)
    subprocess.run(cmd_json, env=env, check=True)


def upload_wrangler(pack_dir: Path, bucket: str, prefix: str) -> None:
    runner = ["npx", "-y", "wrangler@latest"] if not have("wrangler") else ["wrangler"]
    files = iter_files(pack_dir)
    print(f"[wrangler] uploading {len(files)} files to {bucket}/{prefix}", file=sys.stderr)
    for i, f in enumerate(files, 1):
        key = f"{prefix.strip('/')}/{f.relative_to(pack_dir).as_posix()}"
        cmd = [
            *runner, "r2", "object", "put", f"{bucket}/{key}",
            "--file", str(f), "--content-type", content_type_for(f), "--remote",
        ]
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write(res.stdout + res.stderr)
            raise SystemExit(
                f"wrangler upload failed for {key}. If this is an auth error, run "
                f"`npx wrangler login` in an interactive terminal first."
            )
        if i % 10 == 0 or i == len(files):
            print(f"[wrangler] {i}/{len(files)}", file=sys.stderr)


def set_cors(bucket: str, account_id: str, access_key: str, secret: str, origins: list[str]) -> None:
    """Allow browser GETs from the Strudel origin(s) on a custom-domain bucket.

    (r2.dev managed buckets already serve permissive CORS; this is for custom
    domains.) Requires the aws S3 API + R2 keys.
    """
    if not (account_id and access_key and secret):
        print("[cors] skipped: no S3 API keys", file=sys.stderr)
        return
    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret,
        "AWS_DEFAULT_REGION": "auto",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    cfg = {"CORSRules": [{"AllowedOrigins": origins, "AllowedMethods": ["GET", "HEAD"],
                          "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]}
    cmd = ["aws", "s3api", "put-bucket-cors", "--bucket", bucket,
           "--cors-configuration", json.dumps(cfg),
           "--endpoint-url", endpoint_url(account_id)]
    print(f"[cors] {origins}", file=sys.stderr)
    subprocess.run(cmd, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload a Strudel sample-pack to Cloudflare R2.")
    ap.add_argument("--pack-dir", required=True, type=Path)
    ap.add_argument("--prefix", required=True, help="key prefix in the bucket (e.g. track id)")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", ""))
    ap.add_argument("--account-id", default=os.environ.get("R2_ACCOUNT_ID", ""))
    ap.add_argument("--public-base", default=os.environ.get("R2_PUBLIC_BASE", ""))
    ap.add_argument("--backend", choices=["auto", "aws", "wrangler"], default="auto")
    ap.add_argument("--set-cors", nargs="*", metavar="ORIGIN",
                    help="set bucket CORS to allow these origins (custom-domain buckets)")
    args = ap.parse_args()

    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")

    pack_dir = args.pack_dir
    if not ((pack_dir / "samples.json").exists() or (pack_dir / "strudel.json").exists()):
        raise SystemExit(f"no samples.json/strudel.json in {pack_dir}; run build_sample_pack.py "
                         f"or generate_hybrid_strudel.py first")
    if not args.bucket:
        raise SystemExit("set --bucket or R2_BUCKET")

    backend = pick_backend(args.backend, args.account_id, access_key, secret)
    print(f"[upload_r2] backend={backend} bucket={args.bucket} prefix={args.prefix}", file=sys.stderr)

    if args.set_cors is not None:
        set_cors(args.bucket, args.account_id, access_key, secret, args.set_cors or ["*"])

    if backend == "aws":
        upload_aws(pack_dir, args.bucket, args.prefix, args.account_id, access_key, secret)
    else:
        upload_wrangler(pack_dir, args.bucket, args.prefix)

    base = (args.public_base.rstrip("/") + "/" + args.prefix.strip("/")) if args.public_base else ""
    result = {
        "backend": backend,
        "bucket": args.bucket,
        "prefix": args.prefix,
        "public_base": base,
        "strudel_samples_url": (base + "/strudel.json") if base else "(set --public-base / R2_PUBLIC_BASE)",
    }
    print(json.dumps(result, indent=2))
    if not base:
        print("\nWARNING: no --public-base/R2_PUBLIC_BASE; upload done but the "
              "public URL for Strudel is unknown.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
