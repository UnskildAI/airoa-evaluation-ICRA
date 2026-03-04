from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    normalized = expected_sha256.strip().lower()
    if not normalized:
        return
    actual = _sha256_file(path)
    if actual != normalized:
        raise ValueError(
            f"Checkpoint SHA256 mismatch for {path}: expected={normalized}, actual={actual}"
        )


def _resolve_local_path(uri_or_path: str) -> Path:
    path = Path(uri_or_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_dir():
        raise ValueError(
            f"Checkpoint path must point to a file, got directory: {path}"
        )
    return path.resolve()


def _download_r2_checkpoint(uri: str, cache_dir: Path) -> Path:
    parsed = urlparse(uri)
    bucket = parsed.netloc.strip() or os.getenv("R2_BUCKET", "").strip()
    key = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(
            "R2 bucket is missing. Use r2://<bucket>/<key> or set R2_BUCKET."
        )
    if not key:
        raise ValueError(f"R2 object key is missing in URI: {uri}")

    endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()

    missing = []
    if not endpoint:
        missing.append("R2_ENDPOINT_URL")
    if not access_key:
        missing.append("R2_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("R2_SECRET_ACCESS_KEY")
    if missing:
        raise ValueError(f"Missing required R2 environment variables: {', '.join(missing)}")

    uri_hash = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    local_name = f"{uri_hash}_{Path(key).name}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / local_name
    tmp_path = cache_dir / f".{local_name}.tmp"

    if local_path.exists():
        return local_path

    try:
        import boto3
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "boto3 is required for r2:// checkpoint downloads. "
            "Install pinned deps via `pip install -r requirements.txt`."
        ) from exc

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("R2_REGION", "auto"),
    )
    client.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(local_path)
    return local_path


def resolve_checkpoint(
    uri_or_path: str,
    *,
    expected_sha256: str | None = None,
    cache_dir: str | Path = ".cache/checkpoints",
) -> Path:
    """Resolve checkpoint from local path or r2:// URI and verify checksum."""
    value = uri_or_path.strip()
    if not value:
        raise ValueError("Checkpoint path/URI must be non-empty.")

    cache = Path(cache_dir).expanduser().resolve()
    if value.startswith("r2://"):
        resolved = _download_r2_checkpoint(value, cache)
    else:
        resolved = _resolve_local_path(value)

    _verify_sha256(resolved, expected_sha256)
    return resolved
