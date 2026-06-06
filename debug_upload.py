#!/usr/bin/env python3
"""Diagnostic script for debugging large file upload through vite proxy.

Creates a test tar.gz file of configurable size, uploads it through the
vite proxy, and logs every detail of the HTTP exchange to help identify
where a 400 with empty body originates.

Usage:
    python debug_upload.py --url http://127.0.0.1:3045 --api-key scai_XXX --size-mb 200

The script sends the upload via both the proxy AND directly to the backend
(if accessible) to isolate whether the issue is in the proxy layer.
"""
import argparse
import io
import json
import logging
import sys
import tarfile
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("upload_diag")


def create_test_archive(size_mb: int) -> Path:
    """Create a tar.gz with a single file of ~size_mb MB."""
    out = Path(f"/tmp/diag_upload_{size_mb}mb.tar.gz")
    if out.exists() and abs(out.stat().st_size - size_mb * 1024 * 1024) < 1024:
        log.info("Reusing existing archive: %s (%d bytes)", out, out.stat().st_size)
        return out

    log.info("Creating %d MB test archive …", size_mb)
    data = b"\x00" * (size_mb * 1024 * 1024)
    with tarfile.open(out, "w:gz") as tf:
        info = tarfile.TarInfo(name="test_data.bin")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    log.info("Archive created: %s (%d bytes)", out, out.stat().st_size)
    return out


def upload_with_detailed_logging(
    base_url: str,
    api_key: str,
    archive_path: Path,
    trust: bool = False,
    timeout: float = 86400,
    label: str = "",
) -> dict | None:
    """Upload archive and log every step of the HTTP exchange."""

    size = archive_path.stat().st_size
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
        "X-Filename": archive_path.name,
    }
    if trust:
        headers["X-Trust"] = "true"

    log.info("[%s] Starting upload: %s (%d bytes = %.1f MB)", label, archive_path, size, size / 1024 / 1024)
    log.info("[%s] Target: %s", label, base_url)
    log.info("[%s] Headers: %s", label, {k: v for k, v in headers.items() if k != "Authorization"})

    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(timeout),
    )

    t0 = time.time()
    result = None
    try:
        with open(archive_path, "rb") as fh:
            log.info("[%s] Sending POST /api/session/import …", label)
            r = client.post(
                "/api/session/import",
                content=fh,
                headers=headers,
            )

        elapsed = time.time() - t0
        log.info("[%s] Response received in %.1fs", label, elapsed)
        log.info("[%s] Status: %d", label, r.status_code)
        log.info("[%s] Headers: %s", label, dict(r.headers))
        log.info("[%s] Body (first 500 chars): %s", label, r.text[:500])
        log.info("[%s] Body length: %d", label, len(r.content))

        if r.status_code >= 400:
            log.error("[%s] UPLOAD FAILED: HTTP %d", label, r.status_code)
            log.error("[%s] Response body: %s", label, r.text)
            result = {
                "status": "error",
                "http_status": r.status_code,
                "body": r.text,
                "elapsed_s": elapsed,
                "headers": dict(r.headers),
            }
        else:
            try:
                body = r.json()
                log.info("[%s] Success response: %s", label, json.dumps(body, indent=2))
                import_id = body.get("data", {}).get("import_id")
                session_id = body.get("session_id")
                result = {
                    "status": "ok",
                    "http_status": r.status_code,
                    "import_id": import_id,
                    "session_id": session_id,
                    "elapsed_s": elapsed,
                }
            except Exception as e:
                log.warning("[%s] Could not parse JSON: %s", label, e)
                result = {
                    "status": "parse_error",
                    "http_status": r.status_code,
                    "body": r.text[:500],
                    "elapsed_s": elapsed,
                }
    except httpx.TimeoutException as e:
        elapsed = time.time() - t0
        log.error("[%s] TIMEOUT after %.1fs: %s", label, elapsed, e)
        result = {"status": "timeout", "error": str(e), "elapsed_s": elapsed}
    except httpx.ConnectError as e:
        elapsed = time.time() - t0
        log.error("[%s] CONNECT ERROR after %.1fs: %s", label, elapsed, e)
        result = {"status": "connect_error", "error": str(e), "elapsed_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        log.error("[%s] UNEXPECTED ERROR after %.1fs: %s", label, elapsed, e, exc_info=True)
        result = {"status": "unexpected_error", "error": str(e), "elapsed_s": elapsed}
    finally:
        client.close()

    return result


def poll_import(base_url: str, api_key: str, import_id: str, max_wait: float = 600):
    """Poll import status until completion or failure."""
    log.info("Polling import %s for up to %.0fs …", import_id, max_wait)
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(30),
    )
    t0 = time.time()
    try:
        while time.time() - t0 < max_wait:
            r = client.get(f"/api/session/import/{import_id}")
            if r.status_code == 404:
                log.warning("Import %s not found (404) — job may have been cleaned up", import_id)
                return None
            try:
                data = r.json()
            except Exception:
                log.error("Poll response: status=%d body=%s", r.status_code, r.text[:200])
                return None

            status = data.get("data", {}).get("status", data.get("status", "unknown"))
            phase = data.get("data", {}).get("phase", "")
            bp = data.get("data", {}).get("bytes_processed", 0)
            bt = data.get("data", {}).get("bytes_total", 0)
            fp = data.get("data", {}).get("files_processed", 0)
            cf = data.get("data", {}).get("current_file", "")

            elapsed = time.time() - t0
            log.info(
                "  [%5.1fs] status=%s phase=%s bytes=%s/%s files=%d file=%s",
                elapsed, status, phase,
                f"{bp/(1024*1024):.1f}MB" if bp else "0",
                f"{bt/(1024*1024):.1f}MB" if bt else "?",
                fp,
                cf[:40] if cf else "",
            )

            if status in ("completed", "failed", "cancelled"):
                return data

            time.sleep(2)
    finally:
        client.close()

    return None


def main():
    parser = argparse.ArgumentParser(description="Diagnostic upload through vite proxy")
    parser.add_argument("--url", default="http://127.0.0.1:3045", help="Base URL (vite proxy)")
    parser.add_argument("--backend-url", default=None, help="Direct backend URL (e.g., http://127.0.0.1:8000). Skipped if not set.")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--size-mb", type=int, default=200, help="Test archive size in MB (default 200)")
    parser.add_argument("--timeout", type=float, default=86400, help="Upload timeout in seconds (default 86400)")
    parser.add_argument("--trust", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true", help="Skip proxy upload, only test direct backend")
    parser.add_argument("--skip-poll", action="store_true", help="Skip polling, just do upload")
    args = parser.parse_args()

    archive = create_test_archive(args.size_mb)

    results = {}

    # Test via proxy
    if not args.skip_proxy:
        log.info("=" * 60)
        log.info("TEST 1: Upload via VITE PROXY (%s)", args.url)
        log.info("=" * 60)
        result = upload_with_detailed_logging(
            args.url, args.api_key, archive,
            trust=args.trust, timeout=args.timeout, label="PROXY",
        )
        results["proxy"] = result

        if result and result.get("status") == "ok" and result.get("import_id") and not args.skip_poll:
            poll_result = poll_import(args.url, args.api_key, result["import_id"])
            results["proxy_poll"] = poll_result

    # Test direct backend (if accessible)
    direct_url = args.backend_url
    if direct_url and not args.skip_proxy:
        log.info("")
        log.info("=" * 60)
        log.info("TEST 2: Upload DIRECT to BACKEND (%s)", direct_url)
        log.info("=" * 60)
        result = upload_with_detailed_logging(
            direct_url, args.api_key, archive,
            trust=args.trust, timeout=args.timeout, label="DIRECT",
        )
        results["direct"] = result

        if result and result.get("status") == "ok" and result.get("import_id") and not args.skip_poll:
            poll_result = poll_import(direct_url, args.api_key, result["import_id"])
            results["direct_poll"] = poll_result

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for key, val in results.items():
        log.info("  %s: %s", key, json.dumps(val, indent=2, default=str))


if __name__ == "__main__":
    main()