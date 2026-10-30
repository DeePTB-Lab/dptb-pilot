from __future__ import annotations

import json
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


ARTIFACT_MANIFEST_NAME = "artifacts.json"
_TEXT_PREVIEW_LIMIT = 8_192
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_BOHRIUM_RE = re.compile(r"bohrium://[^\s<>()\[\]{}\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s<>|\"'`]+")
_UNIX_PATH_RE = re.compile(r"(?<![:/\w])/(?:[^\s<>()\[\]{}\"']+/)*[^\s<>()\[\]{}\"']+")
_RELATIVE_ARTIFACT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.(?:csv|json|yaml|yml|png|jpg|jpeg|pdf|zip|h5|pth|vasp|cif|out|log|npy|npz)(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_BARE_ARTIFACT_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])[A-Za-z0-9_.-]+\.(?:csv|json|yaml|yml|png|jpg|jpeg|pdf|zip|h5|pth|vasp|cif|out|log|npy|npz)(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_STAGEABLE_SCHEMES = {"bohrium", "oss", "http", "https"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(work_root: str | Path, session_id: str) -> Path:
    return Path(work_root).expanduser().resolve() / session_id / ARTIFACT_MANIFEST_NAME


def _files_dir(work_root: str | Path, session_id: str) -> Path:
    return Path(work_root).expanduser().resolve() / session_id / "files"


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "tasks": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "tasks": []}
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        return {"version": 1, "tasks": []}
    data.setdefault("version", 1)
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _strip_reference(value: str) -> str:
    value = value.strip().strip("`'\"")
    while value and value[-1] in ".,;:!?":
        value = value[:-1]
    # Tool output often appends a path separator when serializing a file path
    # (notably JSON/Windows escaped paths). A trailing separator makes a valid
    # file look like a directory during workspace existence checks.
    value = value.rstrip("/\\")
    return value


def _normalize_workspace_path(value: str | None) -> str | None:
    if not value:
        return None
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    return normalized or None


def _iter_references(response_text: str) -> Iterable[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    candidates.extend(match.group(1).split(maxsplit=1)[0] for match in _MARKDOWN_LINK_RE.finditer(response_text))
    candidates.extend(match.group(0) for match in _BOHRIUM_RE.finditer(response_text))
    candidates.extend(match.group(0) for match in _WINDOWS_PATH_RE.finditer(response_text))
    candidates.extend(match.group(0) for match in _UNIX_PATH_RE.finditer(response_text))
    # Tool results commonly serialize relative paths (e.g. ``outputs/a.npy``)
    # rather than URLs or absolute paths. Capture those too.
    candidates.extend(match.group(0) for match in _RELATIVE_ARTIFACT_PATH_RE.finditer(response_text))
    candidates.extend(match.group(0) for match in _BARE_ARTIFACT_NAME_RE.finditer(response_text))

    for candidate in candidates:
        reference = _strip_reference(candidate)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        yield reference


def _relative_workspace_path(reference: str, files_dir: Path, session_id: str) -> str | None:
    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https"}:
        path = unquote(parsed.path)
    else:
        path = unquote(reference)

    download_prefix = f"/api/download/{session_id}/"
    if path.startswith(download_prefix):
        return path[len(download_prefix):].lstrip("/")

    normalized = path.replace("\\", "/")
    marker = f"/{session_id}/files/"
    if marker in normalized:
        return normalized.split(marker, 1)[1].lstrip("/")
    if "/files/" in normalized:
        candidate = normalized.rsplit("/files/", 1)[1].lstrip("/")
        if (files_dir / candidate).exists():
            return candidate

    # Relative paths emitted by tools are relative to the session workspace,
    # not to the API process' current directory.
    relative_candidate = normalized.lstrip("/")
    if relative_candidate and (files_dir / relative_candidate).is_file():
        return relative_candidate

    try:
        local_path = Path(path).expanduser().resolve()
        return local_path.relative_to(files_dir).as_posix()
    except (OSError, RuntimeError, ValueError):
        pass

    basename = Path(normalized).name
    if basename and (files_dir / basename).exists():
        return basename
    return None


def extract_artifacts(response_text: str, work_root: str | Path, session_id: str) -> list[dict[str, Any]]:
    files_dir = _files_dir(work_root, session_id)
    artifacts: list[dict[str, Any]] = []
    for reference in _iter_references(response_text):
        workspace_path = _relative_workspace_path(reference, files_dir, session_id)
        if not workspace_path and not reference.startswith("bohrium://"):
            continue
        artifacts.append({
            "id": uuid.uuid4().hex,
            "source_path": reference,
            "workspace_path": workspace_path,
        })
    return artifacts


def _artifact_identity(artifact: dict[str, Any]) -> tuple[str, str]:
    workspace_path = _normalize_workspace_path(artifact.get("workspace_path"))
    if workspace_path:
        return "workspace", workspace_path.casefold()
    source_path = _strip_reference(str(artifact.get("source_path") or ""))
    return "source", source_path.casefold()


def record_completed_task_artifacts(
    work_root: str | Path,
    session_id: str,
    chat_id: str,
    task: str,
    response_text: str,
    artifact_text: str = "",
    ignored_artifact_text: str = "",
) -> dict[str, Any] | None:
    # Include references observed in intermediate tool/model events as well as
    # the final answer. This keeps links that the final summary omits.
    combined_text = "\n".join(value for value in (response_text, artifact_text) if value)
    artifacts = extract_artifacts(combined_text, work_root, session_id)
    if ignored_artifact_text:
        ignored_identities = {
            _artifact_identity(item)
            for item in extract_artifacts(ignored_artifact_text, work_root, session_id)
        }
        produced_identities = {
            _artifact_identity(item)
            for item in extract_artifacts(artifact_text, work_root, session_id)
        }
        artifacts = [
            item
            for item in artifacts
            if _artifact_identity(item) not in ignored_identities
            or _artifact_identity(item) in produced_identities
        ]
    if not artifacts:
        return None

    manifest_path = _manifest_path(work_root, session_id)
    manifest = _load_manifest(manifest_path)
    task_record = {
        "id": uuid.uuid4().hex,
        "chat_id": chat_id,
        "task": task,
        "completed_at": _utc_now(),
        "artifacts": artifacts,
    }
    manifest["tasks"].append(task_record)
    _write_manifest(manifest_path, manifest)
    return task_record


def _safe_artifact_file(files_dir: Path, workspace_path: str | None) -> Path | None:
    workspace_path = _normalize_workspace_path(workspace_path)
    if not workspace_path:
        return None
    try:
        candidate = (files_dir / workspace_path).resolve()
        candidate.relative_to(files_dir)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def get_recorded_artifact(
    work_root: str | Path,
    session_id: str,
    artifact_id: str,
) -> dict[str, Any] | None:
    """Return one persisted artifact without trusting a client-supplied URI."""
    manifest = _load_manifest(_manifest_path(work_root, session_id))
    for task in manifest.get("tasks", []):
        for artifact in task.get("artifacts", []):
            if artifact.get("id") == artifact_id:
                return dict(artifact)
    return None


def link_recorded_artifact(
    work_root: str | Path,
    session_id: str,
    artifact_id: str,
    workspace_path: str,
) -> dict[str, Any] | None:
    """Attach a downloaded workspace file to its persisted artifact record."""
    normalized_path = _normalize_workspace_path(workspace_path)
    files_dir = _files_dir(work_root, session_id)
    if _safe_artifact_file(files_dir, normalized_path) is None:
        raise ValueError("Artifact workspace path does not point to a file")

    manifest_path = _manifest_path(work_root, session_id)
    manifest = _load_manifest(manifest_path)
    for task in manifest.get("tasks", []):
        for artifact in task.get("artifacts", []):
            if artifact.get("id") == artifact_id:
                artifact["workspace_path"] = normalized_path
                _write_manifest(manifest_path, manifest)
                return dict(artifact)
    return None


def _content_preview(path: Path, mime_type: str) -> tuple[str, str | None]:
    if mime_type.startswith("image/"):
        return "image", None
    if path.suffix.lower() in {".npy", ".npz"}:
        try:
            import numpy as np

            loaded = np.load(path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    summary = "npz archive\n" + "\n".join(
                        f"{name}: shape={loaded[name].shape}, dtype={loaded[name].dtype}"
                        for name in loaded.files
                    )
                finally:
                    loaded.close()
                return "text", summary[:_TEXT_PREVIEW_LIMIT]
            return "text", f"npy array: shape={loaded.shape}, dtype={loaded.dtype}"
        except (OSError, ValueError, ImportError) as exc:
            return "binary", f"NumPy preview unavailable: {exc}"
    if mime_type.startswith("text/") or path.suffix.lower() in {
        ".json", ".yaml", ".yml", ".xml", ".csv", ".log", ".out", ".md", ".py"
    }:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "unavailable", None
        if len(content) > _TEXT_PREVIEW_LIMIT:
            content = content[:_TEXT_PREVIEW_LIMIT] + "\n…"
        return "text", content
    return "binary", None


def list_recorded_artifacts(work_root: str | Path, session_id: str) -> list[dict[str, Any]]:
    manifest = _load_manifest(_manifest_path(work_root, session_id))
    files_dir = _files_dir(work_root, session_id)
    tasks: list[dict[str, Any]] = []

    for task in reversed(manifest.get("tasks", [])):
        enriched_artifacts: list[dict[str, Any]] = []
        for artifact in task.get("artifacts", []):
            item = dict(artifact)
            item["workspace_path"] = _normalize_workspace_path(item.get("workspace_path"))
            path = _safe_artifact_file(files_dir, item.get("workspace_path"))
            item["exists"] = path is not None
            item["can_stage"] = (
                path is None
                and urlparse(str(item.get("source_path") or "")).scheme.lower() in _STAGEABLE_SCHEMES
            )
            item["file_name"] = path.name if path else None
            item["size"] = path.stat().st_size if path else None
            mime_type = mimetypes.guess_type(path.name)[0] if path else None
            item["mime_type"] = mime_type or "application/octet-stream"
            if path:
                preview_kind, content_preview = _content_preview(path, item["mime_type"])
            else:
                preview_kind, content_preview = "unavailable", None
            item["preview_kind"] = preview_kind
            item["content_preview"] = content_preview
            enriched_artifacts.append(item)

        enriched_task = dict(task)
        enriched_task["artifacts"] = enriched_artifacts
        tasks.append(enriched_task)
    return tasks
