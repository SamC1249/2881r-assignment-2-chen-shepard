"""Shared utilities: config + content hashing, JSONL I/O with the resume discipline
from PIPELINE.md (one write+flush per record; tolerate exactly one truncated FINAL line;
error on interior corruption; dedupe last-wins; config hash in filenames)."""

import hashlib
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "configs/run.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def repo_path(cfg: dict, *parts) -> Path:
    p = Path(cfg["_repo_root"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(*args, digits: int = 8) -> str:
    """Deterministic short hash of the args' JSON repr."""
    s = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:digits]


def stable_seed(*args) -> int:
    """Deterministic 31-bit seed (vLLM wants an int). NEVER share seeds across requests."""
    return int(stable_hash(*args, digits=8), 16) % (2**31)


def file_hash(path: Path, digits: int = 8) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:digits]


def gen_hash(cfg: dict) -> str:
    """Covers everything that determines the generation distribution."""
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = "none"
    tpl = file_hash(repo_path(cfg, cfg["paths"]["prompts"], "templates.yaml"))
    return stable_hash(cfg["target_model"], cfg["generation"], tpl, vllm_version)


def rubric_hash(cfg: dict, mode: str) -> str:
    rubric_file = {
        "sycophancy": "judge_sycophancy.md",
        "correctness": "judge_correctness.md",
        "agreement": "judge_agreement.md",
    }[mode]
    rub = file_hash(repo_path(cfg, cfg["paths"]["prompts"], rubric_file))
    return stable_hash(cfg["judge"], rub, mode)


# ---------------------------------------------------------------- JSONL I/O

def read_jsonl(path: Path, key: str | None = None, repair: bool = True) -> list | dict:
    """Read JSONL. Tolerates exactly one truncated FINAL line (crash mid-write) by
    truncating the file to the last newline; raises on any interior malformed line.
    If `key` is given, returns a dict deduped last-wins on that key."""
    path = Path(path)
    if not path.exists():
        return {} if key else []
    raw = path.read_bytes()
    if repair and raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        print(f"[common] WARNING: truncated final line in {path.name}; dropping {len(raw) - cut} bytes")
        with open(path, "r+b") as f:
            f.truncate(cut)
        raw = raw[:cut]
    rows = []
    for i, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{path}:{i + 1} interior corruption — refusing to continue") from e
    if key is None:
        return rows
    out = {}
    for r in rows:  # last-wins
        out[r[key]] = r
    return out


class JsonlWriter:
    """Append-only writer: one write+flush+fsync per record. One writer per file, ever."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pending_ids(all_ids: list, done: dict, extra_ok=None) -> list:
    """IDs not yet present in `done` (a dict keyed by id). Optional predicate marks a done
    row as invalid (e.g. parse failure) so it gets redone."""
    ok = set()
    for k, row in done.items():
        if extra_ok is None or extra_ok(row):
            ok.add(k)
    return [i for i in all_ids if i not in ok]
