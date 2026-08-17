import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "treebanks" / "registry.json"
MANIFEST_PATH = REPO_ROOT / "content_manifest.json"
LESSON_PREFIXES = ("lessons/en/", "lessons/fa/")


def _glob_suffix(file_glob: str) -> str:
    if file_glob and file_glob.startswith("*."):
        return file_glob[1:].lower()
    return ".xml"


def _files_under(prefix: str, suffix: str) -> list[str]:
    folder = REPO_ROOT / prefix
    if not folder.is_dir():
        return []
    return sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.name.lower().endswith(suffix)
    )


def build_manifest() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    treebanks = {}
    for corpus in registry.get("corpora", []):
        prefix = corpus.get("path", "")
        if not prefix:
            continue
        treebanks[prefix] = _files_under(prefix, _glob_suffix(corpus.get("file_glob", "*.xml")))

    lessons = {prefix: _files_under(prefix, ".md") for prefix in LESSON_PREFIXES}
    return {"treebanks": treebanks, "lessons": lessons}


if __name__ == "__main__":
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for section, folders in manifest.items():
        for prefix, files in folders.items():
            print(f"{section}: {prefix} -> {len(files)} files")
    print(f"wrote {MANIFEST_PATH}")
