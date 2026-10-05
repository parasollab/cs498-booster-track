"""Check retained files, documented removals/patches, and absence of extra source."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]; vendor=root/"vendor/booster_mjlab"
    original=json.loads((root/"vendor-manifest.json").read_text())
    classified=json.loads((root/"vendor-classification.json").read_text())
    assert set(original["files"])==set(classified["files"])
    expected=set()
    for name,item in classified["files"].items():
        assert original["files"][name]==item["original_sha256"]
        file=vendor/name
        if item["retained"]:
            expected.add(name)
            assert file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==item["current_sha256"], name
        else: assert not file.exists(), name
    actual={str(p.relative_to(vendor)) for p in vendor.rglob("*") if p.is_file()
        and not any(x in p.relative_to(vendor).parts for x in (".venv","__pycache__",".cache","dist","build"))}
    assert actual==expected, (actual-expected,expected-actual)
    assert not (vendor/".git").exists()
    print(f"PASS: {len(expected)} retained files; {len(original['files'])-len(expected)} documented removals; all hashes; no extra source/nested Git")


if __name__=="__main__": main()
