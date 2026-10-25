#!/usr/bin/env python3
"""Verify actual GitHub asset bytes/digests before installing an Index app catalog.

GitHub's server-computed SHA-256 is checked for every downloadable part. Public
URL probes are optional while the release remains a draft, required for shipping.
This does not attest to NPU execution or translation quality.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import urllib.request
from build_index_download_catalog import identity_key

REPO = "indexix/bookrack-cdn"


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalogs", nargs="+", type=pathlib.Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--report", type=pathlib.Path, required=True)
    p.add_argument("--require-public", action="store_true")
    a = p.parse_args()
    release_pages = gh_json("api", "--paginate", "--slurp", f"repos/{REPO}/releases?per_page=100")
    release = next(release for page in release_pages for release in page if release["tag_name"] == a.tag)
    if a.require_public:
        assert not release["draft"], "Release is still a draft"
    releases = {item["tag_name"]: item for page in release_pages for item in page}
    assets_by_tag = {}
    entries, checks, seen = [], [], set()
    for path in a.catalogs:
        catalog = json.loads(path.read_text())
        assert catalog["schema"] == 1 and catalog["models"]
        for entry in catalog["models"]:
            raw = entry["manifest"]
            assert hashlib.sha256(raw.encode()).hexdigest() == entry["manifestSha256"]
            meta = json.loads(raw)
            identity = identity_key(meta)
            assert identity[0] == "index2" and identity not in seen
            assert not meta.get("testOnly") and meta["schema"] == 1
            seen.add(identity)
            assert set(meta["files"]) == set(entry["downloads"])
            for name, parts in entry["downloads"].items():
                assert sum(part["size"] for part in parts) == meta["files"][name]["size"]
                for part in parts:
                    prefix = f"https://github.com/{REPO}/releases/download/"
                    assert part["url"].startswith(prefix)
                    asset_tag, asset_name = part["url"][len(prefix):].split("/", 1)
                    assert "/" not in asset_name and "?" not in asset_name
                    asset_release = releases[asset_tag]
                    if a.require_public:
                        assert not asset_release["draft"], asset_tag
                    if asset_tag not in assets_by_tag:
                        pages = gh_json("api", "--paginate", "--slurp",
                            f"repos/{REPO}/releases/{asset_release['id']}/assets?per_page=100")
                        assets_by_tag[asset_tag] = {asset["name"]: asset for page in pages for asset in page}
                    asset = assets_by_tag[asset_tag][asset_name]
                    assert asset["state"] == "uploaded"
                    assert asset["size"] == part["size"]
                    assert asset["digest"] == "sha256:" + part["sha256"], asset_name
                    if a.require_public:
                        request = urllib.request.Request(
                            part["url"], headers={"Range": "bytes=0-0"}
                        )
                        with urllib.request.urlopen(request, timeout=60) as response:
                            assert response.status in (200, 206) and response.read(1)
                    checks.append({"tag": asset_tag, "name": asset_name, "size": asset["size"], "sha256": part["sha256"]})
            entries.append(entry)
    entries.sort(key=lambda entry: json.loads(entry["manifest"])["runtime"] + "/" + json.loads(entry["manifest"])["target"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"schema": 1, "models": entries}, ensure_ascii=False, indent=2) + "\n")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"release": release["html_url"], "publicUrlsChecked": a.require_public, "bundles": len(entries), "assets": checks}, indent=2) + "\n")
    print(f"Verified {len(entries)} Index bundles / {len(checks)} downloadable parts")


if __name__ == "__main__":
    main()
