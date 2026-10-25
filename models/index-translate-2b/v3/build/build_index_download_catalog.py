#!/usr/bin/env python3
"""Create Index-only release assets and the pinned app catalog from VERIFIED bundles.

Each input directory must have manifest.json plus all its checked files. Readiness
is explicit in conversion.json: conversionVerified true and an explicit referenceVerified value, with
onDeviceVerified reported honestly. This command never uploads or invents models.
"""

import argparse, hashlib, json, pathlib, shutil, re

REPO = "indexix/bookrack-cdn"
REVISION = "9bc08445315723eb9b77d1b93505915eb03c3f01"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity_key(meta):
    bits = 0
    if meta["runtime"] == "snpe":
        bits = meta.get("activationBits") or int(re.search(r"W\d+A(8|16)\b", meta["quantization"])[1])
    return (meta["model"], meta["runtime"], meta["target"], bits)


def build(folders, assets, catalog, tag, reuse_catalog=None):
    assets.mkdir(parents=True, exist_ok=True)
    previous = json.loads(reuse_catalog.read_text())["models"] if reuse_catalog else []
    entries = list(previous)
    reusable = {}
    for entry in previous:
        old = json.loads(entry["manifest"])
        for name, info in old["files"].items():
            reusable[(name, info["size"], info["sha256"])] = entry["downloads"][name]
    seen = set()
    checksums = []
    for folder in folders:
        raw = (folder / "manifest.json").read_text()
        meta = json.loads(raw)
        assert (
            meta["model"] == "index2"
            and meta["schema"] == 1
            and not meta.get("testOnly")
        )
        key = identity_key(meta)
        identity = f"index2-{meta['runtime']}-{meta['target']}" + (f"-a{key[3]}" if key[3] else "")
        assert identity not in seen
        seen.add(identity)
        conversion = json.loads((folder / "conversion.json").read_text())
        assert (
            conversion["source"] == "IndexTeam/Index-Translate-2B"
            and conversion["revision"] == REVISION
        )
        assert conversion["conversionVerified"] is True
        assert isinstance(conversion["referenceVerified"], bool)
        assert (
            isinstance(conversion["validationScope"], str)
            and conversion["validationScope"]
        )
        assert (
            "onDeviceVerified" in conversion
            and (folder / "LICENSE.txt").stat().st_size > 1000
        )
        downloads = {}
        for name, info in meta["files"].items():
            assert pathlib.Path(name).name == name and ".." not in name
            file = folder / name
            assert (
                file.stat().st_size == info["size"] and digest(file) == info["sha256"]
            )
            reused = reusable.get((name, info["size"], info["sha256"]))
            if reused:
                downloads[name] = reused
                continue
            pieces = []
            with file.open("rb") as source:
                count = (info["size"] + 999_999_999) // 1_000_000_000
                for i in range(count):
                    asset = f"{identity}-{name}.part{i:02d}"
                    target = assets / asset
                    size = min(1_000_000_000, info["size"] - i * 1_000_000_000)
                    sha = hashlib.sha256()
                    with target.open("wb") as out:
                        remaining = size
                        while remaining:
                            data = source.read(min(4 * 1024 * 1024, remaining))
                            assert data
                            out.write(data)
                            sha.update(data)
                            remaining -= len(data)
                    value = sha.hexdigest()
                    checksums.append(f"{value}  {asset}")
                    pieces.append(
                        {
                            "url": f"https://github.com/{REPO}/releases/download/{tag}/{asset}",
                            "size": size,
                            "sha256": value,
                        }
                    )
            downloads[name] = pieces
        manifest_asset = assets / (identity + "-manifest.json")
        manifest_asset.write_text(raw)
        checksums.append(f"{digest(manifest_asset)}  {manifest_asset.name}")
        entries = [entry for entry in entries if identity_key(json.loads(entry["manifest"])) != key]
        entries.append(
            {
                "manifest": raw,
                "manifestSha256": hashlib.sha256(raw.encode()).hexdigest(),
                "downloads": downloads,
            }
        )
    entries.sort(key=lambda entry: json.loads(entry["manifest"])["runtime"] + "/" + json.loads(entry["manifest"])["target"])
    payload = {"schema": 1, "models": entries}
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(catalog, assets / "vendor-model-catalog.json")
    checksums.append(f"{digest(catalog)}  vendor-model-catalog.json")
    (assets / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(
        f"{len(entries)} bundles, {len(checksums)} assets; upload/verification still required"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", nargs="+", type=pathlib.Path)
    p.add_argument("--assets", required=True, type=pathlib.Path)
    p.add_argument("--catalog", required=True, type=pathlib.Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--reuse-catalog", type=pathlib.Path, help="Preserve other profiles and reuse identical checksummed release assets")
    a = p.parse_args()
    assert all(c.isalnum() or c in "-_." for c in a.tag)
    build(a.folders, a.assets, a.catalog, a.tag, a.reuse_catalog)
