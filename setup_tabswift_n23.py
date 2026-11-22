"""Fetch only the pinned upstream TabSwift inference package and official checkpoint."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile

import requests

OUT=Path(__file__).resolve().parent/"artifacts/nov23_tabswift"
SOURCE="8edf8f0b4225bc03e1f5db011912619cd92b798d"
WEIGHTS="b829456edb7c41ad93a2851a8df245db362e1c83"
DIGEST="16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363"


def source():
    response=requests.get(f"https://codeload.github.com/LAMDA-Tabular/TabSwift/tar.gz/{SOURCE}",timeout=80)
    response.raise_for_status()
    base=f"TabSwift-{SOURCE}/TALENT/model/lib/"
    with tarfile.open(fileobj=io.BytesIO(response.content),mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(base+"tabswift/") or not member.name.endswith(".py"):
                continue
            rel=Path(member.name.removeprefix(base))
            if ".." in rel.parts:
                raise ValueError(rel)
            dest=OUT/SOURCE/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.extractfile(member).read())
    assert (OUT/SOURCE/"tabswift/classifier.py").is_file()
    print("official inference files",len(list((OUT/SOURCE).rglob("*.py"))))


def weights():
    path=OUT/"swift.ckpt"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==DIGEST:
        return
    link=f"https://huggingface.co/LAMDA-Tabular/TabSwift/resolve/{WEIGHTS}/swift.ckpt"
    response=requests.get(link,timeout=90)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest()!=DIGEST:
        raise ValueError("Checkpoint digest mismatch")
    tmp=path.with_suffix(".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    print("verified checkpoint bytes",path.stat().st_size)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("asset",choices=["source","weights"])
    OUT.mkdir(exist_ok=True)
    (source if parser.parse_args().asset=="source" else weights)()
