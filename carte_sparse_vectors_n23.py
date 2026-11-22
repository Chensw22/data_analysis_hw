"""Extract just the needed input rows from the pinned official English fastText binary."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/nov23_carte"
URL="https://huggingface.co/facebook/fasttext-en-vectors/resolve/a80392390daaee1a91000da45c376d512e1dc555/model.bin"
SIZE=7237176312


def fetch(start,end):
    key=OUT/f"bytes_{start}_{end}"
    if key.exists() and key.stat().st_size==end-start+1:
        return key.read_bytes()
    for retry in range(4):
        try:
            req=urllib.request.Request(f"{URL}?interval={start}_{end}",headers={"Range":f"bytes={start}-{end}"})
            with urllib.request.urlopen(req,timeout=60) as res:
                assert res.status==206 and res.headers["Content-Range"]==f"bytes {start}-{end}/{SIZE}"
                body=res.read(end-start+2)
            assert len(body)==end-start+1
            key.write_bytes(body)
            return body
        except Exception:
            if retry==3:raise
            time.sleep(1+retry)


def compact(path,args,words,vectors):
    dim=vectors.shape[1]
    header=bytearray(args)
    struct.pack_into("<i",header,8,dim)
    struct.pack_into("<iii",header,40,0,0,0)
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<iiiqq",len(words),len(words),0,len(words),-1))
        for w in words:f.write(w.encode()+b"\0"+struct.pack("<qb",1,0))
        for m in (vectors,np.zeros_like(vectors)):
            f.write(struct.pack("<?qq",False,len(words),dim)); f.write(m.astype("<f4").tobytes())


def main():
    sys.path.insert(0,str(OUT/"packages"))
    import fasttext
    # The publisher exposes uncompressed matrices. Their sizes predict the
    # dictionary boundary, which we check by decoding every record and shape.
    offset=SIZE-(4_000_000+2_000_000)*300*4-34
    header=fetch(0,offset+16)
    assert struct.unpack_from("<ii",header)==(793712314,12)
    nwords=struct.unpack_from("<i",header,68)[0]
    dim=struct.unpack_from("<i",header,8)[0]
    assert (nwords,dim)==(2_000_000,300)
    assert struct.unpack_from("<?qq",header,offset)==(False,4000000,dim)
    cols=["HomePlanet","CryoSleep","Destination","Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Deck","CabinNumber","Side"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["HomePlanet","CryoSleep","Destination","VIP","Cabin"])
    cabin=raw.pop("Cabin").str.split("/",expand=True);raw["Deck"],raw["Side"]=cabin[0],cabin[2]
    names=set(cols)|{str(s).lower() for col in raw for s in raw[col].dropna().unique()}
    words=sorted({s for name in names for s in name.split()}|{"</s>","bread"})
    # Give native fastText the full original dictionary and hash parameters in
    # one dimension. This obtains exact C++ dictionary/OOV indices without
    # materialising any of the 7.2GB input/output embedding matrices.
    temp=OUT/"lookup_1d.bin"
    hd=bytearray(header[:offset]);struct.pack_into("<i",hd,8,1)
    with temp.open("wb") as f:
        f.write(hd)
        for nr in (4_000_000,2_000_000):
            f.write(struct.pack("<?qq",False,nr,1));f.seek(nr*4-1,1);f.write(b"\0")
    native=fasttext.load_model(str(temp))
    ids={w:native.get_subwords(w)[1] for w in words}
    # Official model has fixed length five character ngrams. Some short OOV
    # strings yield no feature rows and correctly map to the zero vector.
    ids_unique=sorted({int(j) for ix in ids.values() for j in ix})
    groups=[]
    for i in ids_unique:
        if not groups or i-groups[-1][-1]>128:groups.append([])
        groups[-1].append(i)
    def download(group):
        start=offset+17+group[0]*1200
        body=fetch(start,offset+17+(group[-1]+1)*1200-1)
        data=np.frombuffer(body,dtype="<f4").reshape(-1,300)
        return {k:data[k-group[0]].copy() for k in group}
    matrix={}
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        for a in executor.map(download,groups):matrix.update(a)
    vectors=[]
    for w in words:
        value=np.zeros(300,np.float32)
        for k in ids[w]:value+=matrix[int(k)]
        if len(ids[w]):value*=np.float32(1./len(ids[w]))
        vectors.append(value)
    v=np.stack(vectors)
    # Independent native one-coordinate forward using fetched official values,
    # including native OOV handling, verifies the manually composed input rows.
    with temp.open("r+b") as f:
        for k,value in matrix.items():
            f.seek(offset+17+4*k);f.write(value[:1].tobytes())
    native=fasttext.load_model(str(temp))
    np.testing.assert_allclose([native.get_word_vector(w)[0] for w in words],v[:,0],rtol=2e-6,atol=1e-8)
    temp.unlink()
    compact(OUT/"manifest_words.bin",header[:64],words,v)
    subset=fasttext.load_model(str(OUT/"manifest_words.bin"))
    for name in names:
        parts=[v[words.index(w)] for w in name.split()]
        nonzero=[p/np.linalg.norm(p) for p in parts if np.linalg.norm(p)>0]
        expected=np.mean(nonzero,axis=0) if nonzero else np.zeros(300)
        np.testing.assert_allclose(subset.get_sentence_vector(name),expected,atol=2e-6)
    report=ROOT/"reports/nov23_carte";report.mkdir(exist_ok=True)
    (report/"words.json").write_text(json.dumps(dict(model=URL,source_bytes=SIZE,distinct_names=len(names),
        words=words,unique_rows=len(matrix),data_bytes=sum(p.stat().st_size for p in OUT.glob("bytes_*")),
        native_coordinate_verified=len(words),compact_sha256=hashlib.sha256((OUT/"manifest_words.bin").read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":main()
