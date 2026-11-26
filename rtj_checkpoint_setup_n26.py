"""Fetch the two public, pinned local RT-J inputs, verifying all model bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent/'artifacts/nov26_rtj_probe'
WEIGHTS = '67b06ac8483094fdf80e157683edb50e685f752b7e25706dd9f66c100df55be4'
RT_REV = '4ee4c583895304cb3529120bf86fc9e04711b4b6'
EMBED = 'sentence-transformers/all-MiniLM-L12-v2'
EMBED_REV = 'a50ef00143b4d5391434df20ae11632588ac25be'
EMBED_FILES = ['config.json','model.safetensors','config_sentence_transformers.json','modules.json',
               'sentence_bert_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json',
               'vocab.txt','1_Pooling/config.json']


def fetch(repo,revision,folder,names,host):
    cmd = ['curl','-fLsS','--retry','2','--connect-timeout','12','--max-time','600']
    api = host+'/api/models/'+repo+('/revision/'+revision if revision else '')+'?blobs=true'
    meta = json.loads(subprocess.check_output(cmd+[api]))
    root = OUT/folder
    files = [e for e in meta['siblings'] if e['rfilename'] in names]
    assert {e['rfilename'] for e in files}>=set(['model.safetensors','config.json'])
    def download(e):
        dest = root/e['rfilename']
        dest.parent.mkdir(parents=True,exist_ok=True)
        def verified():
            if not dest.is_file() or dest.stat().st_size!=e['size']:
                return False
            b=dest.read_bytes()
            if 'lfs' in e:
                digest=hashlib.sha256(b).hexdigest()
                if repo.endswith('/rt-j'):
                    assert e['lfs']['sha256']==WEIGHTS
                return digest==e['lfs']['sha256']
            return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['blobId']
        if not verified():
            temp=dest.with_suffix(dest.suffix+'.part')
            subprocess.run(cmd+[host+'/'+repo+'/resolve/'+meta['sha']+'/'+e['rfilename']+'?download=true&cache='+str(int(time.time())),
                                '-o',str(temp)],check=True)
            temp.replace(dest)
            assert verified()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(download,files))
    return dict(repo=repo,revision=meta['sha'],files=files)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='https://huggingface.co')
    host=parser.parse_args().host
    emb=fetch(EMBED,EMBED_REV,EMBED,EMBED_FILES,host)
    rt=fetch('stanford-star/rt-j',RT_REV,'weights_pinned',['config.json','model.safetensors'],host)
    (OUT/'sources.json').write_text(json.dumps(dict(rt=rt,embedder=emb),indent=2)+'\n')
