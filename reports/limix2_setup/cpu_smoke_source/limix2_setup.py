"""Fetch pinned official LimiX-2 source/weights; never execute downloaded scripts."""
import argparse, hashlib, json, os, shutil, tarfile, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'artifacts/limix2_setup'
REPORT=ROOT/'reports/limix2_setup'
CODE='db592af4a2f6f97ae20d9bf5fe56facdc00d7441'
REVISION='de07b679e74a41b50b9de18251a8fa245e537440'
WEIGHT_SHA='ed01f7a0a18d25451bc9d1058b479ba6b68048aa1ebde06156c8fcfaf8b8fe81'
WEIGHT_SIZE=1625774719

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')

def download(url,path,expected_sha=None,expected_size=None):
 if path.exists():
  if expected_sha and sha(path)!=expected_sha:raise RuntimeError('Existing hash differs: '+str(path))
  if expected_size and path.stat().st_size!=expected_size:raise RuntimeError('Existing size differs.')
  return
 path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.partial')
 n=part.stat().st_size if part.exists() and expected_size else 0;last=n
 request=urllib.request.Request(url,headers={'Range':f'bytes={n}-'} if n else {})
 response=urllib.request.urlopen(request,timeout=120)
 if n and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {n}-')):
  response.close();raise RuntimeError('Server did not honor exact resume range.')
 with response as r,part.open('ab' if n else 'wb') as f:
  while b:=r.read(4*1024*1024):
   n+=len(b);f.write(b)
   if n-last>=128*1024*1024:print(path.name,n,'bytes',flush=True);last=n
 if expected_size and n!=expected_size:raise RuntimeError('Downloaded size differs.')
 if expected_sha and sha(part)!=expected_sha:raise RuntimeError('Downloaded SHA differs.')
 part.rename(path)

def fetch():
 start=time.monotonic();free=shutil.disk_usage(ROOT).free
 if free<8*1024**3:raise RuntimeError('Need8GiBfree before source/weight/environment preparation.')
 WORK.mkdir(parents=True,exist_ok=True)
 archive=WORK/'official_source.tar.gz';source=WORK/'source'
 download('https://codeload.github.com/limix-ldm-ai/LimiX/tar.gz/'+CODE,archive)
 if not source.exists():
  source.mkdir()
  with tarfile.open(archive) as tf:
   for m in tf:
    pieces=Path(m.name).parts
    if len(pieces)<2:continue
    rel=Path(*pieces[1:]);target=source/rel
    if not target.resolve().is_relative_to(source.resolve()):raise RuntimeError('Unsafe archive path.')
    if m.isdir():target.mkdir(parents=True,exist_ok=True)
    elif m.isfile():
     target.parent.mkdir(parents=True,exist_ok=True)
     with tf.extractfile(m) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
    else:raise RuntimeError('Nonregular archive member rejected: '+m.name)
 weight=WORK/'weights/LimiX-2.ckpt';licensefile=WORK/'weights/LICENSE'
 base='https://huggingface.co/stable-ai/LimiX-2/resolve/'+REVISION+'/'
 download(base+'LICENSE',licensefile)
 download(base+'LimiX-2.ckpt',weight,WEIGHT_SHA,WEIGHT_SIZE)
 files={str(p.relative_to(ROOT)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
 result=dict(status='downloaded_pinned_verified_not_loaded',code_commit=CODE,hf_revision=REVISION,
  weight_path=str(weight.relative_to(ROOT)),weight_size=weight.stat().st_size,weight_sha256=sha(weight),
  source_archive_sha256=sha(archive),source_files=files,model_license_sha256=sha(licensefile),
  source_license_sha256=sha(source/'LICENSE.txt'),attribution='Built with StableAI LimiX',
  use='Three-person non-commercial education/Kaggle prediction; no hosted API or commercial competing foundation-model development.',
  source_url='https://github.com/limix-ldm-ai/LimiX/tree/'+CODE,
  model_url='https://huggingface.co/stable-ai/LimiX-2/tree/'+REVISION,
  no_model_execution=True,no_remote_scripts_executed=True,seconds=time.monotonic()-start,initial_disk_free_bytes=free)
 write(REPORT/'download.json',result);print('Pinned source/weights verified.',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--fetch',action='store_true',required=True);parser.parse_args();fetch()
