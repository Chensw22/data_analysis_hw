"""Fetch pinned public inputs for original_tabstar_probe_n26.py. No competition data is accessed.

Use the project .venv-tabpfn Python environment (existing Torch runtime), then install
the original implementation into the experiment-specific package directory with

    uv pip install --python .venv-tabpfn/bin/python --no-deps --target artifacts/nov26_original_tabstar/package tabstar==1.2.2 peft==0.17.1 accelerate==1.10.1

Run this file with Python before executing the probe. The default Hugging Face source
can be overridden using --hub https://hf-mirror.com. Existing verified files are reused.
"""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from shutil import copyfileobj

ROOT=Path(__file__).resolve().parent/'artifacts/nov26_original_tabstar'
TAB='73560fb23d34307f4b6fa849f38f8299a8980a4a'
E5='ffb93f3bd4047442299a41ebb6fa998a38507c52'
SOURCE='4d14a660ae32b978b9bcde8c9f39e119513cf030'


def fetch(url,filename,digest):
    path=ROOT/filename
    def verify(p):
        if not p.is_file():
            return False
        with p.open('rb') as f:
            return hashlib.file_digest(f,'sha256').hexdigest()==digest
    if verify(path):
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with urlopen(url,timeout=150) as response,partial.open('wb') as f:
        copyfileobj(response,f)
    if not verify(partial):
        raise RuntimeError(f'Checksum does not match for {filename}')
    partial.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--hub',default='https://huggingface.co')
    hub=parser.parse_args().hub.rstrip('/')
    tab=f'{hub}/alana89/TabSTAR/resolve/{TAB}/'
    e5=f'{hub}/intfloat/e5-small-v2/resolve/{E5}/'
    catalog='src/tabstar/datasets/all_datasets.py'
    # All checksums were calculated on the actual verification/training inputs.
    for url,filename,digest in [
        (tab+'pretrain_args.json','manifest.json','fd0a7e18c1ef21e6f7cdf5d1c55930a3420094af94d37bad433fac4a5c6c934b'),
        (tab+'model.safetensors','tabstar.safetensors','455272659647d442117ea31ca274e9b683d47295738957229a4ffb632f6fa465'),
        (tab+'config.json','config.json','1e992dc8c7aafda0e90b2410b4b136ded8cd9157442e3b5846b794aafa0f1264'),
        (e5+'model.safetensors','e5/model.safetensors','45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1'),
        (e5+'config.json','e5/config.json','5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f'),
        (e5+'tokenizer.json','e5/tokenizer.json','d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'),
        (e5+'special_tokens_map.json','e5/special_tokens_map.json','b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'),
        (e5+'vocab.txt','e5/vocab.txt','07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
        (e5+'tokenizer_config.json','e5/tokenizer_config.json','ae83fa6ca0333117ff12606020af925d648667ef70d92ff7f27d781ba0ca4544'),
        (f'https://raw.githubusercontent.com/alanarazi7/TabSTAR/{SOURCE}/{catalog}',f'TabSTAR-{SOURCE}/{catalog}','d6b1e4b48d2d4a9afd2aa721ea88b873c1f213a0fbfb1660c4a0b7fc8046a149')
    ]:
        fetch(url,filename,digest)
