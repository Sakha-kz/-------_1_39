"""Fetch the exact assigned notebook at a pinned Git revision in Colab."""
import hashlib,json,subprocess
from pathlib import Path
import urllib.request

def fetch(folder='upstream/parsgpt'):
    folder=Path(folder)
    if not folder.exists(): subprocess.run(['git','clone','--depth','1','https://github.com/hooshvare/parsgpt.git',str(folder)],check=True)
    sha=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    notebook=folder/'notebooks/Persian_Poetry_FineTuning.ipynb'
    if not notebook.exists(): raise FileNotFoundError('Assigned notebook is absent in this revision; do not silently replace it.')
    obj=json.loads(notebook.read_text(encoding='utf-8'))
    record=dict(repository='https://github.com/hooshvare/parsgpt',revision=sha,path=str(notebook),sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),cells=len(obj['cells']))
    Path('poetry').mkdir(exist_ok=True); Path('poetry/original_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    # Code cell inventory supports an explicit comparison instead of claiming an unseen notebook was reproduced.
    code=['# Cell '+str(i)+'\n'+''.join(c['source']) for i,c in enumerate(obj['cells']) if c['cell_type']=='code']
    Path('poetry/original_code.py.txt').write_text('\n\n'.join(code),encoding='utf-8')
    print(json.dumps(record,indent=2)); return str(notebook)

if __name__=='__main__': fetch()
