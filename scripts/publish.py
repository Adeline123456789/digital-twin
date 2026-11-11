"""Publish explicit reviewed paths; reject common secrets/raw media, preserve failed-push commits."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def git(*args): return subprocess.run(['git',*args],cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--message',required=True);p.add_argument('paths',nargs='+');a=p.parse_args()
    files=[]
    for value in a.paths:
        path=(ROOT/value).resolve()
        if not path.is_relative_to(ROOT) or '.git' in path.parts or '.local' in path.parts: raise ValueError('Private/outside path')
        files.extend([path] if path.is_file() else [f for f in path.rglob('*') if f.is_file() and '__pycache__' not in f.parts])
    for f in files:
        if f.suffix not in ('.py','.md','.json','.jsonl','') and f.name!='.gitignore': raise ValueError('Unsupported public artifact: '+f.name)
        if f.stat().st_size>2_000_000: raise ValueError('Use dataset reference for large artifacts')
        text=f.read_text()
        # Heuristic check supplements manual review; never a guarantee of secret absence.
        if re.search(r'(?i)(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----|(?:api_key|password|access_token)\s*[=:]\s*[\"\x27][^\"\x27]{8,})',text):
            raise ValueError('Potential secret in '+str(f.relative_to(ROOT)))
    paths=[str(f.relative_to(ROOT)) for f in files]
    git('add','--',*paths)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if set(staged)-set(paths): raise ValueError('Unrelated staged files; refusing to commit')
    git('commit','-m',a.message)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=subprocess.run(['git','push','-u','origin','HEAD'],cwd=ROOT,text=True,capture_output=True)
    log=ROOT/'.local';log.mkdir(exist_ok=True)
    (log/('push-'+commit+'.txt')).write_text(result.stdout+result.stderr)
    print('commit',commit,'push_returncode',result.returncode)
    if result.returncode: print(result.stderr,file=sys.stderr)
    return result.returncode
if __name__=='__main__': sys.exit(main())
