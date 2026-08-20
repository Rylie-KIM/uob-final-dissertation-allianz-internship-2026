# RUN — commands only
## 2026-08-20: 실행 결국 안하고 model pkl 파일만 feature 추출해서, 아직 module 인지 못하는 문제는 해결 못함 
PowerShell, from repo root. Fill in `<...>`. Explanations: `RUNBOOK.md`.

```powershell
# ─── 0. paths ─────────────────────────────────────────────────────────────────
.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'src');import config;print(config.repo('v1'));print(config.path('model','v1'));print(config.path('preprocessor','v1'))"
```

```powershell
# ─── 1. fttl.pth  (module_name) ───────────────────────────────────────────────
src\envs\v1\.venv\python.exe -c "import sysconfig,pathlib,sys;r=sys.argv[1];d=[r]+[str(p) for p in pathlib.Path(r).iterdir() if p.is_dir() and list(p.glob('*.py'))];f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text('\n'.join(d));print(f);print(*d,sep='\n')" "<repo path from 0>"

$env:PYTHONPATH = ""
src\envs\v1\.venv\python.exe -c "import module_name; print('OK', module_name.__file__)"
```

```powershell
# ─── 2. registry v1  (from the pkl) ───────────────────────────────────────────
dir "<v1-repo>\outputs"

src\envs\v1\.venv\python.exe features\extract_features_v1.py --model "<v1-repo>\.....pkl" --dry-run

src\envs\v1\.venv\python.exe features\extract_features_v1.py --model "<v1-repo>\.....pkl"

.venv\Scripts\python.exe -c "import json,pathlib;[print(f.name,len(json.loads(f.read_text())['model_features']),'model /',len(json.loads(f.read_text())['raw_features']),'raw') for f in sorted(pathlib.Path('features/registry').glob('*.json'))]"
```

---

## if it fails

```powershell
# module missing -> append its parent to fttl.pth
src\envs\v1\.venv\python.exe -c "import sysconfig,pathlib,sys;f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';f.write_text(f.read_text().rstrip()+'\n'+sys.argv[1]+'\n');print(f.read_text())" "<parent dir>"

# what is on sys.path / is the .pth BOM-corrupted
src\envs\v1\.venv\python.exe -c "import sysconfig,pathlib,sys;f=pathlib.Path(sysconfig.get_paths()['purelib'])/'fttl.pth';print(f,f.exists());print(repr(f.read_bytes()[:120]));print(*[p for p in sys.path if 'model_repos' in p],sep='\n')"

# env stack
src\envs\v1\.venv\python.exe -c "import sys,numpy,joblib,sklearn,xgboost;print(sys.version.split()[0],'numpy',numpy.__version__,'joblib',joblib.__version__,'sklearn',sklearn.__version__,'xgboost',xgboost.__version__)"

# file header
src\envs\v1\.venv\python.exe -c "import sys,re;d=open(sys.argv[1],'rb').read();print('size',len(d));print('head',repr(d[:16]));print('modules',[m.decode('ascii','replace') for m in sorted(set(re.findall(rb'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+',d[:6000])))][:12])" "<pkl>"
```

fallback registry, no pkl:

```powershell
.venv\Scripts\python.exe features\extract_features_v1.py --from-matrix src\data\real\inputs\features_v1_train.parquet
```
