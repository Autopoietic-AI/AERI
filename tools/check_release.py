"""Check the code-only release without importing ML runtimes or loading weights."""
from pathlib import Path
import ast,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.weights','.bin','.h5','.hdf5','.pkl','.pickle','.onnx','.engine','.tflite','.pb','.pyc','.pyo'}
required=['aeri.py','stagefusion/model.py','stagefusion/model_a5_tig_stagefusion.py','stagefusion/model_a5_tig_stagefusion_ablation.py','stagefusion/main_cat101_a5_tig_stagefusion.py','stagefusion/eval.py','cat101_native/7_train_m8_igcoattn_v1_2_dualgate_framecls.py','cat101_native/build_cat101_sam2_masks_full_packed.py','sics_cross/build_sics155_100_sam2_masks_full_packed.py','LICENSES/MS-TCN2-MIT.txt','docs/source_snapshot_manifest.json']
for rel in required:assert (ROOT/rel).is_file(),rel
paths=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
for p in paths:
 rel=p.relative_to(ROOT)
 assert p.suffix.lower() not in FORBIDDEN,str(rel)
 assert 'weights' not in rel.parts and '__pycache__' not in rel.parts,str(rel)
 assert p.stat().st_size<10_000_000,str(rel)
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(rel))
 raw=p.read_bytes()
 assert not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',raw),str(rel)
m=json.loads((ROOT/'docs/source_snapshot_manifest.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rec in m['source_files']:assert sha(ROOT/rec['released_path'])==rec['sha256'],rec['released_path']
assert sha(ROOT/m['upstream_decoder']['released_path'])==m['upstream_decoder']['sha256']
for rel,cls in [('stagefusion/model.py','MS_TCN2'),('stagefusion/model_a5_tig_stagefusion.py','A5TIGDFusionModel')]:
 tree=ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
 assert any(isinstance(n,ast.ClassDef) and n.name==cls for n in tree.body),(rel,cls)
print(json.dumps({'project':'AERI','files_checked':len(paths),'python_files_checked':sum(p.suffix=='.py' for p in paths),'original_python_files_preserved':m['original_python_files'],'weights_found':0,'secrets_found':0,'full_model_definitions_present':True,'runtime_training_tested':False},ensure_ascii=False,indent=2))
