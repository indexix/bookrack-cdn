#!/usr/bin/env python3
"""Split Index's vocabulary projection before DLC conversion. Does not attest to NPU readiness."""
import argparse,pathlib,onnx,numpy as np,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=pathlib.Path)
p.add_argument('output',type=pathlib.Path)
p.add_argument('--shards',type=int,default=4)
a=p.parse_args()
assert 1 <= a.shards <= 16
src=a.source;out=a.output;out.mkdir(parents=True,exist_ok=True)
base=onnx.load(src,load_external_data=False)
vocab=next(v for v in base.graph.output if v.name=='logits').type.tensor_type.shape.dim[-1].dim_value
bounds=[i*vocab//a.shards for i in range(a.shards+1)];parts=[]
for shard,(start,end) in enumerate(zip(bounds,bounds[1:])):
 m=onnx.load(src,load_external_data=False);d=out/f'head-{shard:02d}';d.mkdir(exist_ok=True)
 with (d/'weights.bin').open('wb') as f:
  for t in m.graph.initializer:
   if not t.external_data:continue
   ext={e.key:e.value for e in t.external_data};matrix=np.memmap(src.parent/ext['location'],dtype='<f4',offset=int(ext['offset']),shape=tuple(t.dims));offset=f.tell()
   if t.name.endswith('embed_tokens.weight'):
    for r in range(0,t.dims[0],64):f.write(matrix[r:r+64,start:end].copy().tobytes())
    t.dims[-1]=end-start
   else:f.write(matrix.tobytes())
   for e in t.external_data:
    if e.key=='offset':e.value=str(offset)
    if e.key=='length':e.value=str(f.tell()-offset)
 for v in m.graph.output:
  if v.name=='logits':v.type.tensor_type.shape.dim[-1].dim_value=end-start
 del m.graph.value_info[:]
 onnx.save(m,d/'decoder.onnx');onnx.checker.check_model(str(d/'decoder.onnx'));print(d,flush=True)
 parts.append({'modelFile':f'head-{shard:02d}.dlc','output':'logits','states':[],'vocabOffset':start,'outputSize':end-start})
(out/'parts.json').write_text(json.dumps(parts,indent=2)+'\n')
