#!/usr/bin/env python3
"""Split each four-layer Index export into two CPU partitions for a bounded-memory hybrid.
The emitted calibration row is sufficient for the CPU FLOAT32 weight-only execution
plan; it is NOT representative activation calibration for an NPU decoder body.
Use fresh representative calibration for the separately quantized NPU heads.
"""
import argparse,pathlib,json,os,copy,re,numpy as np,onnx,onnxruntime as ort
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=pathlib.Path)
p.add_argument('calibration',type=pathlib.Path)
p.add_argument('output',type=pathlib.Path)
p.add_argument('--head-parts',type=pathlib.Path,required=True)
p.add_argument('--runtime-root',type=pathlib.Path,default=pathlib.Path('/work'))
a=p.parse_args()
from onnx import helper as H,TensorProto as T
root=a.output;root.mkdir(parents=True,exist_ok=True)
meta={'decoder':json.loads((a.source/'decoder.json').read_text())};parts=[]
assert len(meta['decoder']['partitions'])==7 and meta['decoder']['hiddenSize']==2048, 'Expected the pinned four-layer Index export'
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
for i in range(6):
 src=a.source/f'part-{i:02d}/decoder.onnx';full=onnx.load(src,load_external_data=False);part=meta['decoder']['partitions'][i];last_weight=f'model.language_model.layers.{i*4+1}.mlp.down_proj.weight';linear=next(n.output[0] for n in full.graph.node if n.op_type=='MatMul' and last_weight in n.input);boundary=next(n.output[0] for n in full.graph.node if n.op_type=='Add' and linear in n.input)
 cal=a.calibration/f'part-{i:02d}/calibration/inputs-onnx.txt';line=cal.read_text().splitlines()[0];feed={}
 shape={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in full.graph.input}
 for entry in line.split():
  name,path=entry.split(':=');feed[name]=np.fromfile(path.replace(str(a.runtime_root).rstrip('/')+'/',str(pathlib.Path.cwd())+'/'),dtype='<f4').reshape(shape[name])
 # Run one float fixture to provide the second half's exact hidden input.
 probe=copy.deepcopy(full);probe.graph.output.append(H.make_tensor_value_info(boundary,T.FLOAT,[1,1,2048]));tmp=root/f'part-{i:02d}-probe';tmp.mkdir(exist_ok=True);os.link(src.parent/'weights.bin',tmp/'weights.bin') if not (tmp/'weights.bin').exists() else None;onnx.save(probe,tmp/'probe.onnx');session=ort.InferenceSession(str(tmp/'probe.onnx'),sess_options=opts);middle=session.run([boundary],feed)[0];del session
 for half in range(2):
  name=f'part-{i*2+half:02d}';folder=root/name;folder.mkdir(exist_ok=True);m=copy.deepcopy(full);states=part['states'][:4] if half==0 else part['states'][4:];out=boundary if half==0 else 'hidden';needed={out}|{s['output'] for s in states};kept=[]
  for n in reversed(m.graph.node):
   if n.output[0] in needed and not (half==1 and n.output[0]==boundary):kept.append(n);needed.update(n.input)
  kept.reverse()
  for n in kept:
   for j,value in enumerate(n.input):
    if half==1 and value==boundary:n.input[j]='embedding'
  if half==0:kept.append(H.make_node('Identity',[boundary],['hidden'],name='hidden'))
  del m.graph.node[:];m.graph.node.extend(kept);initializers=[t for t in m.graph.initializer if t.name in needed];del m.graph.initializer[:];m.graph.initializer.extend(initializers)
  ins=[v for v in m.graph.input if v.name in needed and (half==0 or v.name!='embedding')]
  if half==1:ins.insert(0,H.make_tensor_value_info('embedding',T.FLOAT,[1,1,2048]))
  del m.graph.input[:];m.graph.input.extend(ins);outs=[H.make_tensor_value_info('hidden',T.FLOAT,[1,1,2048])]+[v for v in full.graph.output if v.name in {s['output'] for s in states}];del m.graph.output[:];m.graph.output.extend(outs)
  if not (folder/'weights.bin').exists():os.link(src.parent/'weights.bin',folder/'weights.bin')
  onnx.save(m,folder/'decoder.onnx');onnx.checker.check_model(str(folder/'decoder.onnx'))
  entries=[]
  for v in ins:
   data=middle if half==1 and v.name=='embedding' else feed[v.name];p=folder/(v.name+'.raw');data.tofile(p);entries.append(v.name+':='+str(a.runtime_root/p.resolve().relative_to(pathlib.Path.cwd())))
  (folder/'inputs.txt').write_text(' '.join(entries)+'\n');parts.append(dict(modelFile=name+'.dlc',output='hidden',states=states));print(name,flush=True)
meta['decoder']['partitions']=parts+json.loads(a.head_parts.read_text());(root/'decoder.json').write_text(json.dumps(meta['decoder'],indent=2)+'\n')
