#!/usr/bin/env python3
"""Rewrite export_index_snpe.py RMS/L2 norms exactly using augmented LpNormalization.
Float parity and target quantized quality must be checked separately.
Usage: rewrite_index_snpe_norm.py source.onnx target.onnx
"""
import pathlib,sys,onnx,numpy as np,os
from onnx import helper as H,numpy_helper as N,TensorProto as T
source=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2]);target.parent.mkdir(parents=True,exist_ok=True)
m=onnx.load(source,load_external_data=False);m=onnx.shape_inference.infer_shapes(m)
shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
by={n.output[0]:n for n in m.graph.node};constants={t.name:N.to_array(t) for t in m.graph.initializer if not t.external_data};nodes=[];count=0
for n in m.graph.node:
 if n.op_type=='Div' and n.input[1] in by and by[n.input[1]].op_type=='Sqrt':
  x=n.input[0];root=by[n.input[1]];v=by[root.input[0]];dim=shapes[x][-1];shape=shapes[x];eps=None;scale=1.0
  if v.op_type=='ReduceMean':
   add=by[v.input[0]]
   if add.op_type=='Add':eps=float(constants[add.input[1]].item())*dim;scale=float(np.sqrt(dim))
  elif v.op_type=='Add' and v.input[0] in by and by[v.input[0]].op_type=='ReduceSum':eps=float(constants[v.input[1]].item())
  if eps is not None:
   prefix=n.output[0]+'_exact_l2';pad=prefix+'_pad';padded=prefix+'_padded';norm=prefix+'_norm';sliced=prefix+'_slice'
   m.graph.initializer.append(N.from_array(np.full(shape[:-1]+[1],np.sqrt(eps),dtype=np.float32),pad))
   nodes.extend([H.make_node('Concat',[x,pad],[padded],name=padded,axis=-1),H.make_node('LpNormalization',[padded],[norm],name=norm,axis=-1,p=2)])
   for key,val in [('starts',[0]),('ends',[dim]),('axes',[-1])]:m.graph.initializer.append(N.from_array(np.array(val,np.int64),prefix+'_'+key))
   nodes.append(H.make_node('Slice',[norm,prefix+'_starts',prefix+'_ends',prefix+'_axes'],[sliced],name=sliced))
   m.graph.initializer.append(N.from_array(np.array(scale,np.float32),prefix+'_scale'))
   nodes.append(H.make_node('Mul',[sliced,prefix+'_scale'],list(n.output),name=n.name));count+=1;continue
 nodes.append(n)
# Remove dead arithmetic so it cannot be quantized or run.
needed={v.name for v in m.graph.output};live=[]
for n in reversed(nodes):
 if any(o in needed for o in n.output):live.append(n);needed.update(n.input)
del m.graph.node[:];m.graph.node.extend(reversed(live));del m.graph.value_info[:]
initializers=[t for t in m.graph.initializer if t.name in needed];del m.graph.initializer[:];m.graph.initializer.extend(initializers)
for t in initializers:
 for e in t.external_data:
  if e.key=='location':
   path=target.parent/e.value
   if not path.exists():os.link(source.parent/e.value,path)
onnx.save(m,target);onnx.checker.check_model(str(target));print('Rewrote',count,'normalizations')
