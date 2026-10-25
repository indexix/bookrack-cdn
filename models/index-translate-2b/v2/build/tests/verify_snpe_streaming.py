#!/usr/bin/env python3
"""Build small stateful DLC fixtures and compare streaming/prefill to retained CPU execution.

Run inside the pinned QAIRT Linux conversion environment. This does not certify NPU execution.
"""
import argparse, json, os, pathlib, subprocess
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sdk',type=pathlib.Path,default=pathlib.Path('/sdk'))
p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('.local/snpe-streaming-fixture'))
p.add_argument('--fixtures-only',action='store_true',help='Generate fixtures; compile/run the C++ harness separately in the compiler image')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(a.sdk/'lib/python'),LD_LIBRARY_PATH=str(a.sdk/'lib/x86_64-linux-clang'))
rng=np.random.default_rng(42)
(rng.normal(size=(1024,64)).astype('<f4')*.1).tofile(a.output/'embedding.f32')
parts=[]
def save(name,nodes,inputs,outputs,weights):
 model=H.make_model(H.make_graph(nodes,name,inputs,outputs,weights),opset_imports=[H.make_opsetid('',13)])
 model.ir_version=9;path=a.output/(name+'.onnx');onnx.save(model,path)
 subprocess.run([str(a.sdk/'bin/x86_64-linux-clang/snpe-onnx-to-dlc'),'-i',str(path),'-o',str(a.output/(name+'.dlc')),'--preserve_io','layout'],env=env,check=True)
def value(name,dimension=64):return H.make_tensor_value_info(name,T.FLOAT,[1,1,dimension])
for i in range(2):
 name=f'part-{i}';state=f'state_{i}';out=state+'_out'
 nodes=[H.make_node('Add',['embedding',state],['hidden']),H.make_node('Mul',['hidden','scale'],[out])]
 save(name,nodes,[value('embedding'),value(state)],[value('hidden'),value(out)],[N.from_array(np.array(.2+i*.05,dtype=np.float32),'scale')])
 parts.append({'modelFile':name+'.dlc','output':'hidden','states':[{'input':state,'output':out}]})
weight=rng.normal(size=(64,1024)).astype('<f4')*.1
for name,start,end in [('head',0,1024),('head-a',0,512),('head-b',512,1024)]:
 save(name,[H.make_node('MatMul',['embedding','weight'],['logits'])],[value('embedding')],[value('logits',end-start)],[N.from_array(weight[:,start:end].copy(),'weight')])
meta={'contextTokens':256,'decoder':{'hiddenSize':64,'headDim':16,'vocabSize':1024,'ropeTheta':10000,'partitions':parts+[{'modelFile':'head.dlc','output':'logits','states':[]}]}}
(a.output/'full.json').write_text(json.dumps(meta))
meta['decoder']['partitions']=parts+[{'modelFile':f'head-{name}.dlc','output':'logits','states':[],'outputSize':512,'vocabOffset':offset} for name,offset in [('a',0),('b',512)]]
(a.output/'sharded.json').write_text(json.dumps(meta))
if a.fixtures_only:
    print('Fixtures generated; native regression has NOT run')
    raise SystemExit(0)
binary=a.output/'streaming_reference'
subprocess.run(['g++','-std=c++17','-O1','-DOCR_VENDOR_TESTING','-I'+str(a.sdk/'include/SNPE'),'-Ithird_party/llama.cpp/vendor','-Iapp/src/main/cpp','app/src/main/cpp/vendor_snpe.cpp','scripts/tests/snpe_streaming_reference.cpp','-L'+str(a.sdk/'lib/x86_64-linux-clang'),'-lSNPE','-o',str(binary)],env=env,check=True)
subprocess.run([str(binary),str(a.output)],env=env,check=True)
