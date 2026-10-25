#!/usr/bin/env python3
"""Convert already split/re-written Index vocabulary heads in the QAIRT Linux container.

Precede with split_index_snpe_head.py --shards 16, rewrite_index_snpe_norm.py
for each head, and float parity checks. Uses representative original head inputs.
Never marks a model device-verified or publishes it. All calibration paths must
be visible inside the container. QAIRT/SNPE 2.47.0.260601 is required.
"""
import argparse, pathlib, subprocess, os
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('rewritten',type=pathlib.Path)
p.add_argument('calibration_list',type=pathlib.Path)
p.add_argument('output',type=pathlib.Path)
p.add_argument('--sdk',type=pathlib.Path,default=pathlib.Path('/sdk'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ, PYTHONPATH=str(a.sdk/'lib/python'),LD_LIBRARY_PATH=str(a.sdk/'lib/x86_64-linux-clang'))
for i in range(16):
 name=f'head-{i:02}'
 source=a.rewritten/name/'decoder.onnx';floating=a.output/(name+'.float.dlc');target=a.output/(name+'.dlc')
 with (a.output/(name+'.log')).open('w') as log:
  subprocess.run([str(a.sdk/'bin/x86_64-linux-clang/snpe-onnx-to-dlc'),'-i',str(source),'-o',str(floating),'--preserve_io','layout','--no_simplification'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([str(a.sdk/'bin/x86_64-linux-clang/snpe-dlc-quant'),'--input_dlc',str(floating),'--input_list',str(a.calibration_list),'--output_dlc',str(target),'--act_bitwidth','8','--weights_bitwidth','8','--bias_bitwidth','32'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(target,flush=True)
