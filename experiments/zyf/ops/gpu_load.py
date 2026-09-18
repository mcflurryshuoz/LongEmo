"""User-requested bounded GPU activity, separate from benchmark inference."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=12);p.add_argument('--duty',type=float,default=.5);p.add_argument('--size',type=int,default=8192);a=p.parse_args()
assert 0<a.hours<=12 and 0<a.duty<=.8 and 1024<=a.size<=8192
import torch
torch.set_num_threads(1)
stop=False

def halt(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
root=Path('/root/longemo/runtime/ops');started=time.time();deadline=started+a.hours*3600
state={'pid':os.getpid(),'purpose':'user-requested GPU activity; not model inference or benchmark cost','gpu_visible_id':os.environ.get('CUDA_VISIBLE_DEVICES'),'hours':a.hours,'duty_target':a.duty,'started_unix':started,'deadline_unix':deadline}
assert torch.cuda.is_available()
with torch.inference_mode():
 x=torch.randn(a.size,a.size,device='cuda',dtype=torch.float16);y=torch.randn_like(x);z=torch.empty_like(x)
 torch.mm(x,y,out=z);torch.cuda.synchronize();assert torch.isfinite(z[0,:32]).all().item()
 state['device']=torch.cuda.get_device_name(0);state['allocated_MiB']=torch.cuda.memory_allocated()/2**20
 state['status']='running';last=0
 while not stop and time.time()<deadline:
  t=time.monotonic()
  while time.monotonic()-t<.2 and not stop:
   torch.mm(x,y,out=z);torch.cuda.synchronize()
  busy=time.monotonic()-t
  time.sleep(busy*(1/a.duty-1))
  if time.time()-last>=30:
   free,total=torch.cuda.mem_get_info()
   if free<8*2**30:state['stop_reason']='less than 8 GiB free';break
   q=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
   observed=q.stdout.strip();state.update(updated_unix=time.time(),observed_gpu=observed)
   if q.returncode==0 and observed.split(',')[0].strip().isdigit() and int(observed.split(',')[0])>=82:
    state['stop_reason']='temperature >= 82 C';break
   tmp=root/'gpu_load_status.tmp';tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,root/'gpu_load_status.json');print(json.dumps(state),flush=True);last=time.time()
 state.update(status='stopped',ended_unix=time.time());(root/'gpu_load_status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
