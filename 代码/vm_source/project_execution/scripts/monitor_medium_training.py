from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path('/home/hiamichenha/unitree_h1_learn')
CKPT=ROOT/'ACT/ckpt_models_medium_resnet34'
LOG=ROOT/'project_execution/evidence/logs/task2_medium_resnet34_train_10000.log'
if not LOG.exists():
    LOG=ROOT/'project_execution/evidence/logs/task2_medium_resnet34_train_2000.log'
REPORT=ROOT/'project_execution/evidence/reports'
SHOTS=ROOT/'project_execution/evidence/screenshots'
REPORT.mkdir(parents=True,exist_ok=True); SHOTS.mkdir(parents=True,exist_ok=True)
seen=set()
while True:
    files=sorted(CKPT.glob('policy_epoch_*.ckpt'))
    for path in files:
        match=re.search(r'policy_epoch_(\d+)\.ckpt$',path.name)
        if not match: continue
        epoch=int(match.group(1))
        if epoch in seen: continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        (REPORT/f'task2_medium_epoch{epoch}.sha256').write_text(f'{digest}  {path}\n')
        text=LOG.read_text(errors='replace') if LOG.exists() else ''
        epochs=[int(x) for x in re.findall(r'^Epoch (\d+)$',text,re.M)]
        losses=[float(x) for x in re.findall(r'^Train loss: ([0-9.eE+-]+)$',text,re.M)]
        count=min(len(epochs),len(losses)); epochs=epochs[:count]; losses=losses[:count]
        if epochs:
            plt.figure(figsize=(10,5)); plt.plot(epochs,losses,linewidth=.8)
            plt.yscale('log'); plt.xlabel('optimization step'); plt.ylabel('train loss (log scale)')
            plt.title(f'ACT ResNet34 training through checkpoint {epoch}')
            plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(SHOTS/f'task2_medium_loss_epoch{epoch}.png',dpi=160); plt.close()
        (REPORT/f'task2_medium_epoch{epoch}_status.json').write_text(json.dumps({'epoch':epoch,'checkpoint':str(path),'bytes':path.stat().st_size,'sha256':digest,'log':str(LOG),'loss_points':count},indent=2))
        seen.add(epoch)
    if (CKPT/'TRAINING_COMPLETE').exists(): break
    time.sleep(15)
