import hashlib, json, os, re, shutil, subprocess, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/home/hiamichenha/unitree_h1_learn")
CKPT=ROOT/"ACT/ckpt_models_medium_resnet34"
LOG=ROOT/"project_execution/evidence/logs/task2_medium_resnet34_train_12000.log"
REPORT=ROOT/"project_execution/evidence/reports"
SHOTS=ROOT/"project_execution/evidence/screenshots"
REPORT.mkdir(parents=True,exist_ok=True); SHOTS.mkdir(parents=True,exist_ok=True)
seen=set()
while True:
    for path in sorted(CKPT.glob("policy_epoch_*.ckpt")):
        m=re.search(r"policy_epoch_(\d+)\.ckpt$",path.name)
        if not m: continue
        epoch=int(m.group(1))
        if epoch<2000 or epoch in seen: continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        text=LOG.read_text(errors="replace") if LOG.exists() else ""
        pairs=re.findall(r"Epoch (\d+)\nTrain loss: ([0-9.eE+-]+)",text)
        epochs=[int(a) for a,b in pairs]; losses=[float(b) for a,b in pairs]
        curve=SHOTS/f"task2_medium_loss_epoch{epoch}.png"
        if epochs:
            plt.figure(figsize=(10,5)); plt.plot(epochs,losses,linewidth=.8); plt.yscale("log"); plt.xlabel("optimization step"); plt.ylabel("train loss"); plt.title(f"ACT ResNet34 through {epoch}"); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(curve,dpi=160); plt.close()
        desktop=SHOTS/f"task2_medium_desktop_epoch{epoch}.png"
        subprocess.run(["gnome-screenshot","-f",str(desktop)],env={**os.environ,"DISPLAY":":0"},check=False)
        free=shutil.disk_usage("/").free
        status={"epoch":epoch,"checkpoint":str(path),"bytes":path.stat().st_size,"sha256":digest,"loss_points":len(pairs),"curve":str(curve),"desktop_screenshot":str(desktop),"disk_free_bytes":free}
        (REPORT/f"task2_medium_epoch{epoch}_status.json").write_text(json.dumps(status,indent=2))
        (REPORT/f"task2_medium_epoch{epoch}.sha256").write_text(f"{digest}  {path}\n")
        if free<3*1024**3: (REPORT/"LOW_DISK_WARNING.txt").write_text(f"Low disk at epoch {epoch}: {free} bytes\n")
        seen.add(epoch)
    text=LOG.read_text(errors="replace") if LOG.exists() else ""
    if "Training finished:" in text: break
    time.sleep(10)
