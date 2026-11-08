from huggingface_hub import list_repo_files, hf_hub_download
import shutil, os

REPO = "yasirfaizahmed/license-plate-object-detection"
files = list_repo_files(REPO)
print("files in repo:", files)

pts = [f for f in files if f.endswith(".pt")]
if not pts:
    raise SystemExit("No .pt file found, paste the file list to me")

path = hf_hub_download(REPO, pts[0])
os.makedirs("models", exist_ok=True)
shutil.copy(path, "models/plate_detector.pt")
print("saved models/plate_detector.pt from", pts[0])