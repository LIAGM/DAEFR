import os
import shutil
from huggingface_hub import list_repo_files, hf_hub_download


repo_id = "LIAGM/DAEFR_pretrain_model"

save_dir = "./experiments"
os.makedirs(save_dir, exist_ok=True)

# List all files in the repository
files = list_repo_files(repo_id=repo_id)
print(f"Files in repo {repo_id}: {files}")

for file_name in files:
    try:
        model_file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
        
        save_path = os.path.join(save_dir, file_name)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        shutil.copy(model_file_path, save_path)

        print(f"Downloaded {file_name} to {save_path}")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")

print(f"All files downloaded to {save_dir}")
