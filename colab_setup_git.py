# SETUP SCRIPT FOR COLAB (Git-based Workflow)
# Usage: Copy this content into a cell in your Colab notebook.

import os
import sys
from google.colab import drive

# ==========================================
# CONFIGURATION
# ==========================================
# Replace with your GitHub URL after you push the code
GITHUB_REPO_URL = "https://github.com/meeka1l/Manga_onomatopoia_translator.git"
REPO_NAME = "Manga_onomatopoia_translator" # Folder name after cloning

# Path to your project in Google Drive (where weights/data live)
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/COO-Comic-Onomatopoeia-main"
# ==========================================

def setup_environment():
    print("🚀 Starting Setup...")
    
    # 1. Mount Google Drive
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    else:
        print("Drive already mounted.")

    # 2. Clone/Update Code from GitHub
    # Note: If the repo is private, you'll need to use a Personal Access Token (PAT)
    # git clone https://username:token@github.com/...
    
    if not os.path.exists(REPO_NAME):
        print(f"Cloning repository from {GITHUB_REPO_URL}...")
        # For now, we assume public or auth is handled. 
        # If this fails, user needs to set up auth.
        os.system(f"git clone {GITHUB_REPO_URL} {REPO_NAME}")
    else:
        print("Repository exists. Pulling latest changes...")
        os.chdir(REPO_NAME)
        os.system("git pull")
        os.chdir("..")

    # 3. Enter Project Directory
    if os.path.exists(REPO_NAME):
        os.chdir(REPO_NAME)
        print(f"Changed directory to {os.getcwd()}")
    else:
        print(f"❌ Error: Repository {REPO_NAME} not found.")
        return

    # 4. Symlink Large Files/Folders from Drive
    # This tricks the code into thinking the files are local
    
    files_to_link = {
        # 'Local Path inside Repo' : 'Drive Path'
        'weights': f'{DRIVE_PROJECT_ROOT}/weights',
        'COO-data': f'{DRIVE_PROJECT_ROOT}/COO-data',
        'ABCNetv2/ABCNetv2.pth': f'{DRIVE_PROJECT_ROOT}/ABCNetv2/ABCNetv2.pth',
        'TRBA/TRBA_Rot+SAR.pth': f'{DRIVE_PROJECT_ROOT}/TRBA/TRBA_Rot+SAR.pth'
    }
    
    print("\n🔗 Linking Data & Weights from Drive...")
    for local_path, drive_path in files_to_link.items():
        # Check if drive file exists
        if not os.path.exists(drive_path):
            print(f"  ⚠️ Warning: Source not found in Drive: {drive_path}")
            continue
            
        # Check if local link already exists
        if os.path.exists(local_path):
            # If it's a symlink, good. If it's a real file/folder, we might define it.
            if os.path.islink(local_path):
                print(f"  ✓ {local_path} is already linked.")
            else:
                print(f"  ℹ️ {local_path} exists as a real file/folder. Skipping link.")
            continue
            
        # Create parent dir if needed (e.g. for ABCNetv2/ABCNetv2.pth)
        parent_dir = os.path.dirname(local_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            
        # Create Link
        try:
            os.symlink(drive_path, local_path)
            print(f"  ✓ Linked {local_path} -> {drive_path}")
        except Exception as e:
            print(f"  ❌ Failed to link {local_path}: {e}")

    print("\n✅ Setup Complete! You can now run your model inference.")

if __name__ == "__main__":
    setup_environment()
