import os
import glob
import re

def patch_opendatasets():
    target_dir = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\DataProcessing"
    files = glob.glob(os.path.join(target_dir, "*.py"))
    
    count = 0
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        pattern = r"(import opendatasets as od\s+print\(f\"Downloading from Kaggle via opendatasets: \{url\}\"\)\s+od\.download\(url, data_dir=os\.path\.dirname\(filename\)\)\s+)return filename"
        
        replacement = r"\1dataset_slug = url.split('/')[-1].split('?')[0]\n                downloaded_dir = os.path.join(os.path.dirname(filename), dataset_slug)\n                if os.path.isdir(downloaded_dir):\n                    return downloaded_dir\n                return filename"
        
        new_content, num_subs = re.subn(pattern, replacement, content)
        
        if num_subs > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            count += 1
            print(f"Patched {os.path.basename(file_path)}")
            
    print(f"Total patched: {count}")

if __name__ == "__main__":
    patch_opendatasets()
