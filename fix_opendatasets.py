import os
import glob

def patch_opendatasets():
    target_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataProcessing"))
    files = glob.glob(os.path.join(target_dir, "*.py"))
    
    old_code = '''            try:
                import opendatasets as od
                print(f"Downloading from Kaggle via opendatasets: {url}")
                od.download(url, data_dir=os.path.dirname(filename))
                return filename'''

    new_code = '''            try:
                import opendatasets as od
                print(f"Downloading from Kaggle via opendatasets: {url}")
                od.download(url, data_dir=os.path.dirname(filename))
                dataset_slug = url.split('/')[-1].split('?')[0]
                downloaded_dir = os.path.join(os.path.dirname(filename), dataset_slug)
                if os.path.isdir(downloaded_dir):
                    return downloaded_dir
                return filename'''
                
    count = 0
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        modified = False
        if old_code in content:
            content = content.replace(old_code, new_code)
            modified = True
            
        if "engine='python'" in content:
            content = content.replace(", engine='python'", "")
            content = content.replace("engine='python', ", "")
            modified = True
            
        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
            print(f"Patched {os.path.basename(file_path)}")
    print(f"Total patched: {count}")

if __name__ == "__main__":
    patch_opendatasets()
