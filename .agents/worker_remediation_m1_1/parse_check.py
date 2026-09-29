import os, re

for root, dirs, files in os.walk(".agents"):
    for f in files:
        if f.endswith((".md", ".py", ".bib", ".json")):
            path = os.path.join(root, f)
            try:
                with open(path, encoding="utf-8") as fp:
                    content = fp.read()
                if "mehnaz2022ghostpost" in content or "vinayakumar2019deep" in content:
                    print("Found in:", path)
            except Exception as e:
                pass
