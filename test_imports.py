import sys
import os
import glob
sys.path.append(os.getcwd())
for f in glob.glob("backend/**/*.py", recursive=True):
    if not f.endswith("__init__.py"):
        mod_name = f.replace("/", ".").replace(".py", "")
        print(f"Testing {mod_name}...")
        try:
            __import__(mod_name)
        except Exception as e:
            print(f"Failed to import {mod_name}: {e}")
