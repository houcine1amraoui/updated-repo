import subprocess
import sys

steps = [
    # ("Preprocess",            "preprocess.py"),
    ("Sliding Windows",       "build_windows.py"),
    ("Edge List",             "build_graph.py"),
    ("PyG Dataset",           "build_dataset.py"),
    # ("GDN Training",          "train_gdn.py"),
    # ("GDN Evaluation",        "evaluate_gdn.py"),
]

for name, script in steps:
    print(f"\n{'='*50}")
    print(f"  Step: {name}")
    print(f"{'='*50}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

print("\nAll steps completed successfully.")
