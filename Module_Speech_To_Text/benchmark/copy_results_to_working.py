import os
import shutil
import glob

# ════════════════════════════════════════════════════════════════
#  FIND RESULT CSVs UNDER /kaggle/input
# ════════════════════════════════════════════════════════════════
matches = glob.glob("/kaggle/input/**/results__*.csv", recursive=True)

if not matches:
    print("No results__*.csv files found anywhere under /kaggle/input/")
    print("\nAttached input directories:")
    for name in os.listdir("/kaggle/input"):
        print(f"  /kaggle/input/{name}/")
    raise SystemExit(1)

print(f"Found {len(matches)} file(s):\n")
for p in matches:
    print(f"  {p}")

# ════════════════════════════════════════════════════════════════
#  COPY TO /kaggle/working
# ════════════════════════════════════════════════════════════════
print()
for src in matches:
    dst = os.path.join("/kaggle/working", os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"  copied → {dst}")

print(f"\nDone. {len(matches)} file(s) ready in /kaggle/working/")
