"""
run_all.py
Runs all 10 CV Lab 5 tasks in sequence.
Saves all outputs to the outputs/ directory.
"""

import subprocess
import sys
import os
import time

tasks = [
    ("download_images.py", "Downloading images"),
    ("task1.py",  "Task 1 – Global vs Adaptive Thresholding"),
    ("task2.py",  "Task 2 – Adaptive Threshold Strategy"),
    ("task3.py",  "Task 3 – Otsu's Thresholding"),
    ("task4.py",  "Task 4 – HSV Colour Segmentation"),
    ("task5.py",  "Task 5 – Canny Edge Detection"),
    ("task6.py",  "Task 6 – Region Growing"),
    ("task7.py",  "Task 7 – Watershed Segmentation"),
    ("task8.py",  "Task 8 – Watershed Tuning"),
    ("task9.py",  "Task 9 – K-Means Segmentation"),
    ("task10.py", "Task 10 – Multi-Method Segmentation System"),
]

print("=" * 60)
print("  CV LAB 5 — ALL TASKS RUNNER")
print("=" * 60)

passed = []
failed = []

for script, description in tasks:
    print(f"\n{'─'*60}")
    print(f"  Running: {description}")
    print(f"  Script : {script}")
    print(f"{'─'*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        passed.append(script)
        print(f"  ✓ Completed in {elapsed:.1f}s")
    else:
        failed.append(script)
        print(f"  ✗ FAILED (exit code {result.returncode})")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Passed : {len(passed)}/{len(tasks)}")
if passed:
    for s in passed:
        print(f"    ✓ {s}")
if failed:
    print(f"  Failed : {len(failed)}")
    for s in failed:
        print(f"    ✗ {s}")

print(f"\n  Output files in: {os.path.abspath('outputs')}/")
for f in sorted(os.listdir("outputs")):
    size = os.path.getsize(os.path.join("outputs", f))
    print(f"    {f:40s}  {size // 1024:>5} KB")
print("=" * 60)
