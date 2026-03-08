import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decord import VideoReader, cpu

root = "videos_val"
files = [f for f in os.listdir(root) if f.endswith(".mp4")]
total_files = len(files)

bad = []

def check_video(f):
    path = os.path.join(root, f)
    try:
        vr = VideoReader(path, ctx=cpu(0))
        _ = len(vr)
        return None
    except Exception:
        return path

workers = os.cpu_count() * 2

print(f"Scanning {total_files} videos using {workers} threads...\n")

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(check_video, f) for f in files]

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        if result:
            bad.append(result)
            print("CORRUPT:", result)

        if i % 1000 == 0:
            print(f"Checked {i}/{total_files}")

print("\nSCAN COMPLETE")
print("Total scanned:", total_files)
print("Corrupted found:", len(bad))

removed = 0

for p in bad:
    try:
        os.remove(p)
        removed += 1
        print("REMOVED:", p)
    except Exception:
        print("FAILED:", p)

remaining = total_files - removed

print("\nSUMMARY")
print("Original:", total_files)
print("Corrupted removed:", removed)
print("Remaining:", remaining)
