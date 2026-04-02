import os

BASE_DIR = "dataset"

CLASS_MAP = {
    "apple":  ["unripe", "ripe", "overripe"],
    "banana": ["unripe", "ripe", "overripe"],
    "pear":   ["unripe", "ripe", "overripe"],
}

for fruit, stages in CLASS_MAP.items():
    for stage in stages:
        folder = os.path.join(BASE_DIR, fruit, stage)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        for idx, filename in enumerate(files, start=1):
            ext = filename.split(".")[-1]
            new_name = f"{fruit}_{stage}_{idx:03d}.{ext}"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_name}")
