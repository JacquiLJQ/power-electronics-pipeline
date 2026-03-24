from pathlib import Path
import re

dir_path = Path("per_image_json")  # ← 改成你的目录

pattern = re.compile(r"^(\d+_png)\.rf\..*\.json$")

for p in dir_path.iterdir():
    if not p.is_file():
        continue

    m = pattern.match(p.name)
    if not m:
        continue

    new_name = f"{m.group(1)}.json"
    new_path = p.with_name(new_name)

    if new_path.exists():
        print(f"[REPLACE] {new_name} already exists, replacing")

    # replace 会在目标存在时直接覆盖
    p.replace(new_path)
    print(f"{p.name}  ->  {new_name}")
