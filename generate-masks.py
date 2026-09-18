import subprocess
from pathlib import Path

source_path = "data/benchmark"
images = Path(source_path)

result = subprocess.run(
    [
        "uv",
        "run",
        "python",
        "inference.py",
        "--path",
        f"{source_path}",
        "--weights",
        "models/yolo26s-sem.pt",
        "--class-map",
        "0=road,1=sidewalk",
        "--output",
        "./output/baseline-cityscapes",
    ]
)  # noqa: E501

print("All processes completed successfully!")
