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
        "models/a1-v0.3-300-s.pt",
        "--class-map",
        "0=divider,1=road,2=sidewalk,3=vehicle",
        "--output",
        "./output/a1-v0.3-300s",
    ]
)  # noqa: E501

print("All processes completed successfully!")
