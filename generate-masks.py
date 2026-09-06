import subprocess
from pathlib import Path

source_path = "data/benchmark"
images = Path(source_path)

for image in images.iterdir():
    print("Processing - ", image.name)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "inference.py",
            "--path",
            f"{source_path}/{image.name}",
            "--weights",
            "models/a1-v0.1.pt",
        ]
    )  # noqa: E501
    print(f"Finished processing. Output: {result}")

print("All processes completed successfully!")
