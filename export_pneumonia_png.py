from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def export_split(split_name: str, in_dir: Path, out_root: Path) -> None:
    images_path = in_dir / f"{split_name}_images.npy"
    labels_path = in_dir / f"{split_name}_labels.npy"

    images = np.load(images_path)
    labels = np.load(labels_path)

    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        # Images are grayscale 28x28; make sure we save as 2D.
        img2d = np.squeeze(img)
        # Labels are typically shaped (N, 1); convert to Python int.
        label_int = int(label) if np.isscalar(label) else int(label[0])
        filename = f"{split_name}_{idx:05d}_label-{label_int}.png"
        plt.imsave(out_dir / filename, img2d, cmap="gray")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pneumo_dir = project_root / "pneumoniamnist"
    out_root = project_root / "dataset_export_png"

    for split in ("train", "val", "test"):
        export_split(split, pneumo_dir, out_root)


if __name__ == "__main__":
    main()

