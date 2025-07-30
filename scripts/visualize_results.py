"""Simple visualization utility for LBM results."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LBM output")
    parser.add_argument("file", type=Path, help="Path to rho numpy file")
    parser.add_argument("--level", type=int, default=0, help="Vertical level index")
    args = parser.parse_args()

    data = np.load(args.file)
    level = min(args.level, data.shape[2] - 1)
    field = data[:, :, level]
    plt.imshow(field.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Density")
    plt.title(f"Density level {level}")
    plt.show()


if __name__ == "__main__":
    main()
