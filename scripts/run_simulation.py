"""Entry point to run a toy LBM simulation using weather data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from fetch_weather import fetch_weather
from coordinate_transform import local_xy
from lbm_core import LBMCore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple LBM simulation")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--domain-size", type=int, default=50, help="Domain size in km (radius)")
    parser.add_argument("--resolution", type=int, default=100, help="Grid spacing in meters")
    parser.add_argument("--duration", type=int, default=1, help="Number of timesteps")
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    weather = fetch_weather(args.lat, args.lon)
    # Extract relevant weather parameters (e.g., temperature, wind speed)
    temperature = weather.get("temperature", 300)  # Default to 300K if not provided
    wind_speed = weather.get("wind_speed", 0)  # Default to 0 m/s if not provided

    # compute grid size
    length_m = args.domain_size * 1000 * 2
    n = max(1, length_m // args.resolution)
    shape = (int(n), int(n), 10)  # thin vertical extent for toy model

    # Initialize LBMCore with weather-influenced initial conditions
    lbm = LBMCore(shape, initial_temperature=temperature, initial_wind_speed=wind_speed)
    lbm.run(args.duration)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "final_rho.npy", lbm.rho)
    print(f"Simulation finished. Results stored in {args.output}")


if __name__ == "__main__":
    main()
