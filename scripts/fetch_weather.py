import argparse
import json
import time
from pathlib import Path

import requests
from pyproj import Transformer

API_URL = "https://api.open-meteo.com/v1/forecast"

SURFACE_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "windspeed_10m",
    "winddirection_10m",
    "precipitation",
]

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]
PRESSURE_VARS = ["temperature", "relative_humidity", "windspeed", "winddirection"]

DEFAULT_PARAMS = {
    "forecast_days": 1,
    "timezone": "UTC",
}

def build_params(lat: float, lon: float) -> dict:
    hourly = SURFACE_VARS[:]
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_VARS:
            hourly.append(f"{var}_{level}hPa")
    params = dict(DEFAULT_PARAMS)
    params.update({
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly),
    })
    return params


def fetch_weather(lat: float, lon: float, retries: int = 3, delay: float = 5.0):
    params = build_params(lat, lon)
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def wgs84_to_webmercator(lat: float, lon: float):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch weather data for GPS coordinates")
    parser.add_argument("lat", type=float, help="Latitude in degrees")
    parser.add_argument("lon", type=float, help="Longitude in degrees")
    parser.add_argument("--output", type=Path, default=Path("data/weather/latest.json"), help="Output JSON path")
    args = parser.parse_args()

    data = fetch_weather(args.lat, args.lon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))

    mercator = wgs84_to_webmercator(args.lat, args.lon)

    print(f"Coordinates (lat, lon): {args.lat}, {args.lon}")
    print(f"Web Mercator: {mercator[0]:.2f}, {mercator[1]:.2f}")
    print(f"Saved weather data to {args.output}")

if __name__ == "__main__":
    main()
