"""Geocode voting locations using Nominatim.

Default behavior performs a small sample run so you can verify output
before geocoding the full set. Use --all to geocode everything or
--limit to set a custom sample size.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import yaml

DEFAULT_CONFIG = Path("config/default.yaml")


def load_config(config_path: Path | str) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_locations(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def select_targets(locations: Dict[str, Any], limit: int | None) -> List[Tuple[str, Dict[str, Any]]]:
    pending = [(loc_id, data) for loc_id, data in locations.items() if data.get("lat") in (None, "", 0) or data.get("lon") in (None, "", 0)]
    if limit is None or limit <= 0:
        return pending
    return pending[:limit]


def clean_address(address: str) -> str:
    # Replace double commas/spaces and strip
    addr = re.sub(r",\s*,", ", ", address)
    addr = re.sub(r"\s+", " ", addr)
    return addr.strip().strip(',')


def geocode_address(address: str, user_agent: str, fuzzy: bool = False) -> Tuple[float | None, float | None]:
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": user_agent}
    params = {"q": address, "format": "json", "limit": 3 if fuzzy else 1, "addressdetails": 0}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None, None
    first = data[0]
    try:
        return float(first.get("lat")), float(first.get("lon"))
    except (TypeError, ValueError):
        return None, None


def clamp_coordinate(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    return min(max(value, lower), upper)


def clamp_to_boundaries(lat: float | None, lon: float | None, bounds: Dict[str, float]) -> Tuple[float | None, float | None]:
    clamped_lat = clamp_coordinate(lat, bounds["south"], bounds["north"])
    clamped_lon = clamp_coordinate(lon, bounds["west"], bounds["east"])
    return clamped_lat, clamped_lon


def offset_coordinate(lat: float, lon: float, meters: float) -> Tuple[float, float]:
    """Randomly offset a coordinate by up to 'meters' distance."""
    # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
    angle = random.uniform(0, 2 * math.pi)
    distance = random.uniform(0, meters)
    
    dlat = (distance * math.cos(angle)) / 111000
    dlon = (distance * math.sin(angle)) / (111000 * math.cos(math.radians(lat)))
    
    return lat + dlat, lon + dlon


def deduplicate_coordinates(locations: Dict[str, Any], offset_meters: float, bounds: Dict[str, float]) -> None:
    """Find duplicate coordinates and offset them randomly."""
    coord_map: Dict[Tuple[float, float], List[str]] = {}
    
    for loc_id, data in locations.items():
        lat, lon = data.get("lat"), data.get("lon")
        if lat is not None and lon is not None:
            key = (round(lat, 6), round(lon, 6))
            coord_map.setdefault(key, []).append(loc_id)
    
    for coord, loc_ids in coord_map.items():
        if len(loc_ids) > 1:
            # Keep first, offset rest
            for loc_id in loc_ids[1:]:
                lat, lon = locations[loc_id]["lat"], locations[loc_id]["lon"]
                new_lat, new_lon = offset_coordinate(lat, lon, offset_meters)
                new_lat, new_lon = clamp_to_boundaries(new_lat, new_lon, bounds)
                locations[loc_id]["lat"] = new_lat
                locations[loc_id]["lon"] = new_lon


def save_locations(locations: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(locations, fh, ensure_ascii=False, indent=2)


def run(config_path: Path | str, limit: int | None, all_records: bool) -> None:
    cfg = load_config(config_path)
    geo_cfg = cfg.get("geocoding", {})

    source_path = Path(cfg["processing"]["voting_locations_json"])
    output_raw = Path(geo_cfg.get("output_raw", "data/processed/voting_locations_geocoded.json"))
    output_clean = Path(geo_cfg.get("output_clean", "data/processed/voting_locations_geocoded_cleaned.json"))
    user_agent = geo_cfg.get("user_agent", "rio-electoral-map-2024")
    rate_limit = float(geo_cfg.get("rate_limit_seconds", 1.0))

    locations = load_locations(source_path)
    boundaries = geo_cfg.get("boundaries", {})
    offset_meters = float(geo_cfg.get("duplicate_offset_meters", 30))

    effective_limit = None if all_records else (limit if limit is not None else geo_cfg.get("sample_limit", 5))
    targets = select_targets(locations, effective_limit)
    if not targets:
        print("No locations to geocode (all have lat/lon or limit=0)")
        return

    print(f"Geocoding {len(targets)} location(s); rate limit {rate_limit}s")
    successes = 0
    null_ids = []
    raw_results = {loc_id: dict(data) for loc_id, data in locations.items()}

    # First pass: geocode all
    for idx, (loc_id, data) in enumerate(targets, start=1):
        raw_address = data.get("address") or ""
        address = clean_address(raw_address)
        lat, lon = geocode_address(address, user_agent)
        raw_results[loc_id]["lat"] = lat
        raw_results[loc_id]["lon"] = lon
        if lat is None or lon is None:
            null_ids.append(loc_id)
        else:
            successes += 1
        if idx < len(targets):
            time.sleep(rate_limit)

    # Second pass: retry nulls with fuzzy matching
    if null_ids:
        print(f"Retrying {len(null_ids)} failed location(s) with fuzzy matching...")
        for loc_id in null_ids:
            data = raw_results[loc_id]
            raw_address = data.get("address") or ""
            address = clean_address(raw_address)
            lat, lon = geocode_address(address, user_agent, fuzzy=True)
            raw_results[loc_id]["lat"] = lat
            raw_results[loc_id]["lon"] = lon
            if lat is not None and lon is not None:
                successes += 1
            time.sleep(rate_limit)

    # Create cleaned copy with boundary clamp and de-duplication
    cleaned_results = {}
    for loc_id, data in raw_results.items():
        lat, lon = data.get("lat"), data.get("lon")
        lat, lon = clamp_to_boundaries(lat, lon, boundaries)
        cleaned_results[loc_id] = {**data, "lat": lat, "lon": lon}

    print("Deduplicating overlapping coordinates...")
    deduplicate_coordinates(cleaned_results, offset_meters, boundaries)

    save_locations(raw_results, output_raw)
    save_locations(cleaned_results, output_clean)
    print(f"Done. Geocoded {successes}/{len(targets)}. Raw: {output_raw}; Clean: {output_clean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geocode voting locations with Nominatim")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Override sample size (number of locations)")
    parser.add_argument("--all", action="store_true", help="Geocode all pending locations, ignoring limit")
    args = parser.parse_args()

    run(args.config, args.limit, args.all)
