"""Polars-based electoral data pipeline.

Reads raw vote CSV and candidate metadata, filters to a municipality,
aggregates per candidate and location, and writes per-candidate JSONs
plus summary artifacts for front-end consumption.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable

import polars as pl
import yaml

DEFAULT_CONFIG = Path("config/default.yaml")

UTF8_ENCODINGS = {"utf8", "utf-8", "utf8-lossy"}


def ensure_utf8_csv(path: Path, source_encoding: str | None) -> tuple[Path, Path | None]:
    """If the source file is not utf-8, transcode to a temp utf-8 copy and return its path.

    Returns (resolved_path, temp_path_or_None). Caller is responsible for deleting temp_path.
    """
    enc = (source_encoding or "utf-8").lower().replace("_", "-")
    if enc in UTF8_ENCODINGS:
        return path, None

    # Stream-transcode to utf-8 without loading whole file into memory.
    with NamedTemporaryFile(mode="w", delete=False, suffix=".utf8.csv") as tmp:
        with path.open("r", encoding=source_encoding or "latin-1", errors="strict") as src:
            for line in src:
                tmp.write(line)
        temp_path = Path(tmp.name)
    return temp_path, temp_path


def load_config(config_path: Path | str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["processing"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["processing"]["output_base"]).mkdir(parents=True, exist_ok=True)


def build_votes_scan(cfg: Dict[str, Any], votes_path: Path) -> pl.LazyFrame:
    inputs = cfg["inputs"]
    filters = cfg["filters"]
    address_suffix = cfg["processing"]["address_suffix"]

    dtypes = {
        "CD_MUNICIPIO": pl.Utf8,
        "SQ_CANDIDATO": pl.Utf8,
        "QT_VOTOS": pl.Int64,
        "NR_VOTAVEL": pl.Utf8,
        "NM_VOTAVEL": pl.Utf8,
        "DS_LOCAL_VOTACAO_ENDERECO": pl.Utf8,
        "NM_LOCAL_VOTACAO": pl.Utf8,
        "DS_CARGO": pl.Utf8,
    }

    scan = pl.scan_csv(
        votes_path,
        separator=inputs.get("separator", ";"),
        encoding="utf8",
        infer_schema_length=inputs.get("infer_schema_length", 2000),
        dtypes=dtypes,
        has_header=True,
        null_values=["", "null", "NULL"],
    )

    filtered = scan.filter(pl.col("CD_MUNICIPIO") == filters["municipality_code"])
    cargos = filters.get("cargos") or []
    if cargos:
        filtered = filtered.filter(pl.col("DS_CARGO").is_in(cargos))

    id_expr = pl.concat_str(
        [
            pl.col("NM_LOCAL_VOTACAO"),
            pl.lit(", "),
            pl.col("DS_LOCAL_VOTACAO_ENDERECO"),
        ]
    )

    cleaned_address = (
        (pl.col("DS_LOCAL_VOTACAO_ENDERECO") + pl.lit(address_suffix))
        .str.replace(r",\s*,", ", ", literal=False)
        .str.replace(r"\s+", " ", literal=False)
        .str.strip_chars()
    )

    return filtered.with_columns(
        [
            id_expr.alias("ID_ENDERECO"),
            cleaned_address.alias("DS_LOCAL_VOTACAO_ENDERECO"),
        ]
    ).with_columns(pl.col("QT_VOTOS").cast(pl.Int64).alias("QT_VOTOS"))


def compute_valid_votes(base_votes: pl.LazyFrame) -> pl.LazyFrame:
    return (
        base_votes.filter(~pl.col("NR_VOTAVEL").is_in(["95", "96"]))
        .group_by("ID_ENDERECO")
        .agg(pl.col("QT_VOTOS").sum().alias("TOTAL_VOTOS_VALIDOS"))
    )


def aggregate_votes(base_votes: pl.LazyFrame) -> pl.LazyFrame:
    return (
        base_votes.group_by(["SQ_CANDIDATO", "ID_ENDERECO"])
        .agg(
            [
                pl.col("QT_VOTOS").sum().alias("VOTES"),
                pl.col("NR_VOTAVEL").first(),
                pl.col("NM_VOTAVEL").first(),
                pl.col("DS_LOCAL_VOTACAO_ENDERECO").first(),
                pl.col("NM_LOCAL_VOTACAO").first(),
                pl.col("DS_CARGO").first(),
            ]
        )
    )


def load_candidate_metadata(cfg: Dict[str, Any], candidate_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    inputs = cfg["inputs"]
    filters = cfg["filters"]

    c_dtypes = {
        "SG_UE": pl.Utf8,
        "SQ_CANDIDATO": pl.Utf8,
        "NM_URNA_CANDIDATO": pl.Utf8,
        "NR_CANDIDATO": pl.Utf8,
        "SG_PARTIDO": pl.Utf8,
        "NR_PARTIDO": pl.Utf8,
        "NM_PARTIDO": pl.Utf8,
    }

    cand_scan = pl.scan_csv(
        candidate_path,
        separator=inputs.get("separator", ";"),
        encoding="utf8",
        infer_schema_length=inputs.get("infer_schema_length", 2000),
        dtypes=c_dtypes,
        has_header=True,
        null_values=["", "null", "NULL"],
    ).filter(pl.col("SG_UE") == filters["municipality_code"])

    candidate_info = cand_scan.select(
        "SQ_CANDIDATO",
        "NM_URNA_CANDIDATO",
        "NR_CANDIDATO",
        "SG_PARTIDO",
        "NM_PARTIDO",
    ).unique()

    candidate_info = candidate_info.with_columns(
        (
            pl.col("NM_URNA_CANDIDATO")
            + pl.lit(" - ")
            + pl.col("NR_CANDIDATO")
            + pl.lit(" (")
            + pl.col("SG_PARTIDO")
            + pl.lit(")")
        ).alias("VOTAVEL")
    )

    partido_info = (
        cand_scan.select("NR_PARTIDO", "SG_PARTIDO")
        .unique()
        .with_columns((pl.col("NR_PARTIDO") + pl.lit(" - ") + pl.col("SG_PARTIDO")).alias("PARTIDO"))
        .select("NR_PARTIDO", "PARTIDO")
    )

    return candidate_info, partido_info


def enrich_with_candidates(
    agg_votes: pl.LazyFrame, candidate_info: pl.LazyFrame, partido_info: pl.LazyFrame
) -> pl.LazyFrame:
    joined = agg_votes.join(candidate_info, on="SQ_CANDIDATO", how="left")

    joined = joined.with_columns(
        pl.col("NR_VOTAVEL").str.slice(0, 2).alias("NR_PARTIDO_JOIN")
    ).join(partido_info, left_on="NR_PARTIDO_JOIN", right_on="NR_PARTIDO", how="left")

    joined = joined.with_columns(
        [
            pl.when(pl.col("VOTAVEL").is_null() & (pl.col("NR_VOTAVEL") == "95"))
            .then(pl.lit("BRANCO"))
            .when(pl.col("VOTAVEL").is_null() & (pl.col("NR_VOTAVEL") == "96"))
            .then(pl.lit("NULO"))
            .when(pl.col("VOTAVEL").is_null())
            .then(pl.concat_str([pl.lit("Legenda: "), pl.col("PARTIDO")]))
            .otherwise(pl.col("VOTAVEL"))
            .alias("VOTAVEL_FILLED"),
            pl.when(pl.col("NR_VOTAVEL").is_in(["95", "96"]))
            .then(pl.lit("BRANCO/NULO"))
            .when(pl.col("PARTIDO").is_null() | (pl.col("PARTIDO") == ""))
            .then(pl.col("NM_VOTAVEL"))
            .otherwise(pl.col("PARTIDO"))
            .alias("PARTIDO_FILLED"),
        ]
    )

    return joined


def attach_valid_votes(
    enriched_votes: pl.LazyFrame, valid_votes: pl.LazyFrame, precision: int
) -> pl.LazyFrame:
    return (
        enriched_votes.join(valid_votes, on="ID_ENDERECO", how="left")
        .with_columns(
            [
                pl.when(pl.col("NR_VOTAVEL").is_in(["95", "96"]))
                .then(pl.lit(0.0))
                .otherwise((pl.col("VOTES") / pl.col("TOTAL_VOTOS_VALIDOS")) * 100)
                .round(precision)
                .alias("PROP_VOTES"),
                pl.col("VOTAVEL_FILLED").alias("VOTAVEL"),
                pl.col("PARTIDO_FILLED").alias("PARTIDO"),
            ]
        )
        .drop(["VOTAVEL_FILLED", "PARTIDO_FILLED", "NR_PARTIDO_JOIN"])
    )


def build_locations_table(final_votes: pl.LazyFrame) -> pl.DataFrame:
    return (
        final_votes.select(
            "ID_ENDERECO", "DS_LOCAL_VOTACAO_ENDERECO", "NM_LOCAL_VOTACAO"
        )
        .unique()
        .sort("ID_ENDERECO")
        .with_row_count(name="location_id", offset=1)
        .select("location_id", "ID_ENDERECO", "DS_LOCAL_VOTACAO_ENDERECO", "NM_LOCAL_VOTACAO")
        .collect()
    )


def attach_location_ids(final_votes: pl.LazyFrame, locations: pl.DataFrame) -> pl.LazyFrame:
    loc_lazy = locations.lazy().select("ID_ENDERECO", "location_id")
    return final_votes.join(loc_lazy, on="ID_ENDERECO", how="left")


def write_locations_json(locations: pl.DataFrame, output_path: Path) -> None:
    payload = {
        str(int(row["location_id"])): {
            "address": row["DS_LOCAL_VOTACAO_ENDERECO"],
            "name": row["NM_LOCAL_VOTACAO"],
            "lat": None,
            "lon": None,
        }
        for row in locations.iter_rows(named=True)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidate_jsons(final_votes: pl.LazyFrame, pattern: str) -> None:
    subset = final_votes.select(
        "SQ_CANDIDATO", "location_id", "VOTES", "PROP_VOTES", "TOTAL_VOTOS_VALIDOS"
    ).collect()

    for sq_candidato_tuple, group in subset.partition_by("SQ_CANDIDATO", as_dict=True).items():
        # Extract string from tuple - partition_by returns keys as tuples
        sq_candidato = sq_candidato_tuple[0] if isinstance(sq_candidato_tuple, tuple) else sq_candidato_tuple
        
        records = [
            {
                "location_id": int(row["location_id"]),
                "votes": int(row["VOTES"]),
                "proportion": float(row["PROP_VOTES"]),
                "total_votes_location": int(row["TOTAL_VOTOS_VALIDOS"])
                if row["TOTAL_VOTOS_VALIDOS"] is not None
                else None,
            }
            for row in group.iter_rows(named=True)
        ]
        out_path = Path(pattern.format(sq_candidato=sq_candidato))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidates_summary(final_votes: pl.LazyFrame, output_path: Path) -> None:
    summary = (
        final_votes.select(
            "SQ_CANDIDATO",
            "VOTAVEL",
            "PARTIDO",
            "VOTES",
            "PROP_VOTES",
            "location_id",
        )
        .group_by("SQ_CANDIDATO")
        .agg(
            [
                pl.col("VOTAVEL").first(),
                pl.col("PARTIDO").first(),
                pl.col("VOTES").sum().alias("total_votes"),
                pl.col("location_id").n_unique().alias("locations"),
                pl.col("VOTES").min().alias("min_abs_votes"),
                pl.col("VOTES").max().alias("max_abs_votes"),
                pl.col("PROP_VOTES").min().alias("min_prop_votes"),
                pl.col("PROP_VOTES").max().alias("max_prop_votes"),
            ]
        )
        .rename(
            {
                "SQ_CANDIDATO": "sq_candidato",
                "VOTAVEL": "candidate",
                "PARTIDO": "party",
            }
        )
        .sort("sq_candidato")
        .collect()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(output_path)


def run_pipeline(config_path: Path | str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config_path)
    ensure_output_dirs(cfg)

    temp_files: list[Path] = []
    try:
        votes_path, tmp_votes = ensure_utf8_csv(
            Path(cfg["inputs"]["raw_votes_path"]), cfg["inputs"].get("encoding")
        )
        if tmp_votes:
            temp_files.append(tmp_votes)

        candidate_path, tmp_cand = ensure_utf8_csv(
            Path(cfg["inputs"]["candidate_path"]), cfg["inputs"].get("encoding")
        )
        if tmp_cand:
            temp_files.append(tmp_cand)

        base_votes = build_votes_scan(cfg, votes_path)
        valid_votes = compute_valid_votes(base_votes)
        agg_votes = aggregate_votes(base_votes)

        candidate_info, partido_info = load_candidate_metadata(cfg, candidate_path)
        enriched = enrich_with_candidates(agg_votes, candidate_info, partido_info)
        final_votes = attach_valid_votes(
            enriched, valid_votes, cfg["processing"]["proportion_precision"]
        )

        locations = build_locations_table(final_votes)
        final_with_loc = attach_location_ids(final_votes, locations)

        processing_cfg = cfg["processing"]
        write_locations_json(locations, Path(processing_cfg["voting_locations_json"]))
        write_candidate_jsons(final_with_loc, processing_cfg["votes_pattern"])
        write_candidates_summary(final_with_loc, Path(processing_cfg["candidates_summary"]))
    finally:
        for tmp in temp_files:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    run_pipeline()
