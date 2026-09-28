"""Protocol-independent dataset handling: official download, validated loading and A/B audit."""

from __future__ import annotations

import csv
import gzip
import os
import zlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import httpx

from .utils import millis, save_json, sha256

DOMAINS = ("Electronics", "Movies_and_TV")
BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only"


def download(domain, directory):
    if domain not in DOMAINS:
        raise ValueError(f"Choose one of {DOMAINS}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{domain}.csv.gz"
    manifest = directory / f"{domain}.csv.gz.manifest.json"
    for path in (dest, manifest):
        if os.path.lexists(path):
            raise FileExistsError(f"Already exists: {path}; inspect it or choose a new directory")
    url = f"{BASE}/{domain}.csv.gz"
    with NamedTemporaryFile(dir=directory, prefix=f".{domain}-", suffix=".part") as partial:
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    partial.write(chunk)
            partial.flush()
            with gzip.open(partial.name, "rt", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
                if not {"user_id", "parent_asin", "rating", "timestamp"}.issubset(header):
                    raise ValueError("Downloaded file has an unexpected header")
                # Read through the CRC/trailer in bounded chunks before publishing anything.
                while f.read(1024 * 1024):
                    pass
        except (httpx.HTTPError, gzip.BadGzipFile, EOFError, zlib.error, UnicodeError) as exc:
            raise ValueError(f"Invalid or incomplete download for {domain}: {exc}") from exc
        record = {"source_url": url, "bytes": partial.tell(), "sha256": sha256(partial.name)}
        os.link(partial.name, dest)
        # ponytail: data/manifest publish separately; keep valid data on manifest failure.
        # Use an atomic directory bundle if multi-file transactions become necessary.
        save_json(manifest, record)
    return record


def load(con, path, name):
    # Only internal constant table names enter SQL; paths use DuckDB's bound reader API.
    if name not in {"a", "b"}:
        raise ValueError("Invalid internal table name")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing ID file: {path}; download it or update the config")
    relation = con.read_csv(str(Path(path).resolve()), header=True, all_varchar=True)
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    if not required.issubset(relation.columns):
        raise ValueError(f"Missing columns: {required - set(relation.columns)}")
    relation.create_view("incoming", replace=True)
    con.execute(f"""CREATE OR REPLACE TABLE {name} AS
        SELECT user_id, parent_asin, try_cast(rating AS DOUBLE) rating,
               try_cast(timestamp AS BIGINT) AS "timestamp" FROM incoming""")
    # Aggregate queries always return one row; fetchall()[0] makes that non-optional.
    invalid = con.execute(f"""SELECT count(*) FROM {name} WHERE
        user_id IS NULL OR trim(user_id) = '' OR parent_asin IS NULL OR trim(parent_asin) = ''
        OR rating IS NULL OR NOT isfinite(rating) OR rating NOT BETWEEN 1 AND 5
        OR timestamp IS NULL OR timestamp NOT BETWEEN 631152000000 AND 4102444800000""").fetchall()[
        0
    ][0]
    if invalid:
        raise ValueError(
            f"{invalid} invalid rows in {path}; expected Unix milliseconds and ratings 1–5"
        )


def connection():
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='data/.duckdb-spill'")
    return con


def audit(a, b, cutoff, min_rating=1):
    result = {
        "scope": "full supplied files; counts are not a training result",
        "initial_cutoff_utc": cutoff,
        "min_rating": min_rating,
        "domains": {},
    }
    with connection() as con:
        for name, path in [("a", a), ("b", b)]:
            load(con, path, name)
            count, users, items, first, last = con.execute(f"""SELECT count(*),
                count(DISTINCT user_id), count(DISTINCT parent_asin),
                min(timestamp), max(timestamp) FROM {name}""").fetchall()[0]
            unique = con.execute(
                f"SELECT count(*) FROM (SELECT DISTINCT * FROM {name})"
            ).fetchall()[0][0]
            pair_unique = con.execute(
                f"SELECT count(*) FROM (SELECT DISTINCT user_id,parent_asin FROM {name})"
            ).fetchall()[0][0]
            con.execute(
                f"CREATE TABLE {name}_filtered AS SELECT DISTINCT * FROM {name} WHERE rating >= ?",
                [min_rating],
            )
            ties = con.execute(f"""SELECT coalesce(sum(n),0) FROM
                (SELECT count(*) n FROM {name}_filtered GROUP BY user_id,timestamp HAVING count(*)>1)""").fetchall()[
                0
            ][0]
            # Same ambiguity policy as preparation: exclude all tied events, never invent order.
            con.execute(f"""CREATE TABLE {name}_ordered AS SELECT * FROM {name}_filtered
                QUALIFY count(*) OVER (PARTITION BY user_id,timestamp)=1""")
            stats = con.execute(
                f"""WITH lengths AS (
                SELECT user_id, count(*) n,
                count(*) FILTER (WHERE timestamp < ?) initial_n,
                count(*) FILTER (WHERE timestamp >= ?) later_n
                FROM {name}_ordered GROUP BY user_id)
                SELECT count(*), quantile_cont(n,0.5), quantile_cont(n,0.9),
                count(*) FILTER (WHERE initial_n >= 5),
                count(*) FILTER (WHERE initial_n >= 5 AND later_n > 0) FROM lengths""",
                [millis(cutoff), millis(cutoff)],
            ).fetchall()[0]
            result["domains"][name] = {
                "path": str(Path(path).resolve()),
                "sha256": sha256(path),
                "rows": count,
                "users": users,
                "items": items,
                "timestamp_ms_range": [first, last],
                "exact_duplicate_extra_rows": count - unique,
                "repeated_user_item_extra_rows": count - pair_unique,
                "tied_rows_after_rating_filter_and_dedup": ties,
                "after_tie_removal": dict(
                    zip(
                        [
                            "users",
                            "sequence_length_median",
                            "sequence_length_p90",
                            "users_with_at_least_5_initial_events",
                            "those_users_with_later_events",
                        ],
                        stats,
                    )
                ),
            }
        result["overlap"] = {}
        for col in ["user_id", "parent_asin"]:
            result["overlap"][col] = con.execute(f"""SELECT count(*) FROM
                (SELECT DISTINCT {col} FROM a INTERSECT SELECT DISTINCT {col} FROM b)""").fetchall()[0][0]
        result["notes"] = [
            "Overlap uses original identifiers before reindexing.",
            "Sequence lengths use the rating filter and exclude ambiguous timestamp ties.",
            "Five initial events allow three train events, validation and retention targets.",
            "Further vocabulary and cohort filtering can reduce eligibility.",
        ]
    return result
