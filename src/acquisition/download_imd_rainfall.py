"""Discover and download official IMD 0.25-degree yearly rainfall NetCDF.

The IMD page exposes yearly files through page/form-generated links rather than
an advertised data API.  This script therefore discovers the selected year's
request from the official page instead of constructing an unverified URL.

Run without ``--execute`` first.  A download is submitted only after the plan
has been reviewed and ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CATEGORY = "rainfall"
SOURCE_ID = "imd-rainfall-025deg-yearly-netcdf"
SOURCE_PAGE = "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / CATEGORY
ACQUISITION_LOG = PROJECT_ROOT / "data" / "metadata" / "acquisition_log.jsonl"
USER_AGENT = "SIH-Forecast-Bust-Research/1.0 (low-rate official-data acquisition)"
NETCDF_MAGIC = (b"CDF\x01", b"CDF\x02", b"\x89HDF")


@dataclass(frozen=True)
class DownloadPlan:
    """A request discovered from the official IMD page."""

    method: str
    url: str
    fields: dict[str, str]
    discovered_from: str


def build_session() -> requests.Session:
    """Create a low-rate HTTP session with bounded retries."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _is_allowed_imd_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return (
        host in {"imdpune.gov.in", "www.imdpune.gov.in"}
        or host.endswith(".imd.gov.in")
    )


def _candidate_netcdf_link(soup: BeautifulSoup, base_url: str, year: int) -> str | None:
    """Find an explicit NetCDF link/value that identifies the requested year."""
    year_text = str(year)
    candidates: list[str] = []

    for tag in soup.find_all(["a", "option"]):
        value = tag.get("href") if tag.name == "a" else tag.get("value")
        if not value:
            continue
        description = f"{tag.get_text(' ', strip=True)} {value}"
        if year_text in description and ".nc" in value.lower():
            candidates.append(urljoin(base_url, value))

    if not candidates:
        return None
    allowed = [url for url in candidates if _is_allowed_imd_url(url)]
    if not allowed:
        raise ValueError(f"IMD page exposed only non-IMD candidate URLs: {candidates}")
    return allowed[0]


def discover_plan(session: requests.Session, year: int) -> DownloadPlan:
    """Discover a direct link or form submission for one year from IMD HTML."""
    response = session.get(SOURCE_PAGE, timeout=(15, 60))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    direct_url = _candidate_netcdf_link(soup, response.url, year)
    if direct_url:
        return DownloadPlan("GET", direct_url, {}, response.url)

    year_text = str(year)
    for form in soup.find_all("form"):
        selected_name: str | None = None
        selected_value: str | None = None

        for select in form.find_all("select"):
            if not select.get("name"):
                continue
            for option in select.find_all("option"):
                value = str(option.get("value", "")).strip()
                label = option.get_text(" ", strip=True)
                if year_text == label or year_text == value or year_text in f"{label} {value}":
                    selected_name = str(select["name"])
                    selected_value = value or year_text
                    break
            if selected_name:
                break

        if not selected_name or selected_value is None:
            continue

        fields = {
            str(item["name"]): str(item.get("value", ""))
            for item in form.find_all("input")
            if item.get("name") and str(item.get("type", "")).lower() == "hidden"
        }
        fields[selected_name] = selected_value
        action = urljoin(response.url, str(form.get("action") or response.url))
        method = str(form.get("method") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise ValueError(f"Unsupported IMD form method: {method}")
        if not _is_allowed_imd_url(action):
            raise ValueError(f"Refusing non-IMD form action: {action}")
        return DownloadPlan(method, action, fields, response.url)

    raise RuntimeError(
        f"No official download link/form was found for {year}. "
        "Check that IMD lists the year on the source page; do not guess a URL."
    )


def output_path(year: int) -> Path:
    """Return the raw immutable-file destination used by preprocessing."""
    return RAW_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"


def _looks_like_html(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    prefix = response.content[:256].lstrip().lower()
    return "text/html" in content_type or prefix.startswith((b"<!doctype html", b"<html"))


def _follow_html_result(
    session: requests.Session,
    response: requests.Response,
    year: int,
) -> requests.Response:
    """Follow a NetCDF link returned by a form result page."""
    if not _looks_like_html(response):
        return response
    soup = BeautifulSoup(response.text, "html.parser")
    netcdf_url = _candidate_netcdf_link(soup, response.url, year)
    if not netcdf_url:
        raise RuntimeError(
            "IMD returned HTML but no verified NetCDF link for the selected year. "
            "The site form may have changed; inspect it manually before changing this script."
        )
    result = session.get(netcdf_url, timeout=(15, 300))
    result.raise_for_status()
    return result


def request_file(
    session: requests.Session,
    plan: DownloadPlan,
    year: int,
) -> requests.Response:
    """Submit the discovered request and return the file response."""
    if plan.method == "POST":
        response = session.post(plan.url, data=plan.fields, timeout=(15, 300))
    else:
        response = session.get(plan.url, params=plan.fields or None, timeout=(15, 300))
    response.raise_for_status()
    response = _follow_html_result(session, response, year)
    if not _is_allowed_imd_url(response.url):
        raise ValueError(f"Refusing file redirected outside official IMD hosts: {response.url}")
    return response


def validate_and_hash(path: Path) -> tuple[int, str]:
    """Reject HTML/error payloads and return file size and SHA-256."""
    size = path.stat().st_size
    if size < 1024:
        raise ValueError(f"Downloaded payload is implausibly small: {size} bytes")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        magic = handle.read(4)
        handle.seek(0)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    if not any(magic.startswith(signature) for signature in NETCDF_MAGIC):
        raise ValueError(
            f"Downloaded payload is not recognized as NetCDF/HDF5; magic={magic!r}"
        )
    return size, digest.hexdigest()


def append_log(record: dict[str, object]) -> None:
    ACQUISITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACQUISITION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def download_year(plan: DownloadPlan, year: int, overwrite: bool) -> Path:
    target = output_path(year)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Raw file already exists: {target}. It was not replaced. "
            "Use --overwrite only after independently verifying replacement is intended."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        with build_session() as session:
            response = request_file(session, plan, year)
            with partial.open("wb") as handle:
                handle.write(response.content)
        size, sha256 = validate_and_hash(partial)
        os.replace(partial, target)
    except Exception as error:
        partial.unlink(missing_ok=True)
        append_log(
            {
                "record_type": "acquisition",
                "source_id": f"{SOURCE_ID}-{year}",
                "category": CATEGORY,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "source_page": SOURCE_PAGE,
                "local_path": str(target.relative_to(PROJECT_ROOT)),
                "error": str(error),
            }
        )
        raise

    append_log(
        {
            "record_type": "acquisition",
            "source_id": f"{SOURCE_ID}-{year}",
            "category": CATEGORY,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "stored",
            "source_page": SOURCE_PAGE,
            "resolved_url": plan.url,
            "local_path": str(target.relative_to(PROJECT_ROOT)),
            "bytes": size,
            "sha256": sha256,
            "validation_status": "netcdf_magic_verified",
            "units_documented_by_provider": "mm",
            "grid_documented_by_provider": "0.25 degree, 135x129",
        }
    )
    print(f"Downloaded: {target}")
    print(f"Bytes: {size}")
    print(f"SHA-256: {sha256}")
    return target


def print_plan(plan: DownloadPlan, year: int) -> None:
    print(f"Category: {CATEGORY}")
    print(f"Provider: India Meteorological Department")
    print(f"Product: 0.25-degree daily yearly rainfall NetCDF")
    print(f"Year: {year}")
    print(f"Official page: {SOURCE_PAGE}")
    print(f"Discovered request: {plan.method} {plan.url}")
    print(f"Form fields: {json.dumps(plan.fields, sort_keys=True)}")
    print(f"Destination: {output_path(year)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Exact observed rainfall year to discover/download; no year is assumed.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download after discovery. Without this flag the script prints the plan only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing raw file after verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1901 <= args.year <= datetime.now().year:
        raise ValueError(f"Year is outside the plausible IMD archive range: {args.year}")

    existing = output_path(args.year)
    if args.execute and existing.exists() and not args.overwrite:
        size, sha256 = validate_and_hash(existing)
        print(f"Already present and verified: {existing}")
        print(f"Bytes: {size}")
        print(f"SHA-256: {sha256}")
        print("No download was submitted and the immutable raw file was not changed.")
        return

    with build_session() as session:
        plan = discover_plan(session, args.year)
    print_plan(plan, args.year)

    if args.execute:
        download_year(plan, args.year, overwrite=args.overwrite)
    else:
        print("\nDry run only. Add --execute after reviewing the discovered request.")


if __name__ == "__main__":
    main()
