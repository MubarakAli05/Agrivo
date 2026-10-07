"""Validated source/license registry and fail-closed ingestion eligibility.

Inspection here reads the stored metadata snapshot; it never makes network requests.
"""

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

import yaml

from agri.source_catalog import default_registries


SOURCE_PATH = Path("data") / "source_registry.yaml"
LICENSE_PATH = Path("licenses") / "registry.json"
PURPOSES = ("research_training", "commercial_training", "redistribution")


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_pairs(pairs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("Registry keys must be unique strings")
        result[key] = value
    return result


def _mapping(loader: UniqueSafeLoader, node: Any) -> dict[str, Any]:
    return _unique_pairs(loader.construct_pairs(node, deep=True))


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path, *, as_yaml: bool = False) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        result = (yaml.load(text, Loader=UniqueSafeLoader) if as_yaml
                  else json.loads(text, object_pairs_hook=_unique_pairs))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid registry YAML: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Registry must be an object: {path}")
    return result


def validate_registries(sources: Any, licenses: Any) -> None:
    def keys(value: Any, expected: set[str], label: str) -> None:
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"{label}: expected keys {sorted(expected)}")

    def text(value: Any, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: expected nonempty text")

    def choice(value: Any, options: tuple[str, ...], label: str) -> None:
        if not isinstance(value, str) or value not in options:
            raise ValueError(f"{label}: expected one of {options}")

    def url(value: Any) -> None:
        text(value, "URL")
        parts = urlsplit(value)
        if (parts.scheme != "https" or not parts.hostname or parts.username
                or parts.password or any(char.isspace() for char in value)):
            raise ValueError("Registry URLs must be HTTPS and contain no credentials/whitespace")

    def urls(value: Any, *, required: bool = True) -> None:
        if not isinstance(value, list) or (required and not value):
            raise ValueError("Evidence URLs must be a list")
        for item in value:
            url(item)

    def day(value: Any) -> None:
        text(value, "Review date")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Review date must be YYYY-MM-DD")
        date.fromisoformat(value)

    keys(sources, {"schema_version", "registry_version", "sources"}, "sources")
    keys(licenses, {"schema_version", "licenses"}, "licenses")
    for registry in (sources, licenses):
        if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
            raise ValueError("Unsupported registry schema_version")
    text(sources["registry_version"], "registry_version")
    entries = sources["sources"]
    terms = licenses["licenses"]
    if not isinstance(entries, dict) or not entries or not isinstance(terms, dict):
        raise ValueError("Source and license mappings are required")
    if set(entries) != set(terms):
        raise ValueError("Every source must have exactly one matching license record")
    for source_id, entry in entries.items():
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", source_id):
            raise ValueError("Invalid source ID")
        keys(entry, {"name", "type", "url", "enabled", "priority", "access_modes",
                     "source_version", "inspection"}, source_id)
        text(entry["name"], "name")
        choice(entry["type"], ("catalog", "soil_geospatial", "soil_survey", "plant_vision"), "type")
        url(entry["url"])
        if type(entry["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        if type(entry["priority"]) is not int or not 1 <= entry["priority"] <= 5:
            raise ValueError("priority must be an integer in [1, 5]")
        if entry["source_version"] is not None:
            text(entry["source_version"], "source_version")
        if not isinstance(entry["access_modes"], list):
            raise ValueError("access_modes must be a list")
        modes = set()
        for access in entry["access_modes"]:
            keys(access, {"mode", "url", "status", "evidence_url"}, "access mode")
            choice(access["mode"], ("rest", "webdav", "wcs", "wms", "git", "download", "catalog", "competition"), "mode")
            if access["mode"] in modes:
                raise ValueError("Duplicate access mode")
            modes.add(access["mode"])
            choice(access["status"], ("documented", "unverified", "paused"), "access status")
            url(access["url"])
            url(access["evidence_url"])
        inspection = entry["inspection"]
        keys(inspection, {"status", "inspected_on", "evidence_urls", "notes"}, "inspection")
        choice(inspection["status"], ("reviewed", "restricted", "unavailable"), "inspection status")
        day(inspection["inspected_on"])
        urls(inspection["evidence_urls"])
        text(inspection["notes"], "inspection notes")

        license_record = terms[source_id]
        keys(license_record, {"source_url", "license", "status", "scope", "license_url",
                             "evidence_urls", "attribution_requirement", "commercial_use",
                             "redistribution", "notes", "approval"}, "license record")
        if license_record["source_url"] != entry["url"]:
            raise ValueError("License source URL must match the source registry")
        choice(license_record["status"], ("declared", "unresolved"), "license status")
        choice(license_record["scope"], ("dataset", "dataset_distribution", "unknown"), "license scope")
        for key in ("commercial_use", "redistribution"):
            choice(license_record[key], ("allowed_with_conditions", "prohibited", "unknown"), key)
        text(license_record["notes"], "license notes")
        urls(license_record["evidence_urls"], required=False)
        if license_record["status"] == "declared":
            text(license_record["license"], "license")
            url(license_record["license_url"])
            urls(license_record["evidence_urls"])
            text(license_record["attribution_requirement"], "attribution_requirement")
            if license_record["scope"] == "unknown":
                raise ValueError("Declared licenses require an explicit scope")
        else:
            if (license_record["license"] is not None or license_record["license_url"] is not None
                    or license_record["scope"] != "unknown"
                    or license_record["commercial_use"] != "unknown"
                    or license_record["redistribution"] != "unknown"
                    or license_record["attribution_requirement"] is not None):
                raise ValueError("Unresolved licenses must not claim known terms or permissions")
        approval = license_record["approval"]
        keys(approval, {"status", "intended_use", "reviewer", "reviewed_on"}, "approval")
        choice(approval["status"], ("pending", "approved", "rejected"), "approval status")
        if approval["status"] == "pending":
            if any(approval[key] is not None for key in ("intended_use", "reviewer", "reviewed_on")):
                raise ValueError("Pending approval must not contain a completed review")
        else:
            choice(approval["intended_use"], PURPOSES, "intended_use")
            text(approval["reviewer"], "reviewer")
            day(approval["reviewed_on"])
            if approval["status"] == "approved" and license_record["status"] != "declared":
                raise ValueError("Cannot approve unresolved license terms")


def load_registries(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = _read(root / SOURCE_PATH, as_yaml=True)
    licenses = _read(root / LICENSE_PATH)
    validate_registries(sources, licenses)
    return sources, licenses


def initialize_registries(root: Path) -> None:
    source_path, license_path = root / SOURCE_PATH, root / LICENSE_PATH
    sources, licenses = default_registries()
    if source_path.exists():
        sources = _read(source_path, as_yaml=True)
    if license_path.exists():
        licenses = _read(license_path)
    validate_registries(sources, licenses)
    for path, payload in ((source_path, sources), (license_path, licenses)):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                if path == source_path:
                    yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
                else:
                    json.dump(payload, stream, indent=2, allow_nan=False)
                    stream.write("\n")


def _blockers(source: dict[str, Any], license_record: dict[str, Any], purpose: str) -> list[str]:
    reasons = []
    if not source["enabled"]:
        reasons.append("Source disabled")
    if source["type"] == "catalog":
        reasons.append("Catalog only: register and review a specific dataset")
    if source["inspection"]["status"] != "reviewed":
        reasons.append("Source inspection incomplete or unavailable")
    if not any(access["status"] == "documented" and access["mode"] not in ("wms", "catalog")
               for access in source["access_modes"]):
        reasons.append("No documented data acquisition mode")
    if license_record["status"] != "declared":
        reasons.append("Dataset license unresolved")
    approval = license_record["approval"]
    if approval["status"] != "approved" or approval["intended_use"] != purpose:
        reasons.append(f"Intended use not approved: {purpose}")
    if purpose == "commercial_training" and license_record["commercial_use"] != "allowed_with_conditions":
        reasons.append("Commercial-use permission not established")
    if purpose == "redistribution" and license_record["redistribution"] != "allowed_with_conditions":
        reasons.append("Redistribution permission not established")
    return reasons


def require_ingestion_approval(root: Path, source_id: str, purpose: str) -> dict[str, Any]:
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown intended use: {purpose}")
    sources, licenses = load_registries(root)
    if source_id not in sources["sources"]:
        raise ValueError(f"Unknown source: {source_id}")
    source = sources["sources"][source_id]
    reasons = _blockers(source, licenses["licenses"][source_id], purpose)
    if reasons:
        raise PermissionError("; ".join(reasons))
    return deepcopy(source)


def inspect_sources(root: Path, source_id: str | None = None) -> dict[str, Any]:
    sources, licenses = load_registries(root)
    if source_id is not None and source_id not in sources["sources"]:
        raise ValueError(f"Unknown source: {source_id}")
    entries = {}
    for key, source in sources["sources"].items():
        if source_id is not None and key != source_id:
            continue
        terms = licenses["licenses"][key]
        blockers = {purpose: _blockers(source, terms, purpose) for purpose in PURPOSES}
        entries[key] = {
            **source, "license": terms,
            "ingestion": {purpose: {"allowed": not reasons, "blockers": reasons}
                          for purpose, reasons in blockers.items()},
            "license_gate": ("RED" if terms["status"] == "unresolved" or terms["approval"]["status"] == "rejected"
                             else "GREEN" if terms["approval"]["status"] == "approved" else "YELLOW"),
        }
    ready = all(any(gate["allowed"] for gate in entry["ingestion"].values()) for entry in entries.values())
    return {
        "phase": 2, "status": "GREEN" if ready else "YELLOW", "registry_valid": True,
        "scope": "Stored source metadata and license review; not live availability or ingestion",
        "registry_version": sources["registry_version"],
        "registry_sha256": hashlib.sha256((root / SOURCE_PATH).read_bytes()).hexdigest(),
        "license_registry_sha256": hashlib.sha256((root / LICENSE_PATH).read_bytes()).hexdigest(),
        "source_count": len(entries),
        "declared_license_count": sum(e["license"]["status"] == "declared" for e in entries.values()),
        "unresolved_license_count": sum(e["license"]["status"] == "unresolved" for e in entries.values()),
        "sources": entries,
        "next_required_step": "Review Phase 2, then Phase 3: tiny synthetic dataset; external ingestion remains gated",
    }
