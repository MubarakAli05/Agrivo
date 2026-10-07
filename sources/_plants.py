"""Plant metadata normalization; images are never opened or downloaded."""

import re
from typing import Any

from sources._common import exact, identifier, image_path, rows, sha256, encoded, text, url

_BASE = {"record_id", "image_path", "image_sha256", "crop", "disease", "split",
         "image_source_url", "image_license", "rights_notes"}


def schema(source_id: str) -> dict[str, Any]:
    extra = ({"leaf_id", "variant"} if source_id == "plantvillage"
             else {"original_image_id", "annotation_id"})
    return {
        "formats": ["json"], "records_type": "nonempty array of objects",
        "required_columns": sorted(_BASE | extra), "additional_columns": False,
        "column_types": {
            **{key: "nonempty trimmed text" for key in _BASE | extra},
            "record_id": "unique identifier", "image_path": "safe relative path, metadata only",
            "image_sha256": "64 lowercase hex characters; upstream declaration, not verified against image bytes",
            "image_source_url": "HTTPS upstream image URL (not merely dataset URL)",
            "image_license": "per-image upstream license/rights declaration; preserved, never inferred from dataset license",
            "split": ["train", "validation", "test", "unsplit"],
            **({"leaf_id": "stable identity shared across all representations of a leaf",
                "variant": ["color", "grayscale", "segmented"]} if source_id == "plantvillage" else {
                "original_image_id": "stable upstream image identity shared across crops/annotations",
                "annotation_id": "unique annotation identifier"}),
        },
        "canonical_output": ["record_id", "crop", "disease", "split", "group_id", "image_path",
                             "image_sha256", "image_source_url", "image_license", "rights_notes",
                             "raw", "provenance"],
        "grouping": "group_id hashes [source_id,dataset_id,leaf_id/original_image_id], excluding version. A group, image path, URL or declared checksum cannot cross splits within a snapshot. Carry group_id into every downstream split.",
        "limitations": ["Metadata only; no image reads, classifier training or quality evaluation.",
                        "Upstream image rights and checksums remain unverified; dataset license is not an image-rights guarantee.",
                        "Group consistency is validated within one snapshot; downstream combination of releases must enforce group IDs and image identity across splits."],
    }


def normalize(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_id = snapshot["source_id"]
    if snapshot["format"] != "json":
        raise ValueError("Plant metadata requires JSON records")
    definition = schema(source_id)
    group_field = "leaf_id" if source_id == "plantvillage" else "original_image_id"
    output = []
    assignments: dict[tuple[str, str], tuple[str, str]] = {}
    annotations = set()
    for row in rows(snapshot["records"]):
        exact(row, set(definition["required_columns"]), source_id)
        for key in ("record_id", group_field):
            identifier(row[key], key)
        for key in ("crop", "disease", "image_license", "rights_notes"):
            text(row[key], key)
        url(row["image_source_url"], "image_source_url")
        path = image_path(row["image_path"])
        checksum = text(row["image_sha256"], "image_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("image_sha256 must be 64 lowercase hex characters")
        if row["split"] not in ("train", "validation", "test", "unsplit"):
            raise ValueError("Unknown plant split")
        if source_id == "plantvillage":
            if row["variant"] not in ("color", "grayscale", "segmented"):
                raise ValueError("Unknown PlantVillage variant")
        else:
            annotation = identifier(row["annotation_id"], "annotation_id")
            if annotation in annotations:
                raise ValueError("Duplicate annotation_id")
            annotations.add(annotation)
        group_id = sha256(encoded([source_id, snapshot["dataset_id"], row[group_field]]))
        identities = [("group", group_id), ("path", path.casefold()),
                      ("checksum", checksum), ("url", row["image_source_url"])]
        for identity in identities:
            assignment = (row["split"], group_id)
            if identity in assignments and assignments[identity] != assignment:
                raise ValueError("Leaf/image identity has inconsistent groups or splits (leakage)")
            assignments[identity] = assignment
        output.append({"record_id": row["record_id"], "crop": row["crop"],
                       "disease": row["disease"], "split": row["split"], "group_id": group_id,
                       "image_path": path, "image_sha256": checksum,
                       "image_source_url": row["image_source_url"],
                       "image_license": row["image_license"], "rights_notes": row["rights_notes"],
                       "raw": row})
    return output, [{"operation": "normalize_plant_metadata", "source": source_id},
                    {"operation": "validate_leaf_image_group_splits", "group_field": group_field},
                    {"operation": "preserve_upstream_image_rights_without_verification"}]
