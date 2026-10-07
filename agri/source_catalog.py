"""Bootstrap metadata from a manual official-page inspection on 2026-10-07.

These are source descriptions, not imported datasets or ingestion permissions.
"""

from typing import Any


INSPECTED_ON = "2026-10-07"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/3.0/"
SOIL_DOCS = "https://docs.isric.org/globaldata/soilgrids/index.html"
PV_CARD = "https://github.com/spMohanty/PlantVillage-Dataset/blob/master/README_HF.md"
PD_LICENSE = "https://github.com/pratikkayal/PlantDoc-Dataset/blob/master/LICENSE.txt"


def default_registries() -> tuple[dict[str, Any], dict[str, Any]]:
    sources: dict[str, Any] = {}
    licenses: dict[str, Any] = {}

    def add(source_id: str, name: str, source_type: str, url: str,
            modes: list[dict[str, str]], notes: str, *, priority: int = 3,
            inspection: str = "reviewed", evidence: list[str] | None = None,
            license_name: str | None = None, license_url: str | None = None,
            license_evidence: list[str] | None = None,
            license_scope: str = "unknown", attribution: str | None = None,
            license_notes: str = "Dataset-specific terms require review.") -> None:
        sources[source_id] = {
            "name": name, "type": source_type, "url": url,
            "enabled": False, "priority": priority, "access_modes": modes,
            "source_version": None,
            "inspection": {
                "status": inspection, "inspected_on": INSPECTED_ON,
                "evidence_urls": evidence or [url], "notes": notes,
            },
        }
        licenses[source_id] = {
            "source_url": url, "license": license_name,
            "status": "declared" if license_name else "unresolved",
            "scope": license_scope, "license_url": license_url,
            "evidence_urls": license_evidence or [],
            "attribution_requirement": attribution,
            "commercial_use": "allowed_with_conditions" if license_name else "unknown",
            "redistribution": "allowed_with_conditions" if license_name else "unknown",
            "notes": license_notes,
            "approval": {
                "status": "pending", "intended_use": None,
                "reviewer": None, "reviewed_on": None,
            },
        }

    def mode(name: str, url: str, evidence: str, state: str = "documented") -> dict[str, str]:
        return {"mode": name, "url": url, "status": state, "evidence_url": evidence}

    soil_url = "https://isric.org/explore/soilgrids"
    add("soilgrids", "ISRIC SoilGrids", "soil_geospatial", soil_url, [
        mode("rest", "https://rest.isric.org/", SOIL_DOCS, "paused"),
        mode("webdav", "https://files.isric.org/soilgrids/latest/data/", SOIL_DOCS),
        mode("wcs", "https://maps.isric.org/", SOIL_DOCS),
        mode("wms", "https://maps.isric.org/", SOIL_DOCS),
    ], "Official documentation reports REST temporarily paused with no restoration date. "
       "WCS/WebDAV are alternatives; endpoints were not queried. Maps are estimates, "
       "not local laboratory measurements. A latest URL is not a pinned data version.",
        evidence=[soil_url, SOIL_DOCS], license_name="CC-BY-4.0", license_url=CC_BY,
        license_evidence=[soil_url, SOIL_DOCS, CC_BY], license_scope="dataset",
        attribution="Credit ISRIC and the source, link CC-BY-4.0, and indicate changes.",
        license_notes="Map license is declared by ISRIC. Intended-use review and snapshot pinning remain required.")

    url = "https://isric.org/explore"
    add("isric", "ISRIC Data Resources", "catalog", url,
        [mode("catalog", url, url)],
        "Resource catalog containing distinct datasets; do not inherit the SoilGrids license for all ISRIC resources.")

    url = "https://www.nrcs.usda.gov/resources/data-and-reports/ssurgo-portal"
    add("ssurgo", "USDA NRCS SSURGO", "soil_survey", url, [
        mode("download", "https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx", url),
    ], "Portal documents county/AOI spatial and tabular downloads, annual refreshes, and incomplete survey areas. "
       "The portal application's no-license/no-account statement is not a dataset usage license.",
        priority=1,
        license_notes="No dataset-specific license verified from the inspected portal. Do not infer rights from federal hosting.")

    url = "https://data.gov.in/"
    terms = "https://www.data.gov.in/government-open-data-license-india"
    add("data_gov", "Government Open Data India", "catalog", url, [
        mode("catalog", url, url, "unverified"),
    ], "Landing page and license page returned HTTP 403 to this inspection. "
       "Resource-specific API/download mechanisms and credentials remain unverified.",
        priority=1, inspection="restricted", evidence=[url, terms],
        license_notes="Terms could not be read (HTTP 403); review the exact resource and intended use before approval.")

    url = "https://github.com/spMohanty/PlantVillage-Dataset"
    add("plantvillage", "PlantVillage", "plant_vision", url, [mode("git", url, url)],
        "Repository documents raw color/grayscale/segmented data and leaf grouping. "
        "Preserve leaf identity when splitting. Dataset-card license declaration applies to the "
        "described distribution; raw images have not been audited. No repository commit is pinned.",
        evidence=[url, PV_CARD], license_name="CC-BY-SA-3.0", license_url=CC_BY_SA,
        license_evidence=[PV_CARD, CC_BY_SA], license_scope="dataset_distribution",
        attribution="Credit dataset authors, link CC-BY-SA-3.0, indicate changes; ShareAlike applies to adaptations.",
        license_notes="Repository README_HF.md declares cc-by-sa-3.0 (observed blob SHA "
        "e6add25de14e52cc3261690fe7d37d2da21ee71a). Confirm scope for the chosen files and intended use.")

    url = "https://github.com/pratikkayal/PlantDoc-Dataset"
    add("plantdoc", "PlantDoc", "plant_vision", url, [mode("git", url, url)],
        "Repository describes cropped classification data and links a separate object-detection dataset. "
        "Images are described as internet-scraped; upstream image rights need review.",
        priority=2, evidence=[url, PD_LICENSE], license_name="CC-BY-4.0", license_url=CC_BY,
        license_evidence=[url, PD_LICENSE, CC_BY], license_scope="dataset",
        attribution="Credit PlantDoc authors, link CC-BY-4.0, and indicate changes.",
        license_notes="Dataset README and LICENSE.txt declare CC-BY-4.0. This is not a guarantee of upstream image rights.")

    for year, suffix in [(2020, "fgvc7"), (2021, "fgvc8")]:
        slug = f"plant-pathology-{year}-{suffix}"
        url = f"https://www.kaggle.com/c/{slug}"
        landing = f"https://www.kaggle.com/competitions/{slug}"
        rules = landing + "/rules"
        add(f"plant_pathology_{year}", f"Plant Pathology {year} {suffix.upper()}",
            "plant_vision", url, [mode("competition", landing, landing, "unverified")],
            "Landing/rules pages returned only a title in this inspection. "
            "Competition terms, access credentials, files, and data version remain unverified.",
            priority=4, inspection="restricted", evidence=[landing, rules],
            license_notes="Competition-specific rules must be readable and reviewed; public title is not usage permission.")

    url = "https://github.com/AIChallenger/AIChallenger/tree/master/agriculture"
    add("ai_challenger", "AI Challenger agriculture", "plant_vision", url, [],
        "User-specified agriculture URL returned HTTP 404. No replacement source was assumed.",
        priority=5, inspection="unavailable",
        license_notes="Original source unavailable; neither data access nor license was verified.")

    url = "https://www.kaggle.com/datasets?search=agriculture+plant+disease"
    add("kaggle_agriculture", "Kaggle agriculture/plant-disease discovery", "catalog", url,
        [mode("catalog", url, url, "unverified")],
        "Search page returned only a title. This is a discovery catalog, not an approved dataset. "
        "Each selected dataset needs its own source ID, version, and license review.",
        priority=5, inspection="restricted")

    return (
        {"schema_version": 1, "registry_version": "2026-10-07.1", "sources": sources},
        {"schema_version": 1, "licenses": licenses},
    )
