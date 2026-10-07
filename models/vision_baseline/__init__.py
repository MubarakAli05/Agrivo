"""Phase 14 synthetic pattern mechanics; no agricultural diagnosis capability.

Public workflow: vision_plan(root), check_vision(root, dry_run=False), and
validate_vision(root). diagnose_image always returns UNKNOWN. Reports/checkpoints
live in the ignored checkpoints/vision_baseline tree, never alongside real data.
"""

from .workflow import check_vision, diagnose_image, validate_vision, vision_plan, vision_report_path

__all__ = ["check_vision", "diagnose_image", "validate_vision", "vision_plan", "vision_report_path"]
