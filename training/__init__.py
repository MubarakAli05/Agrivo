"""Bounded, offline CPU training for the synthetic AgriTransformer baseline.

``train_model(root, dry_run=True)`` performs a tiny in-memory optimizer smoke
check. An explicit ``dry_run=False`` runs the configured bounded experiment;
the configuration's dry_run flag remains the recommended CLI default.
Checkpoints are inference snapshots, not optimizer-resume checkpoints.
"""
