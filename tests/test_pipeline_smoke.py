"""End-to-end smoke test on synthetic data (classical tasks only)."""

from __future__ import annotations

from pathlib import Path

from vision_robustness.config import Config
from vision_robustness.data import SyntheticDataset
from vision_robustness.pipeline import EvaluationPipeline
from vision_robustness.tasks.edges import CannyEdgesTask
from vision_robustness.tasks.features import ORBFeaturesTask


def test_pipeline_clean_distorted_enhanced(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root / "configs" / "smoke.yaml")
    # Redirect outputs into pytest tmp
    cfg.raw["paths"]["results_dir"] = str(tmp_path / "results")
    cfg.raw["paths"]["distorted_dir"] = str(tmp_path / "distorted")
    cfg.raw["paths"]["enhanced_dir"] = str(tmp_path / "enhanced")
    cfg.raw["paths"]["figures_dir"] = str(tmp_path / "figures")
    cfg.ensure_dirs()

    ds = SyntheticDataset(n=2, size=(96, 96), seed=0)
    pipe = EvaluationPipeline(
        cfg,
        tasks={
            "features": ORBFeaturesTask(n_features=200),
            "edges": CannyEdgesTask(),
        },
    )

    clean = pipe.run_clean(ds, save_visuals=False)
    assert not clean.empty
    assert set(clean["stage"]) == {"clean"}

    distorted = pipe.run_distorted(ds, save_images=True, save_visuals=False)
    assert "snr_db" in distorted.columns
    assert distorted["distortion"].nunique() == 4

    enhanced = pipe.run_enhanced(ds, save_images=True, save_visuals=False)
    assert set(enhanced["stage"]) == {"enhanced"}
    assert (tmp_path / "results" / "clean" / "metrics.csv").exists()
    assert (tmp_path / "results" / "distorted" / "metrics.csv").exists()
    assert (tmp_path / "results" / "enhanced" / "metrics.csv").exists()
