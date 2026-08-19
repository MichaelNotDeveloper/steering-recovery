from pathlib import Path

from hydra import compose, initialize_config_dir


def test_all_primary_configs_compose():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        denoiser = compose(config_name="denoiser")
        baseline = compose(config_name="baseline")
        cache = compose(config_name="cache_activations")
    assert denoiser.model.depth > 0
    assert denoiser.data.mode == "streaming"
    assert denoiser.data.streaming.dataset_name == "Skylion007/openwebtext"
    assert baseline.steering.mode == "once_at_start"
    assert cache.capture.token_selection in {"last", "all"}


def test_sweep_config_registers_parameter_grid():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(
            config_name="denoiser",
            overrides=["experiment=denoiser_sweep"],
            return_hydra_config=True,
        )
    assert "model.depth" in config.hydra.sweeper.params
