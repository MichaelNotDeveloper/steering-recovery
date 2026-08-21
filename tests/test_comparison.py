import json

from matplotlib.axes import Axes

from steering_recovery.comparison import write_comparison


def test_comparison_writes_table_and_barplot(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    scales: list[str] = []
    identity_ticks: list[tuple[list[float], str]] = []
    original_set_xscale = Axes.set_xscale
    original_vlines = Axes.vlines

    def record_xscale(self, value, *args, **kwargs):
        scales.append(value)
        return original_set_xscale(self, value, *args, **kwargs)

    def record_vlines(self, x, ymin, ymax, *args, **kwargs):
        identity_ticks.append((list(x), kwargs.get("label", "")))
        return original_vlines(self, x, ymin, ymax, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xscale", record_xscale)
    monkeypatch.setattr(Axes, "vlines", record_vlines)
    run = tmp_path / "run" / "models"
    for index, sigma in enumerate((0.1, 0.2)):
        directory = run / f"model-{index}"
        directory.mkdir(parents=True)
        payload = {
            "name": f"model-{index}",
            "parameters": {
                "latent_dim": 192,
                "num_layers": index + 1,
                "sigma": sigma,
            },
            "best_step": 10,
            "best_validation": {
                "l2": 0.02 + index * 0.01,
                "rmse": 0.14 + index * 0.01,
                "cosine_distance": 0.03 + index * 0.01,
                "noisy_l2": sigma**2,
                "noisy_rmse": sigma,
                "noisy_cosine_distance": 0.04 + index * 0.01,
                "score_mse": 4.0 + index,
                "score_rms": (4.0 + index) ** 0.5,
            },
        }
        (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "comparison"
    result = write_comparison(tmp_path / "run", output)

    assert result["models"] == 2
    assert result["score_models"] == 2
    assert (output / "denoiser_comparison.csv").is_file()
    assert (output / "denoiser_comparison.md").is_file()
    assert (output / "denoiser_comparison.png").stat().st_size > 0
    assert set(result["plots_by_sigma"]) == {"0.1", "0.2"}
    assert (output / "denoiser_comparison_sigma_0p1.png").stat().st_size > 0
    assert (output / "denoiser_comparison_sigma_0p2.png").stat().st_size > 0
    assert scales == ["log"] * 12
    assert len(identity_ticks) == 12
    assert identity_ticks[0][0] == [0.1**2, 0.2**2]
    assert identity_ticks[0][1] == "Identity f(y)=y"
    assert identity_ticks[3][1] == "Identity f(y)=y: 0 (shown at log floor)"
