import json

from steering_recovery.comparison import write_comparison


def test_comparison_writes_table_and_barplot(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
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
            },
        }
        (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "comparison"
    result = write_comparison(tmp_path / "run", output)

    assert result["models"] == 2
    assert (output / "denoiser_comparison.csv").is_file()
    assert (output / "denoiser_comparison.md").is_file()
    assert (output / "denoiser_comparison.png").stat().st_size > 0
