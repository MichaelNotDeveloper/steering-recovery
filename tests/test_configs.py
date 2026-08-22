from pathlib import Path

from hydra import compose, initialize_config_dir


def test_all_primary_configs_compose():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        denoiser = compose(config_name="denoiser")
        baseline = compose(config_name="baseline")
        cache = compose(config_name="cache_activations")
        statistics = compose(config_name="hidden_statistics")
        steering_vectors = compose(config_name="steering_vectors")
        topic_logistic = compose(config_name="topic_logistic_regression")
        steering_benchmark = compose(config_name="steering_benchmark")
    assert list(denoiser.model.latent_dims) == [192, 768, 3072]
    assert list(denoiser.model.num_layers) == [1, 3, 5]
    assert list(denoiser.model.sigmas) == [0.1, 0.2, 0.5]
    assert (
        len(denoiser.model.latent_dims)
        * len(denoiser.model.num_layers)
        * len(denoiser.model.sigmas)
        == 27
    )
    assert denoiser.data.mode == "streaming"
    assert denoiser.data.streaming.dataset_name == "Skylion007/openwebtext"
    assert denoiser.data.streaming.model_dtype == "float32"
    assert denoiser.training.precision == "fp32"
    assert baseline.steering.mode == "once_at_start"
    assert cache.capture.token_selection in {"last", "all"}
    assert statistics.collection.max_tokens > 0
    assert statistics.source.model_dtype == "float32"
    assert statistics.output_path.startswith("data/")
    assert steering_vectors.generator == "ag_news"
    assert steering_vectors.source.layer_index == 5
    assert steering_vectors.source.model_dtype == "float32"
    assert steering_vectors.extraction.mode == "full_text_all_tokens"
    assert steering_vectors.extraction.max_length == 1024
    assert steering_vectors.prompt.article_tokens == 48
    assert steering_vectors.prompt.prefix == "Article: "
    assert steering_vectors.prompt.suffix.endswith("about")
    assert steering_vectors.prompt.expected_last_token == "about"
    assert steering_vectors.collection.samples_per_topic == 1000
    assert steering_vectors.output_dir.startswith("data/")
    assert topic_logistic.source.model_dtype == "float32"
    assert topic_logistic.source.layer_index == 5
    assert topic_logistic.sampling.hidden_states_per_class == 100000
    assert topic_logistic.training.epochs > 1
    assert topic_logistic.training.l2_strength > 0
    assert topic_logistic.examples.per_class == 4
    assert topic_logistic.output_dir.startswith("data/")
    assert steering_benchmark.model.name == "gpt2"
    assert steering_benchmark.model.dtype == "float32"
    assert steering_benchmark.model.layer_index == 5
    assert list(steering_benchmark.alphas) == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert steering_benchmark.generation.samples_per_point == 100
    assert steering_benchmark.generation.prompt_tokens == 24
    assert steering_benchmark.generation.new_tokens == 40
    assert steering_benchmark.metrics.distinct_n == 3
    assert steering_benchmark.classifier.dtype == "float32"
    assert steering_benchmark.classifier.class_indices.world == 0


def test_sweep_config_registers_parameter_grid():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(
            config_name="denoiser",
            overrides=["experiment=denoiser_sweep"],
            return_hydra_config=True,
        )
    assert "training.learning_rate" in config.hydra.sweeper.params
