from types import SimpleNamespace

import torch
from torch import nn

from steering_recovery.steering.artifacts import save_steering_artifacts
from steering_recovery.steering.ag_news import ag_news_one_vs_rest_contrasts
from steering_recovery.steering.core import (
    AllTokenHiddenExtractor,
    ContrastDefinition,
    FullTextTokenBuilder,
    HiddenMoments,
    LabeledText,
    LastTokenHiddenExtractor,
    PromptTokenBuilder,
    TopicDefinition,
    collect_group_moments,
    collect_group_token_hiddens,
    collect_group_token_moments,
    compute_contrasts,
)
from steering_recovery.steering.logistic import (
    BalancedHiddenReservoir,
    EpochLogisticTrainer,
    binary_auc_metrics,
)
from steering_recovery.steering.logistic_reporting import (
    write_token_probability_report,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class NumericPromptBuilder:
    def __call__(self, text):
        return (int(text),)


class NumericExtractor:
    hidden_size = 2

    def __call__(self, token_sequences):
        return torch.tensor(
            [[tokens[0], tokens[0] * 10] for tokens in token_sequences],
            dtype=torch.float32,
        )


class NumericAllTokenExtractor:
    hidden_size = 2

    def __call__(self, token_sequences):
        return [
            torch.tensor([[token, token * 10] for token in tokens], dtype=torch.float32)
            for tokens in token_sequences
        ]


class NumericSequenceBuilder:
    def __call__(self, text):
        return tuple(int(value) for value in text.split(","))


class OffsetLayer(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, hidden):
        return hidden + self.offset


class TinyLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2, n_positions=16)
        self.h = nn.ModuleList([OffsetLayer(index + 1) for index in range(6)])

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
        for layer in self.h:
            hidden = layer(hidden)
        return hidden


def test_prompt_builder_limits_article_before_adding_template():
    builder = PromptTokenBuilder(
        CharacterTokenizer(),
        prefix="Article: ",
        suffix="\nThis article is mainly about",
        article_token_limit=48,
    )
    prompt = builder("a" * 100)
    assert prompt[: len(builder.prefix_ids)] == builder.prefix_ids
    assert (
        prompt[len(builder.prefix_ids) : len(builder.prefix_ids) + 48]
        == (ord("a"),) * 48
    )
    assert prompt[-len(builder.suffix_ids) :] == builder.suffix_ids
    assert builder.metadata()["capture_token"] == "t"


def test_last_token_extractor_captures_zero_based_layer_five():
    extractor = LastTokenHiddenExtractor(
        TinyLayerModel(),
        layer_index=5,
        layer_path="h",
        pad_token_id=99,
        device=torch.device("cpu"),
    )
    hidden = extractor([(1, 2, 3), (7,)])
    # h[0] ... h[5] add 1 + ... + 6 = 21 before the hook captures h[5].
    torch.testing.assert_close(hidden, torch.tensor([[24.0, 24.0], [28.0, 28.0]]))


def test_full_text_builder_and_extractor_capture_every_real_token():
    builder = FullTextTokenBuilder(CharacterTokenizer(), max_length=3)
    assert builder("abcd") == (ord("a"), ord("b"), ord("c"))
    assert builder.metadata()["capture_position"] == "all_non_padding_tokens"
    extractor = AllTokenHiddenExtractor(
        TinyLayerModel(),
        layer_index=5,
        layer_path="h",
        pad_token_id=99,
        device=torch.device("cpu"),
    )
    hidden = extractor([(1, 2, 3), (7,)])
    assert [tuple(chunk.shape) for chunk in hidden] == [(3, 2), (1, 2)]
    torch.testing.assert_close(
        hidden[0], torch.tensor([[22.0, 22.0], [23.0, 23.0], [24.0, 24.0]])
    )
    torch.testing.assert_close(hidden[1], torch.tensor([[28.0, 28.0]]))


def test_collection_enforces_exact_group_quotas_and_computes_contrast():
    examples = [
        LabeledText(1, "1"),
        LabeledText(1, "3"),
        LabeledText(1, "100"),
        LabeledText(2, "5"),
        LabeledText(2, "7"),
        LabeledText(2, "200"),
    ]
    moments = collect_group_moments(
        examples,
        prompt_builder=NumericPromptBuilder(),
        extractor=NumericExtractor(),
        target_counts={1: 2, 2: 2},
        batch_size=3,
    )
    assert moments[1].count == moments[2].count == 2
    torch.testing.assert_close(moments[1].mean, torch.tensor([2.0, 20.0]).double())
    torch.testing.assert_close(moments[2].mean, torch.tensor([6.0, 60.0]).double())

    results = compute_contrasts(
        moments,
        [ContrastDefinition("first", "first", (1,), (2,))],
    )
    torch.testing.assert_close(results[0].steering_vector, torch.tensor([-4.0, -40.0]))
    assert results[0].positive_count == results[0].negative_count == 2


def test_all_token_collection_uses_article_quotas_and_token_weighted_means():
    examples = [
        LabeledText(1, "1,3"),
        LabeledText(1, "100"),
        LabeledText(2, "5,7,9"),
        LabeledText(2, "200"),
    ]
    observed = []
    moments, article_counts = collect_group_token_moments(
        examples,
        token_builder=NumericSequenceBuilder(),
        extractor=NumericAllTokenExtractor(),
        target_articles={1: 1, 2: 1},
        batch_size=2,
        batch_observer=lambda chunks, labels: observed.append((chunks, labels)),
    )
    assert article_counts == {1: 1, 2: 1}
    assert moments[1].count == 2
    assert moments[2].count == 3
    torch.testing.assert_close(moments[1].mean, torch.tensor([2.0, 20.0]).double())
    torch.testing.assert_close(moments[2].mean, torch.tensor([7.0, 70.0]).double())
    assert len(observed) == 1


def test_observer_only_collection_does_not_require_moment_accumulation():
    observed = []
    counts = collect_group_token_hiddens(
        [LabeledText(1, "1,3"), LabeledText(2, "5,7,9")],
        token_builder=NumericSequenceBuilder(),
        extractor=NumericAllTokenExtractor(),
        target_articles={1: 1, 2: 1},
        batch_size=2,
        batch_observer=lambda chunks, labels: observed.append((chunks, labels)),
    )
    assert counts == {1: 1, 2: 1}
    assert len(observed) == 1
    assert [len(chunk) for chunk in observed[0][0]] == [2, 3]


def test_balanced_reservoir_and_epoch_logistic_regressions(tmp_path):
    topics = (
        TopicDefinition(1, "First", "first"),
        TopicDefinition(2, "Second", "second"),
    )
    reservoir = BalancedHiddenReservoir(
        hidden_size=2,
        topics=topics,
        samples_per_class=4,
        seed=7,
    )
    reservoir.update(
        [
            torch.tensor([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
            torch.tensor([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]),
        ],
        [1, 2],
    )
    features = reservoir.finalize()
    assert {label: len(values) for label, values in features.items()} == {1: 4, 2: 4}
    assert reservoir.metadata()["seen_hidden_states"] == {"first": 4, "second": 4}

    trainer = EpochLogisticTrainer(
        hidden_size=2,
        topics=topics,
        learning_rate=0.01,
        l2_strength=0.001,
        device=torch.device("cpu"),
        seed=7,
    )
    history = trainer.fit(
        features,
        epochs=2,
        batch_size_per_class=2,
        evaluation_batch_size=4,
    )
    artifacts = trainer.save(
        tmp_path,
        metadata={"source": {"layer_index": 5}},
        sampling=reservoir.metadata(),
        training={"epochs": 2, "batch_size_per_class": 2},
    )
    payload = torch.load(tmp_path / "logistic_regressions.pt", weights_only=True)
    assert payload["weights"].shape == (2, 2)
    torch.testing.assert_close(payload["steering_vectors"], payload["weights"])
    assert payload["bias"].shape == (2,)
    assert payload["training"]["epochs"] == 2
    assert payload["training"]["validation"] is False
    assert len(history) == 2
    assert 0 <= history[-1]["macro_roc_auc"] <= 1
    assert 0 <= history[-1]["macro_auc_prc"] <= 1
    assert artifacts["epochs"] == 2
    assert (tmp_path / "logistic_first.pt").is_file()
    assert (tmp_path / "logistic_second.pt").is_file()
    assert (tmp_path / "training_history.json").is_file()
    assert (tmp_path / "training_curves.png").is_file()
    first = torch.load(tmp_path / "logistic_first.pt", weights_only=True)
    torch.testing.assert_close(first["steering_vector"], first["weight"])


def test_auc_metrics_are_one_for_perfect_ranking():
    roc_auc, auc_prc = binary_auc_metrics(
        targets=torch.tensor([0, 1, 0, 1]).numpy(),
        scores=torch.tensor([0.1, 0.8, 0.2, 0.9]).numpy(),
    )
    assert roc_auc == 1.0
    assert auc_prc == 1.0


def test_token_probability_report_has_four_modes_and_escapes_text(tmp_path):
    topics = tuple(
        TopicDefinition(index, f"Topic {index}", f"topic_{index}")
        for index in range(1, 5)
    )
    examples = [
        {
            "id": "topic-01",
            "true_label": 1,
            "true_topic": "Topic 1",
            "truncated": False,
            "tokens": [
                {
                    "index": 0,
                    "id": 7,
                    "text": "</script><b>unsafe</b>",
                    "probabilities": [0.1, 0.2, 0.3, 0.4],
                }
            ],
            "metadata": {"source_text": "<unsafe>"},
        }
    ]
    report = write_token_probability_report(
        tmp_path / "report.html",
        topics=topics,
        examples=examples,
        metadata={"source": {"model_name": "gpt2", "layer_index": 5}},
    )
    html = report.read_text(encoding="utf-8")
    assert "Подсветка классификатора" in html
    assert "const DATA" in html
    assert "</script><b>unsafe</b>" not in html
    for topic in topics:
        assert topic.name in html


def test_ag_news_defines_four_balanced_one_vs_rest_contrasts():
    contrasts = ag_news_one_vs_rest_contrasts()
    assert [contrast.name for contrast in contrasts] == [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]
    assert all(len(contrast.positive_labels) == 1 for contrast in contrasts)
    assert all(len(contrast.negative_labels) == 3 for contrast in contrasts)


def test_artifacts_include_loadable_vectors_and_metadata(tmp_path):
    topics = (
        TopicDefinition(1, "first", "first"),
        TopicDefinition(2, "second", "second"),
    )
    moments = {
        1: HiddenMoments(torch.tensor([2.0, 4.0]), torch.zeros(2), 2),
        2: HiddenMoments(torch.tensor([8.0, 12.0]), torch.zeros(2), 2),
    }
    contrasts = compute_contrasts(
        moments,
        [ContrastDefinition("first", "first", (1,), (2,))],
    )
    manifest = save_steering_artifacts(
        tmp_path,
        topics=topics,
        moments=moments,
        contrasts=contrasts,
        metadata={"source": {"layer_index": 5}},
    )
    vector_payload = torch.load(tmp_path / "first.pt", weights_only=True)
    combined_payload = torch.load(tmp_path / "steering_vectors.pt", weights_only=True)
    torch.testing.assert_close(
        vector_payload["steering_vector"], torch.tensor([-3.0, -4.0])
    )
    assert vector_payload["metadata"]["source"]["layer_index"] == 5
    assert combined_payload["steering_vectors"].shape == (1, 2)
    assert manifest["vectors"][0]["positive_count"] == 2
