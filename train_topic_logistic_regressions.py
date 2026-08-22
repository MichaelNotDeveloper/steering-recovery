import logging

import hydra
from omegaconf import DictConfig

from steering_recovery.steering.ag_news_logistic import (
    train_ag_news_topic_logistic_regressions,
)


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="topic_logistic_regression",
)
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = train_ag_news_topic_logistic_regressions(config)
    logging.getLogger(__name__).info(
        "Topic logistic-regression training complete: %s", result
    )


if __name__ == "__main__":
    main()
