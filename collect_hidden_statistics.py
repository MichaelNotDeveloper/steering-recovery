import logging

import hydra
from omegaconf import DictConfig

from steering_recovery.statistics import collect_hidden_statistics


@hydra.main(version_base="1.3", config_path="configs", config_name="hidden_statistics")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = collect_hidden_statistics(config)
    logging.getLogger(__name__).info("Statistics collection complete: %s", result)


if __name__ == "__main__":
    main()
