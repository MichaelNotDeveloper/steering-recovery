import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from steering_recovery.baseline import run_baseline


@hydra.main(version_base="1.3", config_path="configs", config_name="baseline")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_baseline(config, HydraConfig.get().runtime.output_dir)
    logging.getLogger(__name__).info("Baseline complete: %s", result)


if __name__ == "__main__":
    main()
