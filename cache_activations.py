import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from steering_recovery.cache import cache_activations


@hydra.main(version_base="1.3", config_path="configs", config_name="cache_activations")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = cache_activations(config, HydraConfig.get().runtime.output_dir)
    logging.getLogger(__name__).info("Caching complete: %s", result)


if __name__ == "__main__":
    main()
