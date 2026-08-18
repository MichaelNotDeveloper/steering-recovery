import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from steering_recovery.training import train_denoiser


@hydra.main(version_base="1.3", config_path="configs", config_name="denoiser")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = train_denoiser(config, HydraConfig.get().runtime.output_dir)
    logging.getLogger(__name__).info("Training complete: %s", result)


if __name__ == "__main__":
    main()
