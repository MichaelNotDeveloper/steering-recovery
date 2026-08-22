import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from steering_recovery.steering.epistemic.runner import run_epistemic_steering


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="epistemic_steering",
)
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_epistemic_steering(config, HydraConfig.get().runtime.output_dir)
    logging.getLogger(__name__).info(
        "Epistemic steering benchmark complete: %s", result
    )


if __name__ == "__main__":
    main()
