import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from steering_recovery.steering.benchmarking import run_steering_benchmark


@hydra.main(version_base="1.3", config_path="configs", config_name="steering_benchmark")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    output_dir = HydraConfig.get().runtime.output_dir
    result = run_steering_benchmark(config, output_dir)
    logging.getLogger(__name__).info("Steering benchmark complete: %s", result)


if __name__ == "__main__":
    main()
