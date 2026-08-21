import logging

import hydra
from omegaconf import DictConfig

from steering_recovery.steering import generate_steering_vectors


@hydra.main(version_base="1.3", config_path="configs", config_name="steering_vectors")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    result = generate_steering_vectors(config)
    logging.getLogger(__name__).info("Steering-vector generation complete: %s", result)


if __name__ == "__main__":
    main()
