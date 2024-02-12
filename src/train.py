import logging

import autorootcwd  # noqa: F401
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# import warnings
# warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs/training", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main function for training a model using the specified configuration.

    Parameters:
        cfg (DictConfig): The configuration for training.
    """
    logger.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(cfg.trainer)

    logger.info("Instantiating param distributions")
    param_distributions = hydra.utils.instantiate(cfg.param_distributions)

    logger.info("Instantiating search strategy")
    search_strategy = hydra.utils.instantiate(cfg.search_strategy)

    logger.info("Instantiating WandB")
    wandb.init(
        dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    config_dict = {
        "trainer": cfg_dict["trainer"],
        "search_strategy": cfg_dict["search_strategy"],
        "param_distributions": cfg_dict["param_distributions"],
        "task_name": cfg_dict["task_name"],
        "run_name": cfg_dict["run_name"],
    }

    opt_score, test_score, hparams = trainer.fit_cv(
        model_name=cfg.model_name,
        search_strategy=search_strategy,
        param_distributions=param_distributions,
        logger=logger,
    )

    config_dict["hparams"] = hparams
    wandb.config.update(config_dict)

    wandb.log(opt_score)
    wandb.log(test_score)


if __name__ == "__main__":
    main()
