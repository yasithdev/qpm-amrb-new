import importlib

import lightning.pytorch as pl

from config import Config


def get_model(config: Config) -> pl.LightningModule:
    assert config.image_chw

    # load requested module, if available
    try:
        module = importlib.__import__(
            name=f"model_{config.model_name}",
            globals=globals(),
            level=1,
        )
    except ImportError as e:
        raise e

    # instantiate model
    Model = getattr(module, "Model")
    model: pl.LightningModule = Model(config)

    # return model
    return model
