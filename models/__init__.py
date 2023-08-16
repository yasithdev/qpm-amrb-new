import importlib

from config import Config
from models.base import BaseModel


def get_model(config: Config) -> BaseModel:
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
    model: BaseModel = Model(config)

    # return model
    return model
