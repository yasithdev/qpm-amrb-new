import importlib

from config import Config
from models.base import BaseModel


def get_model(config: Config) -> BaseModel:
    assert config.image_chw

    # VARIANTS
    
    # resnet variants
    if config.model_name == "resnet_ce_mse":
        from .model_resnet import Model
        return Model(config, with_decoder=True, classifier_loss="crossent", decoder_loss="mse")
    elif config.model_name == "resnet_ce":
        from .model_resnet import Model
        return Model(config, with_decoder=False, classifier_loss="crossent", decoder_loss="N/A")
    elif config.model_name == "resnet_edl_mse":
        from .model_resnet import Model
        return Model(config, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    elif config.model_name == "resnet_edl":
        from .model_resnet import Model
        return Model(config, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
    
    # resnet18 variants
    elif config.model_name == "resnet18_ce":
        from .model_resnet18 import Model
        return Model(config, classifier_loss="crossent")
    elif config.model_name == "resnet18_edl":
        from .model_resnet18 import Model
        return Model(config, classifier_loss="edl")
    
    # rescaps variants
    elif config.model_name == "rescaps_margin_mse":
        from .model_rescaps import Model
        return Model(config, with_decoder=True, classifier_loss="margin", decoder_loss="mse")
    elif config.model_name == "rescaps_margin":
        from .model_rescaps import Model
        return Model(config, with_decoder=False, classifier_loss="margin", decoder_loss="N/A")
    elif config.model_name == "rescaps_edl_mse":
        from .model_rescaps import Model
        return Model(config, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    elif config.model_name == "rescaps_edl":
        from .model_rescaps import Model
        return Model(config, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
    
    # flow variants
    elif config.model_name == "flow_ce_mse":
        from .model_flow import Model
        return Model(config, with_classifier=True, classifier_loss="crossent", decoder_loss="mse")
    elif config.model_name == "flow_edl_mse":
        from .model_flow import Model
        return Model(config, with_classifier=True, classifier_loss="edl", decoder_loss="mse")
    elif config.model_name == "flow_mse":
        from .model_flow import Model
        return Model(config, with_classifier=False, classifier_loss="N/A", decoder_loss="mse")

    # DEFAULTS

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
