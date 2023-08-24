import importlib

from models.base import BaseModel

import argparse

def get_model(
    image_chw: tuple[int, int, int],
    model_name: str,
    labels: list[str],
    cat_k: int,
    manifold_d: int,
    optim_lr: float,
    _args: argparse.Namespace,
) -> BaseModel:

    args: dict = dict(
        labels=labels,
        cat_k=cat_k,
        manifold_d=manifold_d,
        image_chw=image_chw,
        optim_lr=optim_lr,
    )

    # VARIANTS
    
    # resnet variants
    if model_name == "resnet_ce_mse":
        from .model_resnet import Model
        return Model(**args, with_decoder=True, classifier_loss="crossent", decoder_loss="mse")
    elif model_name == "resnet_ce":
        from .model_resnet import Model
        return Model(**args, with_decoder=False, classifier_loss="crossent", decoder_loss="N/A")
    elif model_name == "resnet_edl_mse":
        from .model_resnet import Model
        return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    elif model_name == "resnet_edl":
        from .model_resnet import Model
        return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
    
    # resnet18 variants
    elif model_name == "resnet18_ce":
        from .model_resnet18 import Model
        return Model(**args, classifier_loss="crossent")
    elif model_name == "resnet18_edl":
        from .model_resnet18 import Model
        return Model(**args, classifier_loss="edl")
    
    # resnet50 variants
    elif model_name == "resnet50_ssl":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=False, encoder_loss="simclr", temperature=_args.temperature)
    elif model_name == "resnet50_ce":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=True, classifier_loss="crossent", temperature=_args.temperature)
    
    # rescaps variants
    elif model_name == "rescaps_margin_mse":
        from .model_rescaps import Model
        return Model(**args, with_decoder=True, classifier_loss="margin", decoder_loss="mse")
    elif model_name == "rescaps_margin":
        from .model_rescaps import Model
        return Model(**args, with_decoder=False, classifier_loss="margin", decoder_loss="N/A")
    elif model_name == "rescaps_edl_mse":
        from .model_rescaps import Model
        return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    elif model_name == "rescaps_edl":
        from .model_rescaps import Model
        return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
    
    # flow variants
    elif model_name == "flow_ce_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=True, classifier_loss="crossent", decoder_loss="mse")
    elif model_name == "flow_edl_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=True, classifier_loss="edl", decoder_loss="mse")
    elif model_name == "flow_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=False, classifier_loss="N/A", decoder_loss="mse")

    # DEFAULTS

    # load requested module, if available
    try:
        module = importlib.__import__(
            name=f"model_{model_name}",
            globals=globals(),
            level=1,
        )
    except ImportError as e:
        raise e

    # instantiate model
    Model = getattr(module, "Model")
    model: BaseModel = Model(**args)

    # return model
    return model
