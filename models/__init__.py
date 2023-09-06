import importlib
from config import Config

from models.base import BaseModel

def get_model(
    model_name: str,
    emb_dims: int,
    optim_lr: float,
    input_shape: tuple[int, int, int],
    labels: list[str],
    cat_k: int,
    opt: Config,
) -> BaseModel:

    args: dict = dict(
        emb_dims=emb_dims,
        optim_lr=optim_lr,
        input_shape=input_shape,
        labels=labels,
        cat_k=cat_k,
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
    elif model_name == "resnet50_simclr":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=False, encoder_loss="simclr", classifier_loss="N/A", opt=opt)
    elif model_name == "resnet50_vicreg":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=False, encoder_loss="vicreg", classifier_loss="N/A", opt=opt)
    elif model_name == "resnet50_ce":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=True, encoder_loss="N/A", classifier_loss="crossent", opt=opt)
    
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

    # fisher exact
    elif model_name == "fisher_exact_ce":
        assert opt.groups is not None
        assert opt.cat_k is not None
        assert opt.group_fn is not None
        from .model_fisher_exact import Model
        return Model(
            labels=labels,
            cat_k=opt.cat_k,
            grp_k=len(opt.groups),
            grp_fn=opt.group_fn,
            in_dims=opt.emb_dims,
            rand_perms=opt.rand_perms,
            optim_lr=opt.optim_lr,
        )

    # DEFAULTS
    
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
