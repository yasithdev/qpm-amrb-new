import importlib

from config import Config
from models.base import BaseModel
from .common import generate_rand_perms


def get_model(
    model_name: str,
    emb_dims: int,
    optim_lr: float,
    input_shape: tuple[int, ...],
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
    if model_name == "resnet_ce":
        from .model_resnet import Model
        return Model(**args, with_decoder=False, classifier_loss="crossent", decoder_loss="N/A")
    if model_name == "resnet_edl_mse":
        from .model_resnet import Model
        return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    if model_name == "resnet_edl":
        from .model_resnet import Model
        return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")

    # resnet18 variants
    if model_name == "resnet18_ce":
        from .model_resnet18 import Model
        return Model(**args, classifier_loss="crossent")
    if model_name == "resnet18_edl":
        from .model_resnet18 import Model
        return Model(**args, classifier_loss="edl")

    # resnet50 variants
    if model_name == "resnet50_simclr":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=False, encoder_loss="simclr", classifier_loss="N/A", opt=opt)
    if model_name == "resnet50_vicreg":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=False, encoder_loss="vicreg", classifier_loss="N/A", opt=opt)
    if model_name == "resnet50_ce":
        from .model_resnet50 import Model
        return Model(**args, with_classifier=True, encoder_loss="N/A", classifier_loss="crossent", opt=opt)

    # rescaps variants
    if model_name == "rescaps_margin_mse":
        from .model_rescaps import Model
        return Model(**args, with_decoder=True, classifier_loss="margin", decoder_loss="mse")
    if model_name == "rescaps_margin":
        from .model_rescaps import Model
        return Model(**args, with_decoder=False, classifier_loss="margin", decoder_loss="N/A")
    if model_name == "rescaps_edl_mse":
        from .model_rescaps import Model
        return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    if model_name == "rescaps_edl":
        from .model_rescaps import Model
        return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")

    # flow variants
    if model_name == "flow_ce_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=True, classifier_loss="crossent", decoder_loss="mse")
    if model_name == "flow_edl_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=True, classifier_loss="edl", decoder_loss="mse")
    if model_name == "flow_mse":
        from .model_flow import Model
        return Model(**args, with_classifier=False, classifier_loss="N/A", decoder_loss="mse")
    if model_name == "flow_vcr_mse":
        from .model_flow_vcr import Model
        return Model(**args, with_classifier=False, classifier_loss="N/A", decoder_loss="mse")
    if model_name == "flow_spherical_mse":
        from .model_flow_spherical import Model
        return Model(**args, with_classifier=False, classifier_loss="N/A", decoder_loss="mse")

    # hypothesis testing variants
    if model_name.startswith("ht_"):
        assert opt.rand_perms > 0
        assert opt.grouping is not None
        ht_args: dict = dict(
            labels=labels,
            cat_k=cat_k,
            emb_dims=emb_dims,
            permutations=generate_rand_perms(opt.rand_perms, cat_k, opt.grouping),
            optim_lr=optim_lr,
        )
        if model_name == "ht_linear_ce":
            from .model_ht_linear import Model
            return Model(**ht_args, classifier_loss="crossent")
        if model_name == "ht_linear_enc_ce":
            from .model_ht_linear import Model
            return Model(**ht_args, classifier_loss="crossent", with_encoder=True)
        if model_name == "ht_mlp_ce":
            from .model_ht_mlp import Model
            return Model(**ht_args, hidden_dims=32, classifier_loss="crossent")
        if model_name == "ht_mlp_enc_ce":
            from .model_ht_mlp import Model
            return Model(**ht_args, hidden_dims=32, classifier_loss="crossent", with_encoder=True)

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
