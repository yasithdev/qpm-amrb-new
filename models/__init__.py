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

    # autoencoder resnet variants
    if model_name.startswith("resnet_"):
        from .model_resnet import Model
        if model_name == "resnet_ce":
            return Model(**args, with_decoder=False, classifier_loss="crossent", decoder_loss="N/A")
        if model_name == "resnet_ce_mse":
            return Model(**args, with_decoder=True, classifier_loss="crossent", decoder_loss="mse")
        if model_name == "resnet_edl":
            return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
        if model_name == "resnet_edl_mse":
            return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")
    
    # autoencoder rescaps variants
    if model_name.startswith("rescaps_"):
        from .model_rescaps import Model
        if model_name == "rescaps_margin":
            return Model(**args, with_decoder=False, classifier_loss="margin", decoder_loss="N/A")
        if model_name == "rescaps_margin_mse":
            return Model(**args, with_decoder=True, classifier_loss="margin", decoder_loss="mse")
        if model_name == "rescaps_ce":
            return Model(**args, with_decoder=False, classifier_loss="crossent", decoder_loss="N/A")
        if model_name == "rescaps_ce_mse":
            return Model(**args, with_decoder=True, classifier_loss="crossent", decoder_loss="mse")
        if model_name == "rescaps_edl":
            return Model(**args, with_decoder=False, classifier_loss="edl", decoder_loss="N/A")
        if model_name == "rescaps_edl_mse":
            return Model(**args, with_decoder=True, classifier_loss="edl", decoder_loss="mse")

    # resnet18 variants
    if model_name.startswith("resnet18_"):
        from .model_resnet18 import Model
        if model_name == "resnet18_ce":
            return Model(**args, with_classifier=True, with_embedder=False, classifier_loss="crossent", embedding_loss="N/A", opt=opt)
        if model_name == "resnet18_edl":
            return Model(**args, with_classifier=True, with_embedder=False, classifier_loss="edl", embedding_loss="N/A", opt=opt)
        if model_name == "resnet18_simclr":
            return Model(**args, with_classifier=False, with_embedder=True, classifier_loss="N/A", embedding_loss="simclr", opt=opt)
        if model_name == "resnet18_simclr_ce":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="crossent", embedding_loss="simclr", opt=opt)
        if model_name == "resnet18_simclr_edl":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="edl", embedding_loss="simclr", opt=opt)
        if model_name == "resnet18_vicreg":
            return Model(**args, with_classifier=False, with_embedder=True, classifier_loss="N/A", embedding_loss="vicreg", opt=opt)
        if model_name == "resnet18_vicreg_ce":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="crossent", embedding_loss="vicreg", opt=opt)
        if model_name == "resnet18_vicreg_edl":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="edl", embedding_loss="vicreg", opt=opt)

    # resnet50 variants
    if model_name.startswith("resnet50_"):
        from .model_resnet50 import Model
        if model_name == "resnet50_ce":
            return Model(**args, with_classifier=True, with_embedder=False, classifier_loss="crossent", embedding_loss="N/A", opt=opt)
        if model_name == "resnet50_edl":
            return Model(**args, with_classifier=True, with_embedder=False, classifier_loss="edl", embedding_loss="N/A", opt=opt)
        if model_name == "resnet50_simclr":
            return Model(**args, with_classifier=False, with_embedder=True, classifier_loss="N/A", embedding_loss="simclr", opt=opt)
        if model_name == "resnet50_simclr_ce":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="crossent", embedding_loss="simclr", opt=opt)
        if model_name == "resnet50_simclr_edl":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="edl", embedding_loss="simclr", opt=opt)
        if model_name == "resnet50_vicreg":
            return Model(**args, with_classifier=False, with_embedder=True, classifier_loss="N/A", embedding_loss="vicreg", opt=opt)
        if model_name == "resnet50_vicreg_ce":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="crossent", embedding_loss="vicreg", opt=opt)
        if model_name == "resnet50_vicreg_edl":
            return Model(**args, with_classifier=True, with_embedder=True, classifier_loss="edl", embedding_loss="vicreg", opt=opt)

    # flow multi-scale variants
    if model_name.startswith("flow_ms_"):
        from .model_flow_ms import Model
        if model_name == "flow_ms_vcr_mse_nll":
            return Model(**args, with_classifier=False, classifier_loss="N/A", u_loss="vcr", v_loss="mse", z_loss="nll")
        if model_name == "flow_ms_vcr_mse_nll_ce":
            return Model(**args, with_classifier=True, classifier_loss="crossent", u_loss="vcr", v_loss="mse", z_loss="nll")
        if model_name == "flow_ms_vcr_mse_nll_edl":
            return Model(**args, with_classifier=True, classifier_loss="edl", u_loss="vcr", v_loss="mse", z_loss="nll")
    
    # flow single-scale variants
    if model_name.startswith("flow_ss_"):
        from .model_flow_ss import Model
        if model_name == "flow_ss_vcr_mse":
            return Model(**args, with_classifier=False, classifier_loss="N/A", u_loss="vcr", v_loss="mse")
        if model_name == "flow_ss_vcr_nll":
            return Model(**args, with_classifier=False, classifier_loss="N/A", u_loss="vcr", v_loss="nll")
    
    # conditional flow matching variants
    if model_name.startswith("cfm_"):
        from .model_cfm import Model
        if model_name == "cfm_otcfm_c":
            return Model(**args, strategy="otcfm", ignore_dims="channel")
        if model_name == "cfm_otcfm_s":
            return Model(**args, strategy="otcfm", ignore_dims="spatial")
        if model_name == "cfm_otcfm":
            return Model(**args, strategy="otcfm", ignore_dims=None)

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
        if model_name.startswith("ht_linear_"):
            from .model_ht_linear import Model
            if model_name == "ht_linear_ce":
                return Model(**ht_args, classifier_loss="crossent")
            if model_name == "ht_linear_enc_ce":
                return Model(**ht_args, classifier_loss="crossent", with_encoder=True)
        
        if model_name.startswith("ht_mlp_"):
            from .model_ht_mlp import Model
            if model_name == "ht_mlp_ce":
                return Model(**ht_args, hidden_dims=32, classifier_loss="crossent")
            if model_name == "ht_mlp_enc_ce":
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
