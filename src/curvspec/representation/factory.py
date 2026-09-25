from curvspec.config import AttributeDict
from curvspec.representation.network import CurvSpec_Net


def build_model(config):
    model_config = AttributeDict(
        visual_input_size=config["visual_feat_dim"],
        query_input_size=config["q_feat_size"],
        hidden_size=config["hidden_size"],
        max_ctx_l=config["max_ctx_l"],
        max_desc_l=config["max_desc_l"],
        map_size=config["map_size"],
        input_drop=config["input_drop"],
        drop=config["drop"],
        n_heads=config["n_heads"],
        initializer_range=config["initializer_range"],
        margin=config["margin"],
        use_hard_negative=False,
        hard_pool_size=config["hard_pool_size"],
        sft_factor=config["sft_factor"],
        curv_init=config["curv_init"],
        learn_curv=config["learn_curv_bool"],
        lorentz_dim=config["lorentz_dim"],
        attention_num=config["attention_num"],
        use_learnable_curvature=config["use_learnable_curvature"],
        curvature_init_val=config["curvature_init_val"],
        min_curvature=config["min_curvature"],
        num_constraints=config["num_constraints"],
        use_adaptive_fusion=config.get("use_adaptive_fusion", False),
        kmeans_use_gpu=config.get("kmeans_use_gpu", False),
        enable_loss_exclusive=config.get("enable_loss_exclusive", True),
    )
    return CurvSpec_Net(model_config)
