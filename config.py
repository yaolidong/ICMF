import os.path as osp
import json
import argparse


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        self.data_root = osp.abspath(osp.join(self.this_dir, 'data'))

    def get_args(self):
        parser = argparse.ArgumentParser()


        parser.add_argument('--batch_size', default=3500, type=int)
        parser.add_argument('--epoch', default=None, type=int)
        parser.add_argument('--random_seed', default=42, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])
        parser.add_argument("--exp_name", default="EA_exp", type=str)
        parser.add_argument("--exp_id", default="001", type=str)
        parser.add_argument(
            "--protocol_id",
            default="",
            type=str,
            help="Immutable rerun protocol identifier recorded in exact metrics.",
        )
        parser.add_argument(
            "--variant_id",
            default="",
            type=str,
            help="Ablation/design variant identifier recorded in exact metrics.",
        )
        parser.add_argument("--model_name_save", default="", type=str,
                            help="Checkpoint name under data/MMKG/ICMF/save for evaluation")
        parser.add_argument(
            "--exact_metrics_output",
            default="",
            type=str,
            help=(
                "Optional JSON path for an atomic, full-precision final-test record. "
                "Strict rerun scripts use this instead of rounded human-readable logs."
            ),
        )
        parser.add_argument('--workers', type=int, default=4)


        parser.add_argument("--data_choice", default="DBP15K", type=str,
                            choices=["DBP15K", "FBYG15K", "FBDB15K"])
        parser.add_argument("--data_rate", type=float, default=0.3)
        parser.add_argument("--data_split", default="fr_en", type=str,
                            choices=["zh_en", "ja_en", "fr_en", "norm"])


        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--weight_decay', type=float, default=0.001)
        parser.add_argument(
            "--optimizer_reset_interval",
            type=int,
            default=None,
            help=(
                "AdamW rebuild interval with scheduler progress preserved; "
                "defaults to 50 for both NI and IL"
            ),
        )
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--attn_dropout", type=float, default=None)
        parser.add_argument('--eval_epoch', default=2, type=int)


        parser.add_argument("--il", action="store_true", default=False)
        parser.add_argument("--il_start", type=int, default=50)
        parser.add_argument("--semi_learn_step", type=int, default=5,
                            help="IL pseudo-label generation frequency (epochs)")


        parser.add_argument("--csls", action=argparse.BooleanOptionalAction, default=True,
                            help="Use CSLS for evaluation and IL pseudo-label retrieval")
        parser.add_argument("--csls_k", type=int, default=3,
                            help="CSLS neighborhood size for evaluation and IL retrieval")
        parser.add_argument("--disable_early_stop", type=int, default=1, choices=[0, 1])
        parser.add_argument("--early_stop_metric", type=str, default=None,
                            choices=["mrr_avg", "hits1_l2r", "mrr_l2r"])


        parser.add_argument("--use_dvdc", type=int, default=1, choices=[0, 1],
                            help="A2 消融: DVDC 开关 (默认开启)")
        parser.add_argument("--dvdc_target_disp", type=float, default=0.1,
                            help="DVDC log-score second-moment budget B_disp")
        parser.add_argument("--dvdc_lr", type=float, default=0.05,
                            help="DVDC exponentiated-controller step size (shared by NI and IL)")
        parser.add_argument("--dvdc_ema_momentum", type=float, default=0.95,
                            help="DVDC log-score second-moment EMA momentum (shared by NI and IL)")
        parser.add_argument("--dvdc_start_epoch", type=int, default=0,
                            help="First epoch (inclusive) at which adaptive DVDC is active")
        parser.add_argument("--dvdc_end_epoch", type=int, default=-1,
                            help="First epoch (exclusive) at which adaptive DVDC is inactive; -1 means no end")
        parser.add_argument("--disp_type", type=str, default="lognormal",
                            choices=["lognormal", "none"],
                            help="DVDC control-statistic type")
        parser.add_argument("--fixed_lambda", type=float, default=-1.0,
                            help="Fixed DVDC lambda (>0 disables auto-tuning)")
        parser.add_argument("--weight_floor", type=float, default=0.0,
                            help=("Naive anti-collapse baseline: per-entity lower bound on "
                                  "available-modality fusion weights, renormalized after "
                                  "clamping. >0 enables it"))
        parser.add_argument("--uniform_fusion", type=int, default=0, choices=[0, 1],
                            help="A1 消融: 等权融合")
        parser.add_argument("--use_global_weights", type=int, default=0, choices=[0, 1],
                            help="A9 消融: 全局标量权重")
        parser.add_argument("--gph_interaction_mode", type=str, default="none",
                            choices=["none", "full"],
                            help="gph 参与 BertLayer 的模式 (默认 none)")
        parser.add_argument("--gph_scale_mode", type=str, default="raw",
                            choices=["raw", "match_non_gph_mean"],
                            help="Scale treatment applied to the graph token before full cross-modal interaction")
        parser.add_argument("--cml_mode", type=str, default="full",
                            choices=["full", "layernorm_only"],
                            help=("CrossModalLayer computation: the full attention-plus-dense block, "
                                  "or only its LayerNorm, which keeps cross-side scale normalization "
                                  "while removing modality mixing"))
        parser.add_argument("--relation_intervention", type=str, default="clean",
                            choices=["clean", "permuted", "removed"],
                            help=("Training/test relation-input intervention: clean input, a fixed "
                                  "within-KG permutation, or complete branch removal"))
        parser.add_argument("--relation_permutation_seed", type=int, default=20260722,
                            help="Fixed seed for the within-KG relation-feature permutation")
        parser.add_argument(
            "--relation_permuted_warmup_epochs",
            type=int,
            default=0,
            help=("Use a fixed within-KG relation-feature permutation only while "
                  "training for the first N epochs; evaluation always uses clean "
                  "relation features"),
        )
        parser.add_argument(
            "--record_training_trace",
            type=int,
            default=0,
            choices=[0, 1],
            help="Store per-epoch training and evaluation diagnostics in exact metrics",
        )
        parser.add_argument("--use_hnc", type=int, default=1, choices=[0, 1],
                            help="启用 Hard Negative Calibration (默认开启)")
        parser.add_argument("--hnc_margin", type=float, default=0.2,
                            help="HNC 难负例排斥 Margin")
        parser.add_argument("--hnc_topk", type=int, default=3,
                            help="每个样本增强的 top-k 最难负例数量")
        parser.add_argument("--overrides", type=str, default="",
                            help="JSON override file for ablation")


        parser.add_argument(
            "--final_relation_noise_ratio",
            type=float,
            default=0.0,
            help=("Final-checkpoint audit only: corrupt this fraction of available relation "
                  "entities independently on each KG side; checkpoint selection remains clean"),
        )
        parser.add_argument(
            "--final_relation_noise_seeds",
            type=str,
            default="101,202,303,404,505",
            help="Comma-separated intervention seeds for the final relation-noise audit",
        )

        self.cfg = parser.parse_args()

    def update_train_configs(self):
        assert not (self.cfg.save_model and self.cfg.only_test)



        is_il = bool(self.cfg.il)
        if self.cfg.epoch is None:
            self.cfg.epoch = 1000 if is_il else 500
        if self.cfg.optimizer_reset_interval is None:
            self.cfg.optimizer_reset_interval = 50
        if self.cfg.optimizer_reset_interval < 0:
            raise ValueError("optimizer_reset_interval must be non-negative")
        if self.cfg.lr is None:
            self.cfg.lr = 5e-4
        if self.cfg.dropout is None:
            self.cfg.dropout = 0.1
        if self.cfg.attn_dropout is None:
            self.cfg.attn_dropout = self.cfg.dropout
        if self.cfg.early_stop_metric is None:
            self.cfg.early_stop_metric = "mrr_l2r"


        self.cfg.model_name = "ICMF"
        self.cfg.device = "cuda"
        self.cfg.clip = 1.0
        self.cfg.tau = 0.1
        self.cfg.ab_weight = 0.5
        self.cfg.adam_epsilon = 1e-8
        self.cfg.hidden_units = "300,300,300"
        self.cfg.heads = "2,2"
        self.cfg.attr_dim = 300
        self.cfg.img_dim = 300
        self.cfg.hidden_size = 300
        self.cfg.intermediate_size = 400
        self.cfg.num_attention_heads = 1
        self.cfg.num_hidden_layers = 1
        self.cfg.use_key_mask = 1

        self.cfg.gph_interaction_mode = str(self.cfg.gph_interaction_mode)

        self.cfg.semi_learn_step = int(self.cfg.semi_learn_step)
        log_category = (
            "icmf_ablations"
            if self.cfg.protocol_id or self.cfg.variant_id
            else "icmf_final"
        )
        self.cfg.dump_path = osp.abspath(
            osp.join(self.this_dir, "logs", log_category, "internal_runs")
        )
        self.cfg.data_path = "MMKG"




        self.cfg.use_intermediate = 0
        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.data_split = "norm"
            data_split_name = f"{self.cfg.data_rate}_"
        else:
            data_split_name = f"{self.cfg.data_split}_"

        self.cfg.exp_id = f"ICMF_{self.cfg.data_choice}_{data_split_name}{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)


        path = str(self.cfg.overrides or "").strip()
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    overrides = json.load(f)
            except Exception as e:
                raise ValueError(f"failed to load overrides={path!r}: {e}") from e
            if not isinstance(overrides, dict):
                raise ValueError(f"overrides must be a JSON object, got {type(overrides)}")
            _allowed = {"tau", "csls", "csls_k", "epoch",
                        "eval_epoch", "weight_decay",
                        "num_attention_heads", "num_hidden_layers",
                        "use_dvdc", "dvdc_target_disp",
                        "dvdc_lr", "dvdc_start_epoch", "dvdc_end_epoch", "disp_type",
                        "uniform_fusion", "use_global_weights",
                        "gph_interaction_mode", "gph_scale_mode", "cml_mode",
                        "relation_intervention", "relation_permutation_seed",
                        "relation_permuted_warmup_epochs", "record_training_trace",
                        "use_hnc", "hnc_margin", "hnc_topk"}
            for k, v in overrides.items():
                if not isinstance(k, str) or not k.strip():
                    continue
                k = k.strip()
                if k not in _allowed:
                    continue
                if isinstance(v, bool):
                    v = int(v)
                setattr(self.cfg, k, v)

        self.cfg.dvdc_start_epoch = int(self.cfg.dvdc_start_epoch)
        self.cfg.dvdc_end_epoch = int(self.cfg.dvdc_end_epoch)
        if self.cfg.disp_type not in {"lognormal", "none"}:
            raise ValueError(
                "disp_type must be one of: lognormal, none"
            )
        if self.cfg.gph_interaction_mode not in {"none", "full"}:
            raise ValueError("gph_interaction_mode must be one of: none, full")
        if self.cfg.gph_scale_mode not in {"raw", "match_non_gph_mean"}:
            raise ValueError(
                "gph_scale_mode must be one of: raw, match_non_gph_mean"
            )
        self.cfg.cml_mode = str(self.cfg.cml_mode)
        if self.cfg.cml_mode not in {"full", "layernorm_only"}:
            raise ValueError("cml_mode must be one of: full, layernorm_only")
        if self.cfg.cml_mode != "full" and self.cfg.num_hidden_layers < 1:
            raise ValueError("cml_mode=layernorm_only requires num_hidden_layers >= 1")
        if self.cfg.dvdc_start_epoch < 0:
            raise ValueError("dvdc_start_epoch must be non-negative")
        if self.cfg.dvdc_start_epoch >= self.cfg.epoch:
            raise ValueError("dvdc_start_epoch must be smaller than epoch")
        if self.cfg.dvdc_end_epoch != -1:
            if self.cfg.dvdc_end_epoch <= self.cfg.dvdc_start_epoch:
                raise ValueError("dvdc_end_epoch must be greater than dvdc_start_epoch")
            if self.cfg.dvdc_end_epoch > self.cfg.epoch:
                raise ValueError("dvdc_end_epoch must not exceed epoch")
        if self.cfg.gph_scale_mode != "raw" and self.cfg.gph_interaction_mode != "full":
            raise ValueError("gph_scale_mode=match_non_gph_mean requires gph_interaction_mode=full")
        if not 0.0 <= float(self.cfg.final_relation_noise_ratio) <= 1.0:
            raise ValueError("final_relation_noise_ratio must lie in [0, 1]")
        self.cfg.relation_permuted_warmup_epochs = int(
            self.cfg.relation_permuted_warmup_epochs
        )
        if self.cfg.relation_permuted_warmup_epochs < 0:
            raise ValueError("relation_permuted_warmup_epochs must be non-negative")
        if self.cfg.relation_permuted_warmup_epochs >= int(self.cfg.epoch):
            raise ValueError("relation_permuted_warmup_epochs must be smaller than epoch")
        if (
            self.cfg.relation_permuted_warmup_epochs > 0
            and self.cfg.relation_intervention != "clean"
        ):
            raise ValueError(
                "relation_permuted_warmup_epochs requires relation_intervention=clean"
            )
        return self.cfg
