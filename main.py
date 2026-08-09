import os
import os.path as osp
import math
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from typing import Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import ICMF

import torch.nn.functional as F
import gc
import copy


def _optimizer_reset_due(reset_interval: int, epoch: int) -> bool:

    return reset_interval > 0 and epoch > 0 and epoch % reset_interval == 0


def _parse_intervention_seeds(value: str) -> Tuple[int, ...]:
    seeds = tuple(int(token.strip()) for token in str(value).split(",") if token.strip())
    if not seeds:
        raise ValueError("final_relation_noise_seeds must contain at least one integer")
    if any(seed < 0 for seed in seeds):
        raise ValueError("final_relation_noise_seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("final_relation_noise_seeds must be unique")
    return seeds

class Runner:
    def __init__(self, args, logger=None):
        self.args = args
        self.final_relation_noise_seeds = (
            _parse_intervention_seeds(args.final_relation_noise_seeds)
            if float(args.final_relation_noise_ratio) > 0.0
            else ()
        )

        self.early_stop_patience = 1000 if bool(args.il) else 100
        self.early_stop_metric = str(args.early_stop_metric)
        self.early_stop_best = -float("inf")
        self.early_stop_bad_count = 0
        self.early_stop_best_epoch = -1
        self.early_stop_metric_warned = False



        self.best_model_epoch = -1
        self.best_model_metric = -float("inf")
        self.best_model_metric_name = str(args.early_stop_metric)
        self.eval_history = []
        self.training_state_history = []
        self.last_dvdc_state = None
        self.last_training_fusion_state = None
        self.final_checkpoint_record = {
            "saved": False,
            "filename": None,
        }
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        set_seed(args.random_seed)
        self.data_init()
        self.model_choice()
        set_seed(args.random_seed)

        if not self.args.only_test:
            self.dataloader_init(self.train_set)
            self.model_list = [self.model]
            if self.args.il:
                assert self.args.il_start < self.args.epoch
                train_epoch_1_stage = self.args.il_start
            else:
                train_epoch_1_stage = self.args.epoch
            self.optim_init(self.args, train_epoch_1_stage)

    def _early_stop_update(self, metrics: dict) -> bool:

        if self.args.only_test or bool(self.args.disable_early_stop):
            return False
        if self.args.il and self.stage == 0:
            return False

        patience = self.early_stop_patience
        if not isinstance(metrics, dict):
            return False

        metric = str(self.args.early_stop_metric)
        value = metrics.get(metric, None)
        if value is None:
            fallback = "mrr_avg" if ("mrr_avg" in metrics) else ("mrr" if ("mrr" in metrics) else None)
            if fallback is None:
                if (not self.early_stop_metric_warned) and self.logger is not None:
                    keys = ", ".join(sorted(metrics.keys()))
                    self.logger.info(
                        f"[EarlyStop][Warn] metric '{metric}' not found in eval metrics; available: [{keys}]"
                    )
                    self.early_stop_metric_warned = True
                return False
            if (not self.early_stop_metric_warned) and self.logger is not None:
                self.logger.info(
                    f"[EarlyStop][Warn] metric '{metric}' not found, fallback to '{fallback}'."
                )
                self.early_stop_metric_warned = True
            metric = fallback
            value = metrics.get(metric, None)
            if value is None:
                return False

        value = float(value)

        improved = value > (self.early_stop_best + 1e-8)
        if improved:
            self.early_stop_best = value
            self.early_stop_bad_count = 0
            self.early_stop_best_epoch = int(self.epoch)
            if self.logger is not None:
                self.logger.info(f"[EarlyStop] improved {metric}={value:.4f} @ ep{self.epoch}; reset bad_count")
            return False

        self.early_stop_bad_count += 1
        if self.logger is not None:
            self.logger.info(
                f"[EarlyStop] no_improve {metric}={value:.4f}; best={self.early_stop_best:.4f} @ ep{self.early_stop_best_epoch}; "
                f"bad_count={self.early_stop_bad_count}/{patience}"
            )
        return self.early_stop_bad_count >= patience

    def model_choice(self):
        self.model = ICMF(self.KGs, self.args)

        self.model = self._load_model(self.model, model_name=self.args.model_name_save)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.total_trainable_params = int(total_params)
        self.logger.info(f"total params num: {total_params}")

    def optim_init(self, opt, total_epoch):
        step_per_epoch = len(self.train_dataloader)
        opt.total_steps = int(step_per_epoch * total_epoch)
        self.logger.info(f"total_steps: {opt.total_steps}")
        self.logger.info(f"weight_decay: {opt.weight_decay}")
        self.optimizer, self.scheduler = set_optim(opt, self.model_list)

    def _reset_optimizer_preserve_schedule(self, reason: str) -> None:

        previous_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        scheduler_state = (
            copy.deepcopy(self.scheduler.state_dict())
            if self.scheduler is not None
            else None
        )
        self.optimizer, self.scheduler = set_optim(self.args, self.model_list)
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        for group, learning_rate in zip(self.optimizer.param_groups, previous_lrs):
            group["lr"] = learning_rate
        self.lr = previous_lrs[-1]
        if self.logger is not None:
            self.logger.info(
                f"[OptimizerProbe] AdamW reinitialized ({reason}) at epoch "
                f"{self.epoch}; scheduler progress preserved, lr={self.lr:.5f}"
            )

    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set = load_data(
            self.logger, self.args
        )
        self.train_ill = self.train_set.data
        eval_pairs = self.eval_set.data
        self.eval_left = torch.LongTensor(eval_pairs[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(eval_pairs[:, 1].squeeze()).cuda()


    def dataloader_init(self, train_set):
        bs = self.args.batch_size
        collator = Collator_base()
        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        self.train_dataloader = self._dataloader(train_set, bs, collator)

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=bool(int(self.args.workers) > 0),
            shuffle=(self.args.only_test == 0),

            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.modal_names = None
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.stage = 0

        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                self.epoch = i
                self.model._runtime_epoch = self.epoch
                if _optimizer_reset_due(
                    int(self.args.optimizer_reset_interval),
                    self.epoch,
                ):
                    reset_reason = f"periodic_reset_{int(self.args.optimizer_reset_interval)}"
                    self._reset_optimizer_preserve_schedule(reason=reset_reason)
                if self.args.il and self.epoch == self.args.il_start and self.stage == 0:
                    self.stage = 1

                    self.logger.info(
                        f"[IL] Unified strategy start at epoch {self.epoch}: "
                        f"lr={float(self.args.lr):.5f}, optimizer state kept, "
                        "scheduler=constant, update_timing=pre_train"
                    )
                    name = self._save_name_define()
                    self.test()
                    if not self.args.only_test and self.args.save_model:
                        self._save_model(self.model, input_name=f"{name}_non_iter")



                    self.logger.info("[IL] Wait for temporal consensus before the first pseudo-label refresh")

                self._run_il_update()

                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(f'Train | Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                if (i + 1) % self.args.eval_epoch == 0:
                    metrics = self.eval()
                    self.eval_history.append({
                        "epoch": int(self.epoch),
                        "mrr_l2r": float(metrics["mrr_l2r"]),
                        "hits1_l2r": float(metrics["hits1_l2r"]),
                    })
                    if self._early_stop_update(metrics):
                        if self.logger is not None:
                            self.logger.info(
                                f"[EarlyStop] stop training: best {self.early_stop_metric}={self.early_stop_best:.4f} "
                                f"@ ep{self.early_stop_best_epoch}, patience={self.early_stop_patience}"
                            )
                        break
                _tqdm.update(1)

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)

        if not self.args.only_test and self.args.save_model:
            checkpoint_path = self._save_model(self.model, input_name=name)
            self.final_checkpoint_record = {
                "saved": True,
                "filename": osp.basename(checkpoint_path),
            }




        self.test()

        self.logger.info(f"min loss {self.loss_log.get_min_loss()}")

    def _run_il_update(self):
        if not self.args.il or self.stage != 1:
            return
        if (self.epoch + 1) % self.args.semi_learn_step == 0:
            self.il_for_ea()
        if (
            (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0
            and len(self.new_links) != 0
        ):
            self.il_for_data_ref()

    def il_for_ea(self):
        with torch.no_grad():
            final_emb, _ = self.model.generate_joint_emb()
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(
                self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links
            )
            del final_emb
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")

        for attr in ('mu_dict', 'mu_ref_dict', 'q_dict', 'fusion_weights'):
            if hasattr(self.model, attr) and getattr(self.model, attr) is not None:
                setattr(self.model, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def il_for_data_ref(self):
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)

        if self.stage == 1 and self.scheduler is not None:
            step_per_epoch = len(self.train_dataloader)
            cycle_epochs = max(1, self.args.epoch - self.epoch)
            total_steps = int(step_per_epoch * cycle_epochs)
            self.args.total_steps = total_steps
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _: 1.0
            )
            self.logger.info(
                f"[IL] Constant scheduler kept after data_refresh: "
                f"steps/ep={step_per_epoch}, total_steps={total_steps}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.5f}"
            )

    def _save_name_define(self):
        prefix = ""
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def train(self, _tqdm):
        self.model.train()
        self.loss_log.reset()
        single_batch_epoch = len(self.train_dataloader) == 1
        if single_batch_epoch:

            self.model.encode_once()

        for batch in self.train_dataloader:
            loss, output = self.model(batch, skip_encode=single_batch_epoch)
            self.scaler.scale(loss).backward()
            self.step += 1
            self.output_statistic(loss, output)

            self.scaler.unscale_(self.optimizer)
            for model in self.model_list:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skip_lr_sched = (scale > self.scaler.get_scale())
            if not skip_lr_sched:
                self.scheduler.step()

            self.lr = self.scheduler.get_last_lr()[-1]
            for model in self.model_list:
                model.zero_grad(set_to_none=True)

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'modal_names' in output and output['modal_names'] is not None:
            self.modal_names = output['modal_names']

    def update_loss_log(self):
        if self.logger is not None:
            denom = max(1, len(self.train_dataloader))
            def _mean(key: str, default: float = 0.0) -> float:
                return float(self.curr_loss_dic.get(key, default)) / float(denom)

            loss_ea = _mean("loss_ea")
            loss_modal = _mean("loss_modal")

            modal_names = self.modal_names
            q_parts = []
            if modal_names is not None:
                for name in modal_names:
                    q_parts.append(f"{name}={_mean(f'q_{name}'):.3f}")

            if int(self.epoch) == 0:
                self.logger.info(
                    f"[ICMF][Config] "
                    f"mode={'IL' if bool(self.args.il) else 'NI'}, "
                    f"lr={float(self.args.lr):.5f}, scheduler=constant, "
                    f"dropout={float(self.args.dropout):.2f}/{float(self.args.attn_dropout):.2f}, "
                    f"optimizer_reset_interval={int(self.args.optimizer_reset_interval)}, "
                    f"dvdc={'ON' if bool(self.args.use_dvdc) else 'OFF'}, "
                    f"dvdc_window=[{int(self.args.dvdc_start_epoch)},"
                    f"{int(self.args.dvdc_end_epoch)}), "
                    f"relation={self.args.relation_intervention}, "
                    f"relation_warmup={int(self.args.relation_permuted_warmup_epochs)}, "
                    f"gph={self.args.gph_interaction_mode}/{self.args.gph_scale_mode}, "
                    f"hnc={'ON' if bool(self.args.use_hnc) else 'OFF'}"
                    f"(k={int(getattr(self.args, 'hnc_topk', 0))}, m={float(getattr(self.args, 'hnc_margin', 0.0)):.3f}), "
                    f"num_heads={int(self.args.num_attention_heads)}, "
                    f"csls={1 if bool(self.args.csls) else 0}, csls_k={int(self.args.csls_k)}, "
                    f"es(patience/eval)={self.early_stop_patience}/{int(self.args.eval_epoch)}"
                )


            w_parts = []
            if modal_names is not None:
                for name in modal_names:
                    w_parts.append(f"{name}={_mean(f'w_{name}'):.3f}")
            w_str = f" | w[{', '.join(w_parts)}]" if w_parts else ""


            dvdc_disp = _mean('dvdc_disp')
            dvdc_lam = _mean('dvdc_lambda')
            dvdc_active = _mean('dvdc_active')
            self.last_dvdc_state = {
                "scope": "stop-epoch training state",
                "epoch": int(self.epoch),
                "active": bool(round(dvdc_active)),
                "lambda": dvdc_lam,
                "raw_dispersion": dvdc_disp,
                "ema_dispersion": _mean('dvdc_disp_ema'),
                "target_dispersion": _mean('dvdc_target_disp_now'),
            }
            per_modality_training_state = {}
            if modal_names is not None:
                for name in modal_names:
                    mean_key = f"w_{name}"
                    if mean_key not in self.curr_loss_dic:
                        continue
                    per_modality_training_state[name] = {
                        "mean": _mean(mean_key),
                        "median": _mean(f"w_{name}_median"),
                        "fraction_lt_0_05": _mean(f"w_{name}_lt_005"),
                    }
            self.last_training_fusion_state = {
                "scope": "stop-epoch training state before checkpoint restoration",
                "epoch": int(self.epoch),
                "per_entity_max_weight_mean": _mean('w_entity_max_mean'),
                "valid_weight_fraction_lt_0_01": _mean('w_valid_lt_001'),
                "per_entity_weight_entropy_mean": _mean('w_entity_entropy_mean'),
                "per_modality": per_modality_training_state,
            }
            if bool(int(getattr(self.args, "record_training_trace", 0))):
                relation_warmup = int(
                    getattr(self.args, "relation_permuted_warmup_epochs", 0)
                )
                self.training_state_history.append({
                    "epoch": int(self.epoch),
                    "relation_training_input": (
                        "permuted"
                        if relation_warmup > 0 and int(self.epoch) < relation_warmup
                        else "clean"
                    ),
                    "relation_q_mean": _mean("q_rel"),
                    "relation_weight_mean": _mean("w_rel"),
                    "relation_weight_median": _mean("w_rel_median"),
                    "relation_weight_fraction_lt_0_05": _mean("w_rel_lt_005"),
                    "per_entity_max_weight_mean": _mean("w_entity_max_mean"),
                    "valid_weight_fraction_lt_0_01": _mean("w_valid_lt_001"),
                    "per_entity_weight_entropy_mean": _mean("w_entity_entropy_mean"),
                    "dvdc_lambda": dvdc_lam,
                    "dvdc_raw_dispersion": dvdc_disp,
                    "dvdc_ema_dispersion": _mean("dvdc_disp_ema"),
                    "dvdc_target_dispersion": _mean("dvdc_target_disp_now"),
                })
            dvdc_str = (
                f" | dvdc[active={dvdc_active:.0f}, λ={dvdc_lam:.4f}, "
                f"disp={dvdc_disp:.4f}]"
            )
            concentration_str = (
                f" | concentration[max={_mean('w_entity_max_mean'):.4f}, "
                f"p(w<.01)={_mean('w_valid_lt_001'):.4f}, "
                f"H={_mean('w_entity_entropy_mean'):.4f}]"
            )

            self.logger.info(
                f"[ICMF] Ep {self.epoch} | loss_ea={loss_ea:.4f}, loss_modal={loss_modal:.4f}{dvdc_str} | "
                f"q[{', '.join(q_parts)}]{w_str}{concentration_str}"
            )

            if modal_names is not None and int(self.epoch) % 50 == 0:
                ks_parts = [f"{n}={_mean(f'kendall_s_{n}'):.3f}" for n in modal_names]
                kw_parts = [f"{n}={_mean(f'kendall_w_{n}'):.3f}" for n in modal_names]
                self.logger.info(
                    f"[Kendall] Ep {self.epoch} | s(log_var)[{', '.join(ks_parts)}] | w(1/sigma2)[{', '.join(kw_parts)}]"
                )

        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    def eval(self, last_epoch=False):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        metrics = self._test(test_left, test_right, last_epoch=last_epoch)
        return metrics

    def _l2r_metrics_from_embedding(self, embedding, left, right):
        embedding = F.normalize(embedding)
        distance = pairwise_distances(embedding[left], embedding[right])
        if self.args.csls:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)
        sorted_indices = torch.argsort(distance, dim=1)
        ground_truth = torch.arange(left.shape[0], device=distance.device)
        ranks = (
            sorted_indices == ground_truth.unsqueeze(1)
        ).nonzero(as_tuple=True)[1]
        result = {
            "hits1": float((ranks < 1).double().mean().item()),
            "hits10": float((ranks < 10).double().mean().item()),
            "mrr": float((1.0 / (ranks.double() + 1)).mean().item()),
            "mean_rank": float((ranks.double() + 1).mean().item()),
        }
        del distance, sorted_indices, ranks, embedding
        return result

    def _evaluate_final_relation_noise(
        self,
        test_left,
        test_right,
        clean_mrr_l2r: float,
        clean_relation_q: torch.Tensor,
    ) -> Dict[str, object]:
        ratio = float(self.args.final_relation_noise_ratio)
        if ratio <= 0.0:
            return {}

        runs = []
        entity_groups = (
            test_left.detach().clone(),
            test_right.detach().clone(),
        )
        try:
            for seed in self.final_relation_noise_seeds:
                self.model.set_relation_noise_intervention(
                    ratio=ratio,
                    seed=int(seed),
                    entity_groups=entity_groups,
                )
                with torch.no_grad():
                    corrupted_embedding, _ = self.model.generate_joint_emb()
                    selected = self.model._last_relation_noise_mask
                    manifest = copy.deepcopy(self.model._last_relation_noise_manifest)
                    changed_relation_q = self.model.q_dict["rel"].detach()
                    selected_clean_q = clean_relation_q[selected]
                    selected_changed_q = changed_relation_q[selected]
                    clean_q_mean = float(selected_clean_q.double().mean().item())
                    changed_q_mean = float(selected_changed_q.double().mean().item())
                    metric = self._l2r_metrics_from_embedding(
                        corrupted_embedding,
                        test_left,
                        test_right,
                    )
                metric.update({
                    "intervention_seed": int(seed),
                    "selection_manifest": manifest,
                    "clean_mrr": float(clean_mrr_l2r),
                    "mrr_relative_change": (
                        (metric["mrr"] - clean_mrr_l2r) / clean_mrr_l2r
                    ),
                    "clean_mean_q_selected": clean_q_mean,
                    "mean_q_selected": changed_q_mean,
                    "q_relative_change": (
                        (changed_q_mean - clean_q_mean) / clean_q_mean
                        if clean_q_mean > 0.0 else 0.0
                    ),
                })
                runs.append(metric)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            self.model.clear_relation_noise_intervention()

        def summarize(key: str) -> Dict[str, float]:
            values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
            return {
                "mean": float(values.mean()),
                "population_std": float(values.std(ddof=0)),
            }

        return {
            "scope": "selected clean RQ1 checkpoint; L2R; paired test-time intervention",
            "modality": "rel",
            "corruption": "Gaussian noise while preserving availability",
            "ratio": ratio,
            "intervention_seeds": list(self.final_relation_noise_seeds),
            "runs": runs,
            "summary": {
                "mrr": summarize("mrr"),
                "mrr_relative_change": summarize("mrr_relative_change"),
                "q_relative_change": summarize("q_relative_change"),
            },
        }


    def test(self, last_epoch=True):
        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(self.eval_left, self.eval_right, last_epoch=last_epoch)

    def _test(self, test_left, test_right, last_epoch=False):
        with torch.no_grad():
            final_emb, _ = self.model.generate_joint_emb()
            final_emb = F.normalize(final_emb)
            fusion_stats = self.model.fusion_statistics(
                torch.cat((test_left, test_right), dim=0)
            )
            if fusion_stats:
                fusion_stats = {
                    "scope": "restored RQ1-selected checkpoint",
                    **fusion_stats,
                }
            clean_relation_q = (
                self.model.q_dict["rel"].detach().clone()
                if last_epoch
                and float(self.args.final_relation_noise_ratio) > 0.0
                and "rel" in self.model.q_dict
                else None
            )

        top_k = [1, 10, 50]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float64)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float64)
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)


        n_test = test_left.shape[0]

        sorted_indices_l2r = torch.argsort(distance, dim=1)
        gt_indices_l2r = torch.arange(n_test, device=distance.device)
        ranks_l2r = (sorted_indices_l2r == gt_indices_l2r.unsqueeze(1)).nonzero(as_tuple=True)[1]


        mean_l2r = (ranks_l2r.double() + 1).mean().item()
        mrr_l2r = (1.0 / (ranks_l2r.double() + 1)).mean().item()
        hits_counts_l2r = [int((ranks_l2r < value).sum().item()) for value in top_k]
        for i, count in enumerate(hits_counts_l2r):
            acc_l2r[i] = count / n_test


        sorted_indices_r2l = torch.argsort(distance, dim=0)
        gt_indices_r2l = torch.arange(n_test, device=distance.device)
        ranks_r2l = (sorted_indices_r2l == gt_indices_r2l.unsqueeze(0)).nonzero(as_tuple=True)[0]
        mean_r2l = (ranks_r2l.double() + 1).mean().item()
        mrr_r2l = (1.0 / (ranks_r2l.double() + 1)).mean().item()
        hits_counts_r2l = [int((ranks_r2l < value).sum().item()) for value in top_k]
        for i, count in enumerate(hits_counts_r2l):
            acc_r2l[i] = count / n_test
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"

        self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
        self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
        acc_avg = np.round(0.5 * (acc_l2r + acc_r2l), 4)
        mean_avg = 0.5 * (mean_l2r + mean_r2l)
        mrr_avg = 0.5 * (mrr_l2r + mrr_r2l)
        self.logger.info(f"Ep {self.epoch} | avg: acc of top {top_k} = {acc_avg}, mr = {mean_avg:.3f}, mrr = {mrr_avg:.3f}{Loss_out}")


        hits1_l2r = hits_counts_l2r[0] / int(n_test)
        hits10_l2r = hits_counts_l2r[1] / int(n_test)
        hits1_r2l = hits_counts_r2l[0] / int(n_test)
        hits10_r2l = hits_counts_r2l[1] / int(n_test)
        mrr_l2r_f = float(mrr_l2r)
        mrr_r2l_f = float(mrr_r2l)

        metrics = {
            "hits1": hits1_l2r,
            "hits10": hits10_l2r,
            "mrr": mrr_l2r_f,

            "hits1_l2r": hits1_l2r,
            "hits10_l2r": hits10_l2r,
            "mrr_l2r": mrr_l2r_f,
            "hits1_r2l": hits1_r2l,
            "hits10_r2l": hits10_r2l,
            "mrr_r2l": mrr_r2l_f,

            "hits1_avg": 0.5 * (hits1_l2r + hits1_r2l),
            "hits10_avg": 0.5 * (hits10_l2r + hits10_r2l),
            "mrr_avg": 0.5 * (mrr_l2r_f + mrr_r2l_f),
            "hits1_min": min(hits1_l2r, hits1_r2l),
            "hits10_min": min(hits10_l2r, hits10_r2l),
            "mrr_min": min(mrr_l2r_f, mrr_r2l_f),
        }

        relation_noise_analysis = None
        if last_epoch and float(self.args.final_relation_noise_ratio) > 0.0:
            if clean_relation_q is None:
                raise RuntimeError("final relation-noise audit requires relation q scores")
            del distance, sorted_indices_l2r, sorted_indices_r2l, ranks_r2l, final_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            relation_noise_analysis = self._evaluate_final_relation_noise(
                test_left,
                test_right,
                mrr_l2r_f,
                clean_relation_q,
            )

        exact_metrics_output = str(
            getattr(self.args, "exact_metrics_output", "") or ""
        ).strip()
        if last_epoch and exact_metrics_output:
            exact_path = osp.abspath(exact_metrics_output)
            os.makedirs(osp.dirname(exact_path), exist_ok=True)
            selected_epoch = int(getattr(self, "best_model_epoch", -1))
            selection_metric_name = str(
                getattr(self, "best_model_metric_name", "")
            )
            selection_metric_value = float(
                getattr(self, "best_model_metric", float("nan"))
            )
            eval_history = list(getattr(self, "eval_history", []))
            if eval_history:
                peak_item = max(eval_history, key=lambda item: item["mrr_l2r"])
                last_item = eval_history[-1]
                training_dynamics = {
                    "evaluation_count": len(eval_history),
                    "peak_mrr_l2r": float(peak_item["mrr_l2r"]),
                    "peak_epoch": int(peak_item["epoch"]),
                    "last_mrr_l2r": float(last_item["mrr_l2r"]),
                    "last_epoch": int(last_item["epoch"]),
                    "peak_to_last_mrr_drop": float(
                        peak_item["mrr_l2r"] - last_item["mrr_l2r"]
                    ),
                }
            else:
                training_dynamics = None

            exact_record = {
                "schema_version": 2,
                "protocol_id": str(getattr(self.args, "protocol_id", "")),
                "variant_id": str(getattr(self.args, "variant_id", "")),
                "experiment_id": str(self.args.exp_id),
                "direction": "L2R",
                "retrieval": (
                    f"CSLS(k={int(self.args.csls_k)})"
                    if bool(self.args.csls)
                    else "raw distance"
                ),
                "data_choice": str(self.args.data_choice),
                "data_split": str(self.args.data_split),
                "data_rate": float(self.args.data_rate),
                "validation_rate": 0.0,
                "split_role": "rq1_shared_heldout",
                "split_manifest": self.KGs.get("alignment_split_manifest"),
                "relation_intervention_manifest": self.KGs.get(
                    "relation_intervention_manifest"
                ),
                "random_seed": int(self.args.random_seed),
                "trainable_parameters": int(self.total_trainable_params),
                "checkpoint": getattr(
                    self,
                    "final_checkpoint_record",
                    {"saved": False, "filename": None},
                ),
                "resolved_config": {
                    "model_name": str(self.args.model_name),
                    "structure_encoder": "gat",
                    "learning_rate": float(self.args.lr),
                    "scheduler": "constant",
                    "dropout": float(self.args.dropout),
                    "attention_dropout": float(self.args.attn_dropout),
                    "optimizer_reset_epoch": -1,
                    "optimizer_reset_interval": int(self.args.optimizer_reset_interval),
                    "il_update_timing": "pre_train",
                    "il_optimizer_policy": "keep",
                    "il_lr_scale": 1.0,
                    "uniform_fusion": bool(self.args.uniform_fusion),
                    "use_global_weights": bool(self.args.use_global_weights),
                    "use_dvdc": bool(self.args.use_dvdc),
                    "dvdc_target_disp": float(self.args.dvdc_target_disp),
                    "dvdc_lr": float(self.args.dvdc_lr),
                    "dvdc_ema_momentum": float(self.args.dvdc_ema_momentum),
                    "dvdc_start_epoch": int(self.args.dvdc_start_epoch),
                    "dvdc_end_epoch": int(self.args.dvdc_end_epoch),
                    "disp_type": str(self.args.disp_type),
                    "fixed_lambda": float(self.args.fixed_lambda),
                    "weight_floor": float(self.args.weight_floor),
                    "num_hidden_layers": int(self.args.num_hidden_layers),
                    "num_attention_heads": int(self.args.num_attention_heads),
                    "gph_interaction_mode": str(self.args.gph_interaction_mode),
                    "gph_scale_mode": str(self.args.gph_scale_mode),
                    "cml_mode": str(self.args.cml_mode),
                    "relation_intervention": str(self.args.relation_intervention),
                    "relation_permutation_seed": int(
                        self.args.relation_permutation_seed
                    ),
                    "relation_permuted_warmup_epochs": int(
                        self.args.relation_permuted_warmup_epochs
                    ),
                    "record_training_trace": bool(self.args.record_training_trace),
                    "attribute_input_dim": int(
                        self.KGs["att_features"].shape[1]
                    ),
                    "final_relation_noise_ratio": float(
                        self.args.final_relation_noise_ratio
                    ),
                    "final_relation_noise_seeds": list(
                        self.final_relation_noise_seeds
                    ),
                    "q_source": "projected",
                    "use_hnc": bool(self.args.use_hnc),
                    "hnc_margin": float(self.args.hnc_margin),
                    "hnc_topk": int(self.args.hnc_topk),
                    "csls": bool(self.args.csls),
                    "csls_k": int(self.args.csls_k),
                    "disable_early_stop": bool(self.args.disable_early_stop),
                    "early_stop_metric": str(self.args.early_stop_metric),
                },
                "n_queries": int(n_test),
                "selected_epoch": selected_epoch,
                "selection_metric": selection_metric_name,
                "selection_metric_value": selection_metric_value,
                "stop_epoch": (
                    int(self.epoch) if isinstance(self.epoch, int) else str(self.epoch)
                ),
                "training_dynamics": training_dynamics,
                "evaluation_history": (
                    self.eval_history
                    if bool(int(getattr(self.args, "record_training_trace", 0)))
                    else None
                ),
                "training_state_history": (
                    self.training_state_history
                    if bool(int(getattr(self.args, "record_training_trace", 0)))
                    else None
                ),
                "dvdc_final_state": self.last_dvdc_state,
                "stop_epoch_fusion_state": self.last_training_fusion_state,
                "fusion_statistics": fusion_stats,
                "relation_noise_analysis": relation_noise_analysis,
                "l2r": {
                    "hits1": hits1_l2r,
                    "hits10": hits10_l2r,
                    "hits1_count": hits_counts_l2r[0],
                    "hits10_count": hits_counts_l2r[1],
                    "mrr": mrr_l2r_f,
                    "mean_rank": float(mean_l2r),
                },
                "audit_only_r2l": {
                    "hits1": hits1_r2l,
                    "hits10": hits10_r2l,
                    "hits1_count": hits_counts_r2l[0],
                    "hits10_count": hits_counts_r2l[1],
                    "mrr": mrr_r2l_f,
                    "mean_rank": float(mean_r2l),
                },
            }
            if not math.isfinite(exact_record["selection_metric_value"]):
                exact_record["selection_metric_value"] = None
            temporary_path = exact_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as exact_file:
                json.dump(
                    exact_record,
                    exact_file,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                exact_file.write("\n")
            os.replace(temporary_path, exact_path)

        metric_key = str(self.args.early_stop_metric)
        metric_val = metrics.get(metric_key, None)
        if metric_val is None:
            metric_key = "mrr"
            metric_val = metrics.get(metric_key, mrr_l2r_f)

        if (
            not self.args.only_test
            and float(metric_val) > self.best_model_metric
            and not last_epoch
        ):
            self.logger.info(
                f"Best model update in Ep {self.epoch}: {metric_key} from "
                f"[{self.best_model_metric:.6f}] --> [{float(metric_val):.6f}] ... "
            )
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_model_epoch = int(self.epoch)
            self.best_model_metric = float(metric_val)
            self.best_model_metric_name = metric_key

        return metrics

    def _load_model(self, model, model_name=None):
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        if len(model_name) == 0 or not os.path.exists(save_path):
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:
                self.logger.info("Random init...")
            model.cuda()
            return model
        model.load_state_dict(torch.load(save_path, map_location=self.args.device))

        model.cuda()
        self.logger.info(f"loading model [{model_name}.pkl] done!")

        return model

    def _save_model(self, model, input_name=""):

        model_name = self.args.model_name

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)

        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')

        if model is None:
            return
        if self.args.save_model:
            temporary_path = save_path + ".tmp"
            torch.save(model.state_dict(), temporary_path)
            os.replace(temporary_path, save_path)

            self.logger.info(f"saving [{save_path}] done!")

        return save_path


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)

    torch.multiprocessing.set_sharing_strategy('file_system')

    logger = initialize_exp(cfgs)

    cfgs.device = torch.device(cfgs.device)


    torch.cuda.set_device(0)
    runner = Runner(cfgs, logger)
    if cfgs.only_test:
        runner.test(last_epoch=True)
    else:
        runner.run()

    logger.info("done!")
