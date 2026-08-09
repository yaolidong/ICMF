

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import pairwise_distances
from .ICMF_tools import (
    MultiModalEncoder, CrossModalLayer,
)
from .ICMF_loss import icl_loss, CustomMultiLossLayer


def _select_available_entities_by_side(
    entity_count: int,
    side_groups: Tuple[torch.Tensor, torch.Tensor],
    available: torch.Tensor,
    ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:

    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must lie in (0, 1]")
    available_cpu = available.detach().to(device="cpu", dtype=torch.bool)
    selected_mask = torch.zeros(int(entity_count), dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    manifest: Dict[str, object] = {
        "type": "fixed_fraction_available_entities_by_kg_side",
        "ratio": float(ratio),
        "seed": int(seed),
        "sides": {},
    }
    for side_name, side_ids in zip(("left", "right"), side_groups):
        ids_cpu = side_ids.detach().to(device="cpu", dtype=torch.long)
        candidates = ids_cpu[available_cpu[ids_cpu]]
        selected_count = int(round(float(ratio) * int(candidates.numel())))
        if candidates.numel():
            selected_count = min(int(candidates.numel()), max(1, selected_count))
            order = torch.randperm(int(candidates.numel()), generator=generator)
            selected = candidates[order[:selected_count]]
        else:
            selected = torch.empty(0, dtype=torch.long)
        selected_mask[selected] = True
        manifest["sides"][side_name] = {
            "available_count": int(candidates.numel()),
            "selected_count": int(selected.numel()),
        }
    return selected_mask.to(available.device), manifest


@torch.no_grad()
def _chunked_csls_nearest_neighbors(
    left_emb: torch.Tensor,
    right_emb: torch.Tensor,
    k: int,
    batch_size: int = 512,
) -> Tuple[List[int], List[int]]:

    if left_emb.ndim != 2 or right_emb.ndim != 2:
        raise ValueError("CSLS expects two rank-2 embedding tensors")
    if left_emb.size(0) == 0 or right_emb.size(0) == 0:
        return [], []
    if left_emb.size(1) != right_emb.size(1):
        raise ValueError("CSLS embedding dimensions must match")

    n_left = int(left_emb.size(0))
    n_right = int(right_emb.size(0))
    batch_size = max(1, int(batch_size))
    k_left = max(1, min(int(k), n_left))
    k_right = max(1, min(int(k), n_right))


    left_scale_parts: List[torch.Tensor] = []
    right_topk = torch.full(
        (n_right, k_left),
        -torch.inf,
        dtype=right_emb.dtype,
        device=right_emb.device,
    )
    for start in range(0, n_left, batch_size):
        left_chunk = left_emb[start:start + batch_size]
        similarity = 1.0 - pairwise_distances(left_chunk, right_emb)
        left_scale_parts.append(
            torch.topk(similarity, k=k_right, dim=1).values.mean(dim=1)
        )
        right_candidates = torch.cat((right_topk, similarity.transpose(0, 1)), dim=1)
        right_topk = torch.topk(right_candidates, k=k_left, dim=1).values

    left_scale = torch.cat(left_scale_parts, dim=0)
    right_scale = right_topk.mean(dim=1)


    preds_l = torch.empty(n_left, dtype=torch.long, device=left_emb.device)
    preds_r = torch.zeros(n_right, dtype=torch.long, device=right_emb.device)
    best_r = torch.full(
        (n_right,),
        -torch.inf,
        dtype=right_emb.dtype,
        device=right_emb.device,
    )
    for start in range(0, n_left, batch_size):
        left_chunk = left_emb[start:start + batch_size]
        similarity = 1.0 - pairwise_distances(left_chunk, right_emb)
        csls_score = (
            2.0 * similarity
            - left_scale[start:start + left_chunk.size(0)].unsqueeze(1)
            - right_scale.unsqueeze(0)
        )
        preds_l[start:start + left_chunk.size(0)] = torch.argmax(csls_score, dim=1)

        chunk_best, chunk_arg = torch.max(csls_score, dim=0)
        improve = chunk_best > best_r
        best_r[improve] = chunk_best[improve]
        preds_r[improve] = start + chunk_arg[improve]

    return preds_l.cpu().tolist(), preds_r.cpu().tolist()


def _build_reliability_nets(
    hidden_dim: int,
    modal_names: List[str],
    enabled: bool,
) -> nn.ModuleDict:

    nets = nn.ModuleDict()
    if not enabled:
        return nets

    hidden = max(1, hidden_dim // 4)
    for name in modal_names:
        nets[name] = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    return nets


def _mask_aware_uniform_weights(
    active: List[str],
    masks: Dict[str, torch.Tensor],
    n_ent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    availability = torch.stack(
        [
            masks.get(name, torch.ones(n_ent, dtype=torch.bool, device=device)).to(
                device=device,
                dtype=dtype,
            )
            for name in active
        ],
        dim=1,
    )
    counts = availability.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("uniform fusion requires at least one available modality per entity")
    return availability / counts


def _match_graph_token_to_non_graph_scale(
    graph_token: torch.Tensor,
    non_graph_tokens: List[torch.Tensor],
    non_graph_masks: Optional[List[torch.Tensor]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:

    if not non_graph_tokens:
        raise ValueError("graph scale matching requires at least one non-graph token")
    if non_graph_masks is not None and len(non_graph_masks) != len(non_graph_tokens):
        raise ValueError("non-graph masks must align with non-graph tokens")

    norms = torch.stack([token.norm(p=2, dim=-1) for token in non_graph_tokens], dim=1)
    if non_graph_masks is None:
        availability = torch.ones_like(norms)
    else:
        availability = torch.stack(
            [
                mask.to(device=norms.device, dtype=norms.dtype)
                if mask is not None
                else torch.ones(norms.size(0), device=norms.device, dtype=norms.dtype)
                for mask in non_graph_masks
            ],
            dim=1,
        )
    available_count = availability.sum(dim=1)
    graph_norm = graph_token.norm(p=2, dim=-1).clamp_min(eps)
    target_norm = (norms * availability).sum(dim=1) / available_count.clamp_min(1.0)
    target_norm = torch.where(available_count > 0, target_norm, graph_norm)
    return graph_token * (target_norm / graph_norm).unsqueeze(-1)


class ICMFFusion(nn.Module):


    def __init__(self, args):
        super().__init__()
        self.gph_mode = str(args.gph_interaction_mode)
        self.gph_scale_mode = str(getattr(args, "gph_scale_mode", "raw"))
        self.layers = nn.ModuleList([
            CrossModalLayer(args) for _ in range(args.num_hidden_layers)
        ])

    def _build_key_mask(self, modal_masks, n_ent, device, prepend_gph=False):

        if modal_masks is None:
            return None
        rows = [
            m.to(device=device, dtype=torch.bool) if m is not None
            else torch.ones(n_ent, dtype=torch.bool, device=device)
            for m in modal_masks
        ]
        if prepend_gph:
            rows = [torch.ones(n_ent, dtype=torch.bool, device=device)] + rows
        return torch.stack(rows, dim=1)

    def _run_layers(self, hidden_states, key_mask):

        for layer in self.layers:
            hidden_states = layer(hidden_states, key_mask=key_mask)
        return hidden_states

    def forward(
        self,
        gph_emb: torch.Tensor,
        embs: List[torch.Tensor],
        modal_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:

        n_ent = gph_emb.size(0)
        device = gph_emb.device

        if self.gph_mode == 'none':
            hidden_states = torch.stack(embs, dim=1)
            key_mask = self._build_key_mask(modal_masks, n_ent, device)
            hidden_states = self._run_layers(hidden_states, key_mask)
            return torch.cat([gph_emb.unsqueeze(1), hidden_states], dim=1)


        if self.gph_scale_mode == "match_non_gph_mean":
            gph_emb = _match_graph_token_to_non_graph_scale(
                gph_emb,
                embs,
                modal_masks,
            )
        hidden_states = torch.stack([gph_emb] + embs, dim=1)
        key_mask = self._build_key_mask(modal_masks, n_ent, device, prepend_gph=True)

        hidden_states = self._run_layers(hidden_states, key_mask)
        return hidden_states


class ICMF(nn.Module):


    def __init__(self, kgs, args):
        super().__init__()
        self.args = args
        self.device = self._parse_device(args.device)
        self.ent_num = int(kgs["ent_num"])
        self.hidden_dim = int(args.hidden_size)
        self.eps = 1e-12


        _hidden_units = [int(x) for x in args.hidden_units.strip().split(",")]
        _gph_dim = _hidden_units[-1]
        _attr_dim = int(args.attr_dim)
        _img_dim = int(args.img_dim)
        self.modal_dims: Dict[str, int] = {
            "gph": _gph_dim, "rel": _attr_dim, "attr": _attr_dim,
            "img": _img_dim,
        }


        self.input_idx = self._to_tensor(kgs["input_idx"], torch.long)
        self.adj = kgs.get("adj", None)
        if self.adj is not None and torch.is_tensor(self.adj):
            self.adj = self.adj.to(self.device)

        _img_raw = self._to_tensor(kgs.get("images_list", None), torch.float32)
        if _img_raw is not None:
            self.img_features = F.normalize(_img_raw)
        else:
            self.img_features = None
        self.rel_features = self._to_tensor(kgs.get("rel_features", None), torch.float32)
        self.rel_features_permuted = self._to_tensor(
            kgs.get("rel_features_permuted", None), torch.float32
        )
        self.att_features = self._to_tensor(kgs.get("att_features", None), torch.float32)


        self.img_mask = self._ensure_mask(kgs.get("img_mask", None), self.img_features)
        self.rel_mask = self._ensure_mask(kgs.get("rel_mask", None), self.rel_features)
        self.attr_mask = self._ensure_mask(kgs.get("attr_mask", None), self.att_features)

        self.modal_names = self._resolve_modal_names()
        self._masks: Dict[str, torch.Tensor] = {
            "gph": torch.ones(self.ent_num, dtype=torch.bool, device=self.device),
            "rel": self.rel_mask, "attr": self.attr_mask,
            "img": self.img_mask,
        }


        img_dim = self._infer_dim(kgs.get("images_list", None), args.img_dim)

        self.multi_encoder = MultiModalEncoder(
            args=args, ent_num=self.ent_num, img_feature_dim=img_dim,
            char_feature_dim=100,
            attr_input_dim=int(self.att_features.shape[1]) if self.att_features is not None else 1000,
        )




        self.modal_proj = nn.ModuleDict()
        for name in self.modal_names:
            m_dim = self.modal_dims.get(name, self.hidden_dim)
            if m_dim != self.hidden_dim:
                self.modal_proj[name] = nn.Linear(m_dim, self.hidden_dim)


        self.cross_modal_layer = ICMFFusion(args)


        self.uniform_fusion = bool(args.uniform_fusion)
        self.use_global_weights = bool(args.use_global_weights)
        self.use_key_mask = bool(args.use_key_mask)
        if self.uniform_fusion and self.use_global_weights:
            raise ValueError("uniform_fusion and use_global_weights are mutually exclusive")



        self.reliability_nets = _build_reliability_nets(
            self.hidden_dim,
            self.modal_names,
            enabled=not self.uniform_fusion and not self.use_global_weights,
        )





        if self.use_global_weights:
            self.global_weight_logits = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in self.modal_names
            })








        self.use_dvdc = bool(args.use_dvdc) and not self.uniform_fusion
        self.dvdc_target_disp = float(args.dvdc_target_disp)
        self.dvdc_lr = float(args.dvdc_lr)
        self.dvdc_ema_momentum = float(getattr(args, 'dvdc_ema_momentum', 0.95))
        self.dvdc_start_epoch = int(getattr(args, 'dvdc_start_epoch', 0))
        self.dvdc_end_epoch = int(getattr(args, 'dvdc_end_epoch', -1))
        self.disp_type = str(getattr(args, 'disp_type', 'lognormal'))

        self.weight_floor = float(getattr(args, 'weight_floor', 0.0))

        self._dvdc_log_lambda = 0.0
        self._dvdc_disp_ema = None
        self._dvdc_ema_momentum = self.dvdc_ema_momentum
        self._runtime_epoch = 0


        _tau = float(args.tau)
        _ab = float(args.ab_weight)
        _use_hnc = bool(getattr(args, 'use_hnc', 0))
        _hnc_margin = float(getattr(args, 'hnc_margin', 0.2))
        _hnc_topk = int(getattr(args, 'hnc_topk', 3))
        self.ea_criterion = icl_loss(
            tau=_tau, ab_weight=_ab, n_view=2,
            use_hnc=_use_hnc, hnc_margin=_hnc_margin, hnc_topk=_hnc_topk,
        )

        self.modal_cl_criterion = icl_loss(
            tau=_tau, ab_weight=_ab, n_view=2,
            use_hnc=False, hnc_margin=_hnc_margin, hnc_topk=_hnc_topk,
        )


        _n_modals = len(self.modal_names)
        self.multi_loss_layer = CustomMultiLossLayer(loss_num=_n_modals)



        self.q_dict: Dict[str, torch.Tensor] = {}
        self.mu_dict: Dict[str, torch.Tensor] = {}
        self.mu_ref_dict: Dict[str, torch.Tensor] = {}
        self.fusion_weights: Optional[torch.Tensor] = None
        self._relation_noise_ratio = 0.0
        self._relation_noise_seed = 0
        self._relation_noise_entity_groups = None
        self._last_relation_noise_mask = None
        self._last_relation_noise_manifest = None



    @staticmethod
    def _parse_device(raw_device) -> torch.device:
        if isinstance(raw_device, torch.device):
            return raw_device
        if isinstance(raw_device, int):
            return torch.device("cpu") if raw_device < 0 or (not torch.cuda.is_available()) else torch.device(f"cuda:{raw_device}")
        if isinstance(raw_device, str):
            d = raw_device.strip().lower()
            if d in ("cpu", "") or d.startswith("-"):
                return torch.device("cpu")
            if d.startswith("cuda"):
                if not torch.cuda.is_available():
                    return torch.device("cpu")
                return torch.device(d) if d != "cuda" else torch.device("cuda")
            if d.isdigit():
                return torch.device(f"cuda:{int(d)}") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _infer_dim(self, value, default):
        if torch.is_tensor(value):
            return int(value.shape[1])
        if isinstance(value, np.ndarray):
            return int(value.shape[1])
        if isinstance(value, (list, tuple)):
            return int(self._infer_dim(value[0], default)) if len(value) > 0 else int(default)
        return int(default)

    def _to_tensor(self, value, dtype):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return torch.as_tensor(value[0], dtype=dtype, device=self.device)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _ensure_mask(self, mask_like, feat: Optional[torch.Tensor]) -> torch.Tensor:
        if mask_like is None:
            if feat is None:
                return torch.ones((self.ent_num,), dtype=torch.bool, device=self.device)
            return (feat.abs().sum(dim=1) > 1e-6).to(torch.bool)
        if torch.is_tensor(mask_like):
            return mask_like.to(device=self.device).to(torch.bool)
        if isinstance(mask_like, np.ndarray):
            return torch.from_numpy(mask_like).to(device=self.device).bool()
        return torch.as_tensor(mask_like, dtype=torch.bool, device=self.device)

    def _resolve_modal_names(self) -> List[str]:
        names = ["gph"]
        if str(self.args.relation_intervention) != "removed":
            names.append("rel")
        names.extend(("attr", "img"))
        return names

    def _safe(self, x, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(x):
                return float(x.detach().mean().item()) if x.numel() > 0 else default
            return float(x)
        except Exception:
            return default

    def _batch_as_pair(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(batch):
            b = batch.to(self.device)
        else:
            b = torch.as_tensor(batch, dtype=torch.long, device=self.device)
        if b.dim() != 2 or b.size(1) != 2:
            raise ValueError("ICMF expects batch with shape [B,2]")
        return b[:, 0].long(), b[:, 1].long()

    def _to_float_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {k: self._safe(v) for k, v in d.items()}

    def _get_dvdc_target_disp(self) -> float:

        return self.dvdc_target_disp

    def _is_dvdc_active(self) -> bool:
        epoch = int(getattr(self, "_runtime_epoch", 0))
        if epoch < self.dvdc_start_epoch:
            return False
        return self.dvdc_end_epoch < 0 or epoch < self.dvdc_end_epoch

    def set_relation_noise_intervention(
        self,
        ratio: float,
        seed: int,
        entity_groups,
    ) -> None:

        ratio = float(ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("relation noise ratio must lie in (0, 1]")
        if int(seed) < 0:
            raise ValueError("relation noise seed must be non-negative")
        if not isinstance(entity_groups, (tuple, list)) or len(entity_groups) != 2:
            raise ValueError("relation noise requires left/right evaluation entity groups")
        self._relation_noise_ratio = ratio
        self._relation_noise_seed = int(seed)
        self._relation_noise_entity_groups = tuple(entity_groups)

    def clear_relation_noise_intervention(self) -> None:

        self._relation_noise_ratio = 0.0
        self._relation_noise_seed = 0
        self._relation_noise_entity_groups = None

    def _apply_relation_noise(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        self._last_relation_noise_mask = None
        self._last_relation_noise_manifest = None
        ratio = float(self._relation_noise_ratio)
        if self.training or ratio <= 0.0:
            return encoded
        if "rel" not in encoded:
            raise RuntimeError("relation-noise intervention requires the relation branch")

        relation = encoded["rel"]
        selected, manifest = _select_available_entities_by_side(
            relation.size(0),
            self._relation_noise_entity_groups,
            self.rel_mask,
            ratio,
            self._relation_noise_seed,
        )
        selected_ids = selected.nonzero(as_tuple=False).flatten()
        generator = torch.Generator(device=relation.device)
        generator.manual_seed(self._relation_noise_seed + 2_000_003)
        noise = torch.randn(
            (int(selected_ids.numel()), int(relation.size(1))),
            dtype=relation.dtype,
            device=relation.device,
            generator=generator,
        ) * relation.std()
        changed = relation.clone()
        changed[selected_ids] = noise
        encoded["rel"] = changed
        self._last_relation_noise_mask = selected
        self._last_relation_noise_manifest = manifest
        return encoded



    def _relation_features_for_current_phase(self) -> Optional[torch.Tensor]:

        warmup_epochs = int(
            getattr(self.args, "relation_permuted_warmup_epochs", 0)
        )
        use_permuted = (
            self.training
            and self.rel_features_permuted is not None
            and warmup_epochs > 0
            and int(getattr(self, "_runtime_epoch", 0)) < warmup_epochs
        )
        return self.rel_features_permuted if use_permuted else self.rel_features

    def _encode_modalities(self) -> Dict[str, torch.Tensor]:

        enc = self.multi_encoder
        encoded: Dict[str, torch.Tensor] = {}
        if "gph" in self.modal_names:
            entity_features = enc.entity_emb(self.input_idx)
            encoded["gph"] = enc.cross_graph_model(entity_features, self.adj)
        relation_features = self._relation_features_for_current_phase()
        if "rel" in self.modal_names and relation_features is not None:
            encoded["rel"] = enc.rel_fc(relation_features)
        if "attr" in self.modal_names and self.att_features is not None:
            encoded["attr"] = enc.att_fc(self.att_features)
        if "img" in self.modal_names and self.img_features is not None:
            encoded["img"] = enc.img_fc(self.img_features)
        return encoded

    def _compute_reliability_scores(self, projected: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        q_scores: Dict[str, torch.Tensor] = {}
        N = next(iter(projected.values())).size(0)

        for name in projected:
            if self.use_global_weights:
                logit = self.global_weight_logits[name]
                q = (F.softplus(logit) + self.eps).expand(N)
            else:
                out = self.reliability_nets[name](projected[name])
                q = F.softplus(out).squeeze(-1) + self.eps

            mask = self._masks.get(name, torch.ones(self.ent_num, dtype=torch.bool, device=self.device))
            q = q * mask.float()
            q_scores[name] = q
        return q_scores

    def _encode(self):


        encoded = self._encode_modalities()
        encoded = self._apply_relation_noise(encoded)


        projected: Dict[str, torch.Tensor] = {}
        for name, emb in encoded.items():
            if name in self.modal_proj:
                projected[name] = self.modal_proj[name](emb)
            else:
                projected[name] = emb


        non_gph = [n for n in self.modal_names if n != "gph" and n in projected]
        active = ["gph"] + non_gph
        gph_emb = projected["gph"]
        non_gph_embs = [projected[n] for n in non_gph]
        non_gph_masks = [self._masks.get(n) for n in non_gph] if self.use_key_mask else None

        hidden_states = self.cross_modal_layer(
            gph_emb, non_gph_embs, modal_masks=non_gph_masks,
        )


        mu_refs: Dict[str, torch.Tensor] = {"gph": hidden_states[:, 0, :]}
        for idx, name in enumerate(non_gph):
            mu_refs[name] = hidden_states[:, 1 + idx, :]

        if self.uniform_fusion:

            q_scores: Dict[str, torch.Tensor] = {}
            fusion_weights = _mask_aware_uniform_weights(
                active,
                self._masks,
                self.ent_num,
                gph_emb.device,
                gph_emb.dtype,
            )
        else:

            q_scores = self._compute_reliability_scores(projected)

            q_stack = torch.stack([q_scores[n] for n in active], dim=1)
            fusion_weights = q_stack / (q_stack.sum(dim=-1, keepdim=True) + self.eps)




            if self.weight_floor > 0.0:
                availability = torch.stack(
                    [
                        self._masks.get(
                            name,
                            torch.ones(
                                gph_emb.size(0), dtype=torch.bool, device=gph_emb.device
                            ),
                        )
                        for name in active
                    ],
                    dim=1,
                )
                floored = torch.where(
                    availability,
                    fusion_weights.clamp_min(self.weight_floor),
                    fusion_weights,
                )
                fusion_weights = floored / (
                    floored.sum(dim=-1, keepdim=True) + self.eps
                )

        parts = []
        for idx, name in enumerate(active):
            w = fusion_weights[:, idx:idx+1]
            parts.append(w * F.normalize(mu_refs[name], dim=-1))
        joint_emb = torch.cat(parts, dim=-1)


        self.joint_emb = joint_emb
        self.q_dict = q_scores
        self.mu_dict = encoded
        self.projected_dict = projected
        self.mu_ref_dict = mu_refs
        self.fusion_weights = fusion_weights
        self._active_names = active

        return joint_emb

    def _calc_losses(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        left, right = self._batch_as_pair(batch)
        zero = torch.tensor(0.0, device=self.device)
        pairs = torch.stack([left, right], dim=1)


        loss_ea = self.ea_criterion(self.joint_emb, pairs, norm=True)



        _modal_src = self.mu_dict
        modal_order = self.modal_names
        raw_losses: List[torch.Tensor] = []
        valid_modal_losses: List[bool] = []
        for name in modal_order:
            mu = _modal_src.get(name)
            if mu is None:
                raw_losses.append(zero)
                valid_modal_losses.append(False)
                continue

            modal_pairs = pairs
            m_mask = self._masks.get(name)
            if m_mask is not None:
                valid = m_mask[pairs[:, 0]] & m_mask[pairs[:, 1]]
                modal_pairs = pairs[valid]

            if modal_pairs.size(0) < 2:
                raw_losses.append(zero)
                valid_modal_losses.append(False)
                continue
            raw_losses.append(self.modal_cl_criterion(mu, modal_pairs, norm=True))
            valid_modal_losses.append(True)


        loss_modal = self.multi_loss_layer(
            raw_losses, valid_mask=valid_modal_losses
        )









        q_all = []
        for name in self.q_dict:
            q = self.q_dict[name]
            mask = self._masks.get(name, torch.ones(q.shape[0], dtype=torch.bool, device=self.device))
            q_active = q[mask]
            if q_active.numel() > 0:
                q_all.append(q_active)
        if self.uniform_fusion or self.disp_type == "none":
            loss_disp = zero
        elif q_all:
            q_cat = torch.cat(q_all, dim=0).clamp_min(self.eps)
            loss_disp = (q_cat.log() ** 2).mean()
        else:
            loss_disp = zero


        dvdc_lambda = 0.0
        self._current_disp = 0.0
        current_target_disp = self._get_dvdc_target_disp()
        fixed_lam = float(getattr(self.args, 'fixed_lambda', -1.0))
        dvdc_active = False
        if (not self.uniform_fusion) and fixed_lam > 0:
            dvdc_lambda = fixed_lam
            self._current_disp = loss_disp.detach().item()
        elif self.use_dvdc and self._is_dvdc_active():
            dvdc_active = True
            raw_disp = loss_disp.detach().item()
            if self._dvdc_disp_ema is None:
                self._dvdc_disp_ema = raw_disp
            else:
                self._dvdc_disp_ema = (self._dvdc_ema_momentum * self._dvdc_disp_ema
                                       + (1 - self._dvdc_ema_momentum) * raw_disp)
            self._current_disp = self._dvdc_disp_ema
            disp_gap = self._current_disp - current_target_disp
            self._dvdc_log_lambda += self.dvdc_lr * disp_gap
            self._dvdc_log_lambda = max(-5.0, min(5.0, self._dvdc_log_lambda))
            dvdc_lambda = math.exp(self._dvdc_log_lambda)
        total = loss_ea + loss_modal + dvdc_lambda * loss_disp


        with torch.no_grad():
            stats: Dict[str, torch.Tensor] = {
                "loss_ea": loss_ea,
                "loss_modal": loss_modal,
                "dvdc_disp": loss_disp,
                "dvdc_lambda": torch.tensor(dvdc_lambda, device=self.device),
                "dvdc_disp_ema": torch.tensor(self._current_disp, device=self.device),
                "dvdc_target_disp_now": torch.tensor(current_target_disp, device=self.device),
                "dvdc_active": torch.tensor(float(dvdc_active), device=self.device),
            }
            for n in self.modal_names:
                if n in self.q_dict:
                    stats[f"q_{n}"] = self.q_dict[n].mean()
            if self.fusion_weights is not None:
                active = getattr(self, '_active_names', self.modal_names)
                fw = self.fusion_weights.mean(dim=0)
                valid_mask = torch.stack(
                    [
                        self._masks.get(
                            name,
                            torch.ones(
                                self.fusion_weights.size(0),
                                dtype=torch.bool,
                                device=self.fusion_weights.device,
                            ),
                        ).to(self.fusion_weights.device)
                        for name in active
                    ],
                    dim=1,
                )
                masked_weights = self.fusion_weights.masked_fill(~valid_mask, -1.0)
                stats["w_entity_max_mean"] = masked_weights.max(dim=1).values.mean()
                stats["w_valid_lt_001"] = (
                    ((self.fusion_weights < 0.01) & valid_mask).float().sum()
                    / valid_mask.float().sum().clamp_min(1.0)
                )
                stats["w_entity_entropy_mean"] = -(
                    self.fusion_weights
                    * self.fusion_weights.clamp_min(self.eps).log()
                ).sum(dim=1).mean()
                for idx, name in enumerate(active):
                    if idx < fw.size(0):
                        stats[f"w_{name}"] = fw[idx]
                        active_weights = self.fusion_weights[valid_mask[:, idx], idx]
                        if active_weights.numel() > 0:
                            stats[f"w_{name}_median"] = active_weights.median()
                            stats[f"w_{name}_lt_005"] = (
                                active_weights < 0.05
                            ).float().mean()

            lv = self.multi_loss_layer.log_vars
            for idx, name in enumerate(self.modal_names):
                if idx < lv.size(0):
                    stats[f"kendall_s_{name}"] = lv[idx]
                    stats[f"kendall_w_{name}"] = torch.exp(-lv[idx])
        return total, stats



    def encode_once(self):

        self._encode()

    def forward(self, batch, skip_encode=False):

        if not skip_encode:
            self._encode()

        loss, stats = self._calc_losses(batch)
        return loss, {
            "loss_dic": self._to_float_dict(stats),
            "modal_names": list(self.modal_names),
        }

    def generate_joint_emb(self):

        self._encode()

        if not self.training and self.q_dict:
            parts = []
            for n in self.modal_names:
                if n in self.q_dict:
                    q_mean = self.q_dict[n].mean().item()
                    parts.append(f"{n}={q_mean:.4f}")
            if parts:
                print(f"  Reliability scores: {', '.join(parts)}")
        return self.joint_emb, self.fusion_weights

    @torch.no_grad()
    def fusion_statistics(self, entity_ids=None) -> Dict[str, object]:

        if self.fusion_weights is None:
            return {}
        active = list(getattr(self, "_active_names", self.modal_names))
        weights = self.fusion_weights
        if entity_ids is None:
            ids = torch.arange(weights.size(0), device=weights.device)
        else:
            ids = torch.as_tensor(entity_ids, dtype=torch.long, device=weights.device)
            ids = torch.unique(ids)
        selected = weights[ids]
        availability = torch.stack(
            [
                self._masks.get(
                    name,
                    torch.ones(weights.size(0), dtype=torch.bool, device=weights.device),
                )[ids]
                for name in active
            ],
            dim=1,
        )
        masked = selected.masked_fill(~availability, -1.0)
        valid_count = availability.sum().clamp_min(1)
        result: Dict[str, object] = {
            "entity_count": int(ids.numel()),
            "active_modalities": active,
            "per_entity_max_weight_mean": float(masked.max(dim=1).values.mean().item()),
            "valid_weight_fraction_lt_0_01": float(
                (((selected < 0.01) & availability).sum() / valid_count).item()
            ),
            "per_entity_weight_entropy_mean": float(
                (-(selected * selected.clamp_min(self.eps).log()).sum(dim=1).mean()).item()
            ),
            "per_modality": {},
        }
        per_modality = result["per_modality"]
        for index, name in enumerate(active):
            modality_weights = selected[availability[:, index], index]
            if modality_weights.numel() == 0:
                continue
            modality_mean = float(modality_weights.mean().item())
            modality_std = float(modality_weights.std(unbiased=False).item())
            per_modality[name] = {
                "count": int(modality_weights.numel()),
                "mean": modality_mean,
                "median": float(modality_weights.median().item()),
                "std": modality_std,

                "cv": (modality_std / modality_mean) if abs(modality_mean) > 1e-12 else None,
                "fraction_lt_0_01": float((modality_weights < 0.01).float().mean().item()),
                "fraction_lt_0_05": float((modality_weights < 0.05).float().mean().item()),
            }

        modality_means = [entry["mean"] for entry in per_modality.values()]
        modality_cvs = [
            entry["cv"] for entry in per_modality.values() if entry["cv"] is not None
        ]
        if modality_means:
            means = torch.tensor(modality_means, dtype=torch.float64)
            result["cross_modality_mean_weight_sd"] = float(
                means.std(unbiased=False).item()
            )
        if modality_cvs:
            cvs = torch.tensor(modality_cvs, dtype=torch.float64)
            result["within_modality_cv_mean"] = float(cvs.mean().item())
        return result

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=None):
        if new_links is None:
            new_links = []
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links

        mu_c = final_emb

        left_ids = torch.as_tensor(left_non_train, dtype=torch.long, device=self.device)
        right_ids = torch.as_tensor(right_non_train, dtype=torch.long, device=self.device)

        bs = 512
        right_emb = mu_c[right_ids]
        left_emb = mu_c[left_ids]

        if bool(getattr(self.args, "csls", False)):
            preds_l, preds_r = _chunked_csls_nearest_neighbors(
                left_emb,
                right_emb,
                k=int(getattr(self.args, "csls_k", 3)),
                batch_size=bs,
            )
        else:
            preds_l = []
            for start in range(0, len(left_ids), bs):
                d = pairwise_distances(left_emb[start:start + bs], right_emb)
                preds_l.extend(torch.argmin(d, dim=1).cpu().tolist())
                del d

            preds_r = []
            for start in range(0, len(right_ids), bs):
                d = pairwise_distances(right_emb[start:start + bs], left_emb)
                preds_r.extend(torch.argmin(d, dim=1).cpu().tolist())
                del d

        del right_emb
        del left_emb

        refresh_multiplier = 10
        is_fresh = (epoch + 1) % (int(self.args.semi_learn_step) * refresh_multiplier) == int(self.args.semi_learn_step)
        if is_fresh:
            links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            selected = set((int(a), int(b)) for a, b in new_links)
            links = [
                (left_non_train[i], right_non_train[p])
                for i, p in enumerate(preds_l)
                if preds_r[p] == i and (left_non_train[i], right_non_train[p]) in selected
            ]

        return links

    def data_refresh(self, logger, train_ill, left_non_train, right_non_train, new_links=None):
        if new_links is None:
            new_links = []
        if len(new_links) == 0 or len(left_non_train) == 0 or len(right_non_train) == 0:
            if logger is not None:
                logger.info("len(new_links) is 0")
            return left_non_train, right_non_train, np.asarray(train_ill, dtype=np.int32), []

        new_links_select = list(new_links)
        train_np = np.asarray(train_ill, dtype=np.int64)
        add = np.asarray(new_links_select, dtype=np.int64)
        train_np = np.vstack((train_np, add)) if train_np.size else add

        left_set = set(left_non_train)
        right_set = set(right_non_train)
        for a, b in new_links_select:
            left_set.discard(int(a))
            right_set.discard(int(b))

        left_non_train = list(left_set)
        right_non_train = list(right_set)

        if logger is not None:
            logger.info(f"#new_links_select:{len(new_links_select)}")
            logger.info(f"train_ill.shape:{train_np.shape}")
            logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

        return left_non_train, right_non_train, train_np, []
