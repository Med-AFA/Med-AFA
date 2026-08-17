import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from acquisition_model.cmi_estimator import CMIEstimator
from acquisition_model.utils import get_entropy, ind_to_onehot


class OracleQEstimator(CMIEstimator):


    def __init__(
        self,
        *args,
        oracle_target_fn=None,
        prerequisite_matrix=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.oracle_target_fn = oracle_target_fn
        if prerequisite_matrix is None:
            prerequisite_matrix = torch.zeros((self.mask_size, self.mask_size), dtype=torch.float32)
        elif isinstance(prerequisite_matrix, np.ndarray):
            prerequisite_matrix = torch.tensor(prerequisite_matrix, dtype=torch.float32)
        self.register_buffer("prerequisite_matrix", prerequisite_matrix.float())

    def _legal_action_mask(self, selected_action_mask: torch.Tensor) -> torch.Tensor:
        selected = selected_action_mask > 0.5
        prereq = self.prerequisite_matrix.to(device=selected_action_mask.device, dtype=selected_action_mask.dtype)
        if prereq.numel() == 0 or prereq.shape[0] != self.mask_size:
            return ~selected
        required_counts = prereq.sum(dim=1).unsqueeze(0)
        satisfied_counts = selected_action_mask.to(dtype=prereq.dtype) @ prereq.t()
        prereq_satisfied = satisfied_counts >= (required_counts - 1.0e-6)
        return (~selected) & prereq_satisfied

    def _mask_illegal_scores(self, scores: torch.Tensor, selected_action_mask: torch.Tensor) -> torch.Tensor:
        legal = self._legal_action_mask(selected_action_mask).to(device=scores.device)
        return scores.masked_fill(~legal.bool(), -1.0e6)

    def _sample_random_legal_actions(self, selected_action_mask: torch.Tensor) -> torch.Tensor:
        legal = self._legal_action_mask(selected_action_mask).float()
        row_sums = legal.sum(dim=1, keepdim=True)
        if bool((row_sums <= 0).any()):
            fallback = (selected_action_mask <= 0.5).float()
            legal = torch.where(row_sums <= 0, fallback, legal)
            row_sums = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = legal / row_sums
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _predict_q(self, x_masked: torch.Tensor, pred_without_next_feature: torch.Tensor) -> torch.Tensor:
        if self.cmi_scaling == "bounded":
            entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
            return self.value_network(x_masked).sigmoid() * entropy
        if self.cmi_scaling == "positive":
            return torch.nn.functional.softplus(self.value_network(x_masked))
        return self.value_network(x_masked)

    def _oracle_targets(
        self,
        sample_indices: Optional[torch.Tensor],
        mask_before_action: torch.Tensor,
        actions: torch.Tensor,
        fallback_delta: torch.Tensor,
    ) -> torch.Tensor:
        if self.oracle_target_fn is None or sample_indices is None:
            return fallback_delta
        targets = self.oracle_target_fn(
            sample_indices.detach().cpu(),
            mask_before_action.detach().cpu(),
            actions.detach().cpu(),
        )
        return torch.as_tensor(targets, dtype=fallback_delta.dtype, device=fallback_delta.device)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        if len(batch) == 3:
            x, y, sample_indices = batch
        else:
            x, y = batch
            sample_indices = None

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0

        x_masked = self.mask_layer(x, mask)
        pred_without_next_feature = self.predictor(x_masked)
        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)
        pred_loss = loss_without_next_feature.mean()
        pred_loss_total += pred_loss.detach()
        self.manual_backward(pred_loss / (self.max_features + 1))
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        for _ in range(self.max_features):
            mask_before_action = mask.clone()
            x_masked = self.mask_layer(x, mask_before_action)
            pred_q = self._predict_q(x_masked, pred_without_next_feature)

            scores = self._mask_illegal_scores(pred_q / self.feature_costs, mask_before_action)
            best = torch.argmax(scores, dim=1)
            random = self._sample_random_legal_actions(mask_before_action)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).long()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask_before_action, ind_to_onehot(actions, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred_with_next_feature = self.predictor(x_masked)
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            fallback_delta = loss_without_next_feature - loss_with_next_feature.detach()
            target_q = self._oracle_targets(sample_indices, mask_before_action, actions, fallback_delta)
            value_network_loss = nn.functional.mse_loss(pred_q[torch.arange(len(x)), actions], target_q)

            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
            self.manual_backward(total_loss / (self.max_features + 1))

            value_network_loss_total += torch.mean(value_network_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

        opt.step()
        return {
            "value_network_loss": value_network_loss_total / self.max_features,
            "predictor_loss": pred_loss_total / (self.max_features + 1),
        }


class DoubleHeadOracleQEstimator(OracleQEstimator):


    def __init__(
        self,
        *args,
        oracle_double_target_fn=None,
        full_path_loss_weight: float = 1.0,
        proposal_top_k: int = 3,
        one_step_prefix_steps: int = 0,
        full_path_middle_steps: int = 0,
        intervention_aux_enabled: bool = True,
        intervention_aux_weight: float = 0.10,
        intervention_aux_only_changed_actions: bool = False,
        intervention_aux_mode: str = "oracle_positive_full_only",
        intervention_aux_oracle_margin: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, oracle_target_fn=None, **kwargs)
        self.oracle_double_target_fn = oracle_double_target_fn
        self.full_path_loss_weight = float(max(float(full_path_loss_weight), 0.0))
        self.proposal_top_k = int(max(1, proposal_top_k))
        self.one_step_prefix_steps = int(max(0, one_step_prefix_steps))
        self.full_path_middle_steps = int(max(0, full_path_middle_steps))
        self.intervention_aux_enabled = bool(intervention_aux_enabled)
        self.intervention_aux_weight = float(max(0.0, intervention_aux_weight))
        self.intervention_aux_only_changed_actions = bool(intervention_aux_only_changed_actions)
        if intervention_aux_mode not in {"one_full_ce", "oracle_positive_full_only"}:
            raise ValueError("intervention_aux_mode must be one of 'one_full_ce' or 'oracle_positive_full_only'")
        self.intervention_aux_mode = str(intervention_aux_mode)
        self.intervention_aux_oracle_margin = float(intervention_aux_oracle_margin)
        self.intervention_aux_history = {
            "steps": 0,
            "active_calls": 0,
            "active_count_sum": 0.0,
            "changed_count_sum": 0.0,
            "loss_sum": 0.0,
            "one_loss_sum": 0.0,
            "full_loss_sum": 0.0,
            "one_acc_sum": 0.0,
            "full_acc_sum": 0.0,
        }

    def _use_full_path_for_step(self, selected_action_mask: torch.Tensor) -> torch.Tensor:
        step_idx = selected_action_mask.sum(dim=1).long()
        start = int(self.one_step_prefix_steps)
        end = int(self.one_step_prefix_steps + self.full_path_middle_steps)
        if end <= start:
            return torch.zeros_like(step_idx, dtype=torch.bool)
        return (step_idx >= start) & (step_idx < end)

    def _split_raw_heads(self, x_masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.value_network(x_masked)
        expected = int(self.mask_size) * 2
        if raw.ndim != 2 or int(raw.shape[1]) != expected:
            raise RuntimeError(
                f"DoubleHeadOracleQEstimator expects value_network output shape [batch, {expected}], "
                f"got {tuple(raw.shape)}"
            )
        return raw[:, : self.mask_size], raw[:, self.mask_size :]

    def _scale_head(self, raw_head: torch.Tensor, pred_without_next_feature: torch.Tensor) -> torch.Tensor:
        if self.cmi_scaling == "bounded":
            entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
            return raw_head.sigmoid() * entropy
        if self.cmi_scaling == "positive":
            return torch.nn.functional.softplus(raw_head)
        return raw_head

    def _predict_heads(
        self,
        x_masked: torch.Tensor,
        pred_without_next_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        one_raw, full_raw = self._split_raw_heads(x_masked)
        return (
            self._scale_head(one_raw, pred_without_next_feature),
            self._scale_head(full_raw, pred_without_next_feature),
        )

    def _legal_action_mask_with_matrix(
        self,
        selected_action_mask: torch.Tensor,
        prerequisite_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        selected = selected_action_mask > 0.5
        if prerequisite_matrix is None:
            return ~selected
        prereq = prerequisite_matrix.to(device=selected_action_mask.device, dtype=selected_action_mask.dtype)
        if prereq.numel() == 0 or prereq.shape[0] != self.mask_size:
            return ~selected
        required_counts = prereq.sum(dim=1).unsqueeze(0)
        satisfied_counts = selected_action_mask.to(dtype=prereq.dtype) @ prereq.t()
        prereq_satisfied = satisfied_counts >= (required_counts - 1.0e-6)
        return (~selected) & prereq_satisfied

    def _double_head_policy_scores(
        self,
        one_step_q: torch.Tensor,
        full_path_q: torch.Tensor,
        selected_action_mask: torch.Tensor,
        prerequisite_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        legal = self._legal_action_mask_with_matrix(selected_action_mask, prerequisite_matrix)
        one_step_scores = one_step_q / self.feature_costs
        one_step_scores = one_step_scores.masked_fill(~legal.bool(), -1.0e6)
        k = int(min(max(1, self.proposal_top_k), self.mask_size))
        proposal_idx = torch.topk(one_step_scores, k=k, dim=1).indices
        proposal_mask = torch.zeros_like(one_step_scores, dtype=torch.bool)
        proposal_mask.scatter_(1, proposal_idx, True)
        final_legal = proposal_mask & legal.bool()
        full_scores = full_path_q / self.feature_costs
        return full_scores.masked_fill(~final_legal, -1.0e6)

    def _schedule_policy_scores(
        self,
        one_step_q: torch.Tensor,
        full_path_q: torch.Tensor,
        selected_action_mask: torch.Tensor,
        prerequisite_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        legal = self._legal_action_mask_with_matrix(selected_action_mask, prerequisite_matrix)
        one_step_scores = one_step_q / self.feature_costs
        one_step_scores = one_step_scores.masked_fill(~legal.bool(), -1.0e6)
        rerank_scores = self._double_head_policy_scores(
            one_step_q,
            full_path_q,
            selected_action_mask,
            prerequisite_matrix=prerequisite_matrix,
        )
        use_full_path = self._use_full_path_for_step(selected_action_mask).unsqueeze(1)
        return torch.where(use_full_path, rerank_scores, one_step_scores)

    def predict_policy_scores(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pred: torch.Tensor,
        prerequisite_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_masked = self.mask_layer(x, mask)
        one_step_q, full_path_q = self._predict_heads(x_masked, pred)
        return self._schedule_policy_scores(
            one_step_q,
            full_path_q,
            mask,
            prerequisite_matrix=prerequisite_matrix,
        )

    def _oracle_double_targets(
        self,
        sample_indices: Optional[torch.Tensor],
        mask_before_action: torch.Tensor,
        actions: torch.Tensor,
        fallback_delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.oracle_double_target_fn is None or sample_indices is None:
            ones = torch.ones_like(fallback_delta)
            return fallback_delta, fallback_delta, ones
        targets = self.oracle_double_target_fn(
            sample_indices.detach().cpu(),
            mask_before_action.detach().cpu(),
            actions.detach().cpu(),
        )
        one_step = torch.as_tensor(targets["one_step"], dtype=fallback_delta.dtype, device=fallback_delta.device)
        full_path = torch.as_tensor(targets["full_path"], dtype=fallback_delta.dtype, device=fallback_delta.device)
        full_mask = torch.as_tensor(
            targets["full_path_mask"],
            dtype=fallback_delta.dtype,
            device=fallback_delta.device,
        )
        return one_step, full_path, full_mask

    def _zero_intervention_aux_stats(self, device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        return {
            "loss": zero,
            "one_loss": zero,
            "full_loss": zero,
            "one_acc": zero,
            "full_acc": zero,
            "active_call": zero,
            "active_count": zero,
            "changed_count": zero,
        }

    def _merge_intervention_aux_stats(
        self,
        acc: Dict[str, torch.Tensor],
        item: Dict[str, torch.Tensor],
    ) -> None:
        for key, value in item.items():
            acc[key] = acc.get(key, value * 0.0) + value.detach()

    def _intervention_predictor_auxiliary_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_indices: Optional[torch.Tensor],
        mask_before_action: torch.Tensor,
        one_step_q: torch.Tensor,
        full_path_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats = self._zero_intervention_aux_stats(x.device)
        zero = x.sum() * 0.0
        if not self.intervention_aux_enabled or self.intervention_aux_weight <= 0:
            return zero, stats

        use_full_path = self._use_full_path_for_step(mask_before_action)
        if not bool(use_full_path.any()):
            return zero, stats

        legal = self._legal_action_mask_with_matrix(mask_before_action, self.prerequisite_matrix)
        one_scores = (one_step_q / self.feature_costs).masked_fill(~legal.bool(), -1.0e6)
        one_actions = torch.argmax(one_scores, dim=1)
        full_scores = self._double_head_policy_scores(
            one_step_q,
            full_path_q,
            mask_before_action,
            prerequisite_matrix=self.prerequisite_matrix,
        )
        full_actions = torch.argmax(full_scores, dim=1)
        changed = one_actions != full_actions
        active = use_full_path & legal.any(dim=1)
        if self.intervention_aux_only_changed_actions:
            active = active & changed
        if self.intervention_aux_mode == "oracle_positive_full_only":
            active = active & changed
            if self.oracle_double_target_fn is None or sample_indices is None:
                active = torch.zeros_like(active)
            else:
                fallback = torch.zeros(len(x), dtype=one_step_q.dtype, device=x.device)
                _, one_full_target, one_full_mask = self._oracle_double_targets(
                    sample_indices,
                    mask_before_action,
                    one_actions,
                    fallback,
                )
                _, full_full_target, full_full_mask = self._oracle_double_targets(
                    sample_indices,
                    mask_before_action,
                    full_actions,
                    fallback,
                )
                oracle_supported = (one_full_mask > 0) & (full_full_mask > 0)
                oracle_positive = (full_full_target - one_full_target) > float(self.intervention_aux_oracle_margin)
                active = active & oracle_supported.bool() & oracle_positive.bool()
        if not bool(active.any()):
            stats["changed_count"] = changed.float().sum().detach()
            return zero, stats

        one_mask = torch.max(mask_before_action, ind_to_onehot(one_actions, self.mask_size).to(dtype=mask_before_action.dtype))
        full_mask = torch.max(mask_before_action, ind_to_onehot(full_actions, self.mask_size).to(dtype=mask_before_action.dtype))
        full_pred = self.predictor(self.mask_layer(x, full_mask))
        full_loss_vec = self.loss_fn(full_pred, y)
        active_float = active.to(dtype=full_loss_vec.dtype)
        denom = active_float.sum().clamp_min(1.0)
        full_loss = (full_loss_vec * active_float).sum() / denom
        full_acc = ((full_pred.argmax(dim=1) == y).to(dtype=active_float.dtype) * active_float).sum() / denom
        if self.intervention_aux_mode == "oracle_positive_full_only":
            with torch.no_grad():
                one_pred = self.predictor(self.mask_layer(x, one_mask))
                one_loss_vec = self.loss_fn(one_pred, y)
                one_loss = (one_loss_vec * active_float).sum() / denom
                one_acc = ((one_pred.argmax(dim=1) == y).to(dtype=active_float.dtype) * active_float).sum() / denom
            base_loss = full_loss
        else:
            one_pred = self.predictor(self.mask_layer(x, one_mask))
            one_loss_vec = self.loss_fn(one_pred, y)
            one_loss = (one_loss_vec * active_float).sum() / denom
            one_acc = ((one_pred.argmax(dim=1) == y).to(dtype=active_float.dtype) * active_float).sum() / denom
            base_loss = 0.5 * (one_loss + full_loss)
        weighted_loss = float(self.intervention_aux_weight) * base_loss

        stats["loss"] = weighted_loss.detach()
        stats["one_loss"] = one_loss.detach()
        stats["full_loss"] = full_loss.detach()
        stats["one_acc"] = one_acc.detach()
        stats["full_acc"] = full_acc.detach()
        stats["active_call"] = torch.tensor(1.0, device=x.device)
        stats["active_count"] = active_float.sum().detach()
        stats["changed_count"] = (changed & use_full_path).float().sum().detach()
        return weighted_loss, stats

    def _accumulate_intervention_aux_history(self, stats: Dict[str, torch.Tensor]) -> None:
        if not self.intervention_aux_enabled:
            return
        self.intervention_aux_history["steps"] += 1
        active_call = int(float(stats.get("active_call", torch.tensor(0.0)).detach().cpu()))
        self.intervention_aux_history["active_calls"] += active_call
        self.intervention_aux_history["active_count_sum"] += float(
            stats.get("active_count", torch.tensor(0.0)).detach().cpu()
        )
        self.intervention_aux_history["changed_count_sum"] += float(
            stats.get("changed_count", torch.tensor(0.0)).detach().cpu()
        )
        if active_call > 0:
            for key in ("loss", "one_loss", "full_loss", "one_acc", "full_acc"):
                self.intervention_aux_history[f"{key}_sum"] += float(
                    stats.get(key, torch.tensor(0.0)).detach().cpu()
                )

    def intervention_aux_summary(self) -> Dict[str, Optional[float]]:
        hist = self.intervention_aux_history
        steps = int(hist.get("steps", 0))
        calls = int(hist.get("active_calls", 0))

        def mean_for(key: str, denom: int) -> Optional[float]:
            if denom <= 0:
                return None
            return float(hist.get(key, 0.0) / denom)

        return {
            "enabled": bool(self.intervention_aux_enabled),
            "weight": float(self.intervention_aux_weight),
            "only_changed_actions": bool(self.intervention_aux_only_changed_actions),
            "mode": str(self.intervention_aux_mode),
            "oracle_margin": float(self.intervention_aux_oracle_margin),
            "steps": int(steps),
            "active_calls": int(calls),
            "active_count_mean_per_training_batch": mean_for("active_count_sum", steps),
            "changed_count_mean_per_training_batch": mean_for("changed_count_sum", steps),
            "loss_mean_per_active_call": mean_for("loss_sum", calls),
            "one_loss_mean_per_active_call": mean_for("one_loss_sum", calls),
            "full_loss_mean_per_active_call": mean_for("full_loss_sum", calls),
            "one_acc_mean_per_active_call": mean_for("one_acc_sum", calls),
            "full_acc_mean_per_active_call": mean_for("full_acc_sum", calls),
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        if len(batch) == 3:
            x, y, sample_indices = batch
        else:
            x, y = batch
            sample_indices = None

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        one_step_value_loss_total = 0
        full_path_value_loss_total = 0
        pred_loss_total = 0
        intervention_aux_stats_total: Dict[str, torch.Tensor] = {}

        x_masked = self.mask_layer(x, mask)
        pred_without_next_feature = self.predictor(x_masked)
        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)
        pred_loss = loss_without_next_feature.mean()
        pred_loss_total += pred_loss.detach()
        self.manual_backward(pred_loss / (self.max_features + 1))
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        for _ in range(self.max_features):
            mask_before_action = mask.clone()
            x_masked = self.mask_layer(x, mask_before_action)
            one_step_q, full_path_q = self._predict_heads(x_masked, pred_without_next_feature)

            scores = self._schedule_policy_scores(
                one_step_q,
                full_path_q,
                mask_before_action,
                prerequisite_matrix=self.prerequisite_matrix,
            )
            best = torch.argmax(scores, dim=1)
            random = self._sample_random_legal_actions(mask_before_action)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).long()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask_before_action, ind_to_onehot(actions, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred_with_next_feature = self.predictor(x_masked)
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            fallback_delta = loss_without_next_feature - loss_with_next_feature.detach()
            target_one_step, target_full_path, full_path_mask = self._oracle_double_targets(
                sample_indices,
                mask_before_action,
                actions,
                fallback_delta,
            )
            row_idx = torch.arange(len(x), device=x.device)
            pred_one_step = one_step_q[row_idx, actions]
            pred_full_path = full_path_q[row_idx, actions]
            one_step_value_loss = nn.functional.mse_loss(pred_one_step, target_one_step)
            if bool((full_path_mask > 0).any()):
                full_diff = (pred_full_path - target_full_path).pow(2) * full_path_mask
                full_path_value_loss = full_diff.sum() / full_path_mask.sum().clamp_min(1.0)
            else:
                full_path_value_loss = pred_full_path.sum() * 0.0
            value_network_loss = one_step_value_loss + self.full_path_loss_weight * full_path_value_loss

            intervention_aux_loss, intervention_aux_stats = self._intervention_predictor_auxiliary_loss(
                x,
                y,
                sample_indices,
                mask_before_action,
                one_step_q,
                full_path_q,
            )
            self._merge_intervention_aux_stats(intervention_aux_stats_total, intervention_aux_stats)
            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature) + intervention_aux_loss
            self.manual_backward(total_loss / (self.max_features + 1))

            value_network_loss_total += torch.mean(value_network_loss)
            one_step_value_loss_total += torch.mean(one_step_value_loss)
            full_path_value_loss_total += torch.mean(full_path_value_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

        opt.step()
        self._accumulate_intervention_aux_history(intervention_aux_stats_total)
        return {
            "value_network_loss": value_network_loss_total / self.max_features,
            "one_step_value_loss": one_step_value_loss_total / self.max_features,
            "full_path_value_loss": full_path_value_loss_total / self.max_features,
            "predictor_loss": pred_loss_total / (self.max_features + 1),
            "intervention_aux_loss": intervention_aux_stats_total.get(
                "loss",
                torch.tensor(0.0, device=x.device),
            )
            / (self.max_features + 1),
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        pred_list = [pred]

        for _ in range(self.max_features):
            scores = self.predict_policy_scores(
                x,
                mask,
                pred,
                prerequisite_matrix=self.prerequisite_matrix,
            )
            best_feature_index = torch.argmax(scores, dim=1)
            mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked)
            pred_list.append(pred)

        self._val_epoch_outputs.append((pred_list, y))
