from __future__ import annotations

import numpy as np
import torch

from mappo import MAPPOPolicy, RolloutBatch


def update_policy(
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    train_cfg: dict,
) -> dict:
    device = policy.device
    batch_size = int(batch.states.shape[0])
    minibatch_size = min(int(train_cfg["minibatch_size"]), batch_size)
    update_epochs = int(train_cfg["update_epochs"])
    clip_ratio = float(train_cfg["clip_ratio"])
    value_clip = float(train_cfg["value_clip"])
    entropy_coef = float(train_cfg["entropy_coef"])
    value_loss_coef = float(train_cfg["value_loss_coef"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    target_kl = train_cfg.get("target_kl")
    if target_kl is not None:
        target_kl = max(float(target_kl), 1e-6)

    advantages = batch.advantages.astype(np.float32, copy=True)
    advantage_masks = batch.masks.astype(np.float32, copy=False)
    if advantages.ndim == 2:
        valid_advantages = advantages[advantage_masks > 0.0]
        if valid_advantages.size > 0:
            advantages = (
                (advantages - float(valid_advantages.mean()))
                / (float(valid_advantages.std()) + 1e-8)
            )
    else:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
    states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
    raw_actions = torch.as_tensor(batch.raw_actions, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch.masks, dtype=torch.float32, device=device)
    old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)

    metrics = {
        "loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    indices = np.arange(batch_size)
    stop_early = False

    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            stop = start + minibatch_size
            batch_indices = indices[start:stop]

            obs_mb = obs[batch_indices]
            states_mb = states[batch_indices]
            raw_actions_mb = raw_actions[batch_indices]
            old_log_probs_mb = old_log_probs[batch_indices]
            masks_mb = masks[batch_indices]
            old_values_mb = old_values[batch_indices]
            returns_mb = returns[batch_indices]
            advantages_mb = advantages[batch_indices]

            max_agents = masks_mb.shape[1]
            obs_flat = obs_mb.reshape(-1, policy.obs_dim)
            raw_actions_flat = raw_actions_mb.reshape(-1, policy.action_dim)
            mask_flat = masks_mb.reshape(-1)
            old_log_probs_flat = old_log_probs_mb.reshape(-1)
            if advantages_mb.ndim == 1:
                adv_flat = advantages_mb.unsqueeze(1).expand(-1, max_agents).reshape(-1)
            else:
                adv_flat = advantages_mb.reshape(-1)

            _, new_log_probs_flat, entropy_flat = policy.evaluate_actor(
                obs_flat,
                raw_actions_flat,
            )
            pre_step_kl = (
                ((old_log_probs_flat - new_log_probs_flat) * mask_flat).sum()
                / mask_flat.sum().clamp_min(1.0)
            )
            if target_kl is not None and abs(float(pre_step_kl.item())) > target_kl:
                stop_early = True
                break

            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            unclipped = ratio * adv_flat
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_flat

            valid_count = mask_flat.sum().clamp_min(1.0)
            actor_loss = -(
                torch.minimum(unclipped, clipped) * mask_flat
            ).sum() / valid_count
            entropy = (entropy_flat * mask_flat).sum() / valid_count

            values_pred = policy.evaluate_critic(states_mb)
            if value_clip > 0.0:
                value_delta = torch.clamp(
                    values_pred - old_values_mb,
                    -value_clip,
                    value_clip,
                )
                values_clipped = old_values_mb + value_delta
                value_loss_unclipped = (values_pred - returns_mb).pow(2)
                value_loss_clipped = (values_clipped - returns_mb).pow(2)
                value_loss = torch.maximum(
                    value_loss_unclipped,
                    value_loss_clipped,
                )
            else:
                value_loss = (values_pred - returns_mb).pow(2)

            if value_loss.ndim == 2:
                critic_loss = 0.5 * (
                    (value_loss * masks_mb).sum() / masks_mb.sum().clamp_min(1.0)
                )
            else:
                critic_loss = 0.5 * value_loss.mean()

            loss = (
                actor_loss
                + (value_loss_coef * critic_loss)
                - (entropy_coef * entropy)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.actor.parameters()) + list(policy.critic.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                _, post_step_log_probs_flat, _ = policy.evaluate_actor(
                    obs_flat,
                    raw_actions_flat,
                )
                approx_kl = (
                    ((old_log_probs_flat - post_step_log_probs_flat) * mask_flat).sum()
                    / valid_count
                )

            metrics["loss"].append(float(loss.item()))
            metrics["actor_loss"].append(float(actor_loss.item()))
            metrics["critic_loss"].append(float(critic_loss.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["approx_kl"].append(float(approx_kl.item()))

            if target_kl is not None and abs(float(approx_kl.item())) > target_kl:
                stop_early = True
                break

        if stop_early:
            break

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in metrics.items()
    }
