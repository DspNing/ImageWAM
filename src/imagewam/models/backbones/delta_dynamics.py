"""变化量预测分支(DeltaDynamicsHead)——训练期外挂,推理时整体不加载。

设计(2026-08-23 定稿,生死门 GO:perturb_probe/results/delta_gate/report.txt):

    video expert 最后一层参考帧 token 特征(主相机 49 个,不 detach)
        + 16 步真实动作嵌入(逐步投影 + 步序位置嵌入)
        + 当前本体感觉 token
        → 4 层小 transformer → 预测主相机区潜空间变化量
          dz = z(t+16) - z(t)   (batch 里现成:target_latent - ref_latent)

机制:固定相机下静态内容(背景)在相减中代数消掉 → 该目标天然无背景,
预测它必须读"什么在动/物体在哪" → 梯度经参考帧特征回流塑形主干,
直接攻击"主干把 2.2x 放大成 5.0x"的病灶(A6/A3 探针结论)。

安全设计:
  - 目标 detach(梯度只经预测头流向主干特征,不污染视频目标);
  - 输出层零初始化(初始预测 dz≡0,损失从零基线起步,不给主干注入噪声梯度);
  - loss_lambda * warmup(前 warmup_steps 步线性升到 loss_lambda);
  - 推理时整个头不存在:eval 栈零改动;
  - enabled=false 时 mot 不取锚点、不建头、不加损失 → 前向与现状逐位一致。

腕部相机:只作输入可选项、不作目标(腕部 dz 被相机自运动主导 = 动作捷径;
生死门 G3 进一步显示腕部 token 对主相机运动零线性增益,故输入也只用主相机)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaDynamicsHead(nn.Module):
    """从主干参考帧特征预测主相机区 latent 变化量的小 transformer 头。"""

    def __init__(
        self,
        trunk_dim: int,
        action_dim: int = 7,
        proprio_dim: int = 8,
        latent_channels: int = 128,
        action_horizon: int = 16,
        grid_h: int = 7,
        grid_w: int = 14,
        hidden: int = 1024,
        depth: int = 4,
        num_heads: int = 8,
        loss_lambda: float = 0.1,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.trunk_dim = int(trunk_dim)
        self.action_horizon = int(action_horizon)
        self.loss_lambda = float(loss_lambda)
        self.warmup_steps = max(1, int(warmup_steps))
        # 主相机 token 索引:7x14 网格行主序,左半区(w < grid_w//2)= 第三人称相机。
        idx = torch.arange(grid_h * grid_w)
        self.register_buffer("main_idx", (idx % grid_w) < (grid_w // 2), persistent=False)

        self.feat_proj = nn.Linear(self.trunk_dim, hidden)
        self.action_proj = nn.Linear(action_dim, hidden)
        self.action_pos = nn.Parameter(torch.zeros(self.action_horizon, hidden))
        self.proprio_proj = nn.Linear(proprio_dim, hidden) if proprio_dim else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=4 * hidden,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, latent_channels)
        nn.init.zeros_(self.out_proj.weight)   # 零初始化:初始预测 dz≡0
        nn.init.zeros_(self.out_proj.bias)
        # persistent=True(审计 F11-3):断点续训时 warmup 进度随 ckpt 保存/恢复,
        # 不会重跑一遍 warmup。
        self.register_buffer("num_calls", torch.zeros((), dtype=torch.long))

    def extra_repr(self) -> str:
        return (f"trunk_dim={self.trunk_dim}, action_horizon={self.action_horizon}, "
                f"lambda={self.loss_lambda}, warmup={self.warmup_steps}")

    def forward(
        self,
        ref_hidden: torch.Tensor,      # [B, 98, trunk_dim] 最终层参考帧特征(不 detach)
        action: torch.Tensor,          # [B, 16, action_dim] GT 动作(已归一化)
        proprio: torch.Tensor | None,  # [B, proprio_dim] 当前本体感觉(可 None)
        ref_latent: torch.Tensor,      # [B, C, grid_h, grid_w] 参考帧 latent
        target_latent: torch.Tensor,   # [B, C, grid_h, grid_w] t+16 端点帧 latent
    ):
        """返回 (加权损失, 原始 L1 detach):加权 = lambda * warmup * 原始。"""
        if action.shape[1] != self.action_horizon:
            raise ValueError(
                f"DeltaDynamicsHead expects {self.action_horizon} action steps, "
                f"got {action.shape[1]}")
        with torch.no_grad():
            dz = (target_latent.float() - ref_latent.float())
            dz = dz.flatten(2).transpose(1, 2)          # [B, 98, C] 行主序
            dz_main = dz[:, self.main_idx]              # [B, 49, C]

        _dev = self.feat_proj.weight.device
        vis = self.feat_proj(ref_hidden[:, self.main_idx].to(_dev, dtype=self.feat_proj.weight.dtype))
        act = self.action_proj(action.to(_dev, dtype=self.action_proj.weight.dtype)) + self.action_pos
        toks = [vis, act]
        if self.proprio_proj is not None and proprio is not None:
            p = proprio.to(_dev, dtype=self.proprio_proj.weight.dtype)
            if p.dim() == 3:
                p = p[:, 0]
            toks.append(self.proprio_proj(p).unsqueeze(1))
        x = torch.cat(toks, dim=1)
        x = self.blocks(x)
        n_main = int(self.main_idx.sum())
        # 只在主相机视觉位读出(前 n_main 个 token);动作/本体 token 只作注意力上下文。
        pred = self.out_proj(self.out_norm(x[:, :n_main]))   # [B, 49, C]

        strict = F.l1_loss(pred.float(), dz_main)
        loss = self._match_loss(pred, dz_main, strict)
        warm = min(1.0, float(self.num_calls.item() + 1) / self.warmup_steps)
        self.num_calls += 1
        return self.loss_lambda * warm * loss, strict.detach()

    def _match_loss(
        self,
        pred: torch.Tensor,     # [B, 49, C] 预测(带梯度)
        dz_main: torch.Tensor,  # [B, 49, C] 目标(no_grad, float)
        strict: torch.Tensor,   # 严格逐位 L1
    ) -> torch.Tensor:
        """监督匹配项,默认=严格逐位 L1(v1 原语义)。

        strict 同时是日志口径(loss_delta_raw 恒为严格逐位 L1,跨 run 可比);
        子类(如邻域容差版)只重写本方法来更换"进 total 的加权项"的匹配几何,
        其余(目标构造/头结构/锚点/warmup)全部继承。
        """
        return strict
