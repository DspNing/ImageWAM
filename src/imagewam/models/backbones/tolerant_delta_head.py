"""邻域容差版变化头(TolerantDeltaHead)——训练期外挂,推理时整体不加载。

立项(2026-09-07)。病灶(v1 全量 eval,锚=Ning 08-21):Camera 87.05→82.68(−4.37)、
Light −2.90、Layout −2.36,BG +9.20。诊断:逐位监督 h_i→Δz_i 把主干 ref 特征往
绝对网格坐标上锁;camera 扰动整体平移画面时位置对应失效。机制上 eval 扰动是
整段 episode 一致偏移,Δz 场内部一致,损伤来源 = VAE/主干的平移等变残差 ×
逐位特化。处方:把监督的空间精确度从"逐格"放宽到"k×k 邻域"。

两种容差几何(2026-09-07 单元测试的玩具定量分析后定序,记录防回绕):

  pool(默认,主推): 目标侧池化
      target_i = N(i) 内合法格的 Δz 均值,  L = ‖Δẑ_i − target_i‖₁
      位移解耦**完全**:目标场平移任意 ≤(k−1)/2 格,pred=池化场仍零损失;
      质量守恒(blob 1 格 → 块内 1/|B| × |B| 格);主干只需块级定位。
  nbmean(对照,外部方案 uniform-A 版): 邻域均匀平均 L1
      L_i = (1/|N(i)|) Σ_{j∈N(i)} ‖Δẑ_i − Δz_j‖₁
      对**高对比场几乎不解耦**:逐格最优值是邻域目标的中位数(L1 几何),
      blob 中心邻域 8 零 vs 1 blob → 中位数 0 → 预测被拉向 0(中值侵蚀);
      玩具定量:目标 blob 偏移 1 格、预测原位,惩罚仅 −12%(0.0408→0.0363)。
      对平滑场它 ≈ 稳健平滑先验。保留作对照模式。

  min 匹配(外部方案原式)彻底否决,勿复活:pred≡0 时
  min_j‖0−Δz_j‖=0 几乎处处成立 → 全零预测是全局最优 → 辅助任务塌缩。
  learned A_ij 同理否决:训练分布没有 camera shift,恒等映射即最优,
  学出来只会安静退化回 v1。

共同实现约束:
  - 窗口边缘用合法邻居掩码均值/求和,绝不能零填充(7×7 网格 24/49 格子贴边;
    零填充 = 在边缘重新种上位置先验,恰是要除的病);
  - 中心项在窗内:k=1 两种模式都严格退回 v1 逐位匹配(单测有逐位等价断言);
  - loss_delta_raw 恒为严格逐位 L1(跨 run 可比;注意 pool 模式下 raw 地板会比
    v1 略高——目标含块内不可解析的细节,no-op 判读阈值 0.246 需相应放宽);
  - 核宽默认 k=3:7×7 上 k=5 已占半张图,会把预测在整个物体上抹平(blob 本身
    2~3 格宽)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # 包内导入(正式路径)
    from .delta_dynamics import DeltaDynamicsHead
except ImportError:  # 独立脚本/单元测试直接按文件加载
    from delta_dynamics import DeltaDynamicsHead  # type: ignore


class TolerantDeltaHead(DeltaDynamicsHead):
    """同 v1 头结构/锚点/零初始化,只把监督匹配从逐位放宽到邻域几何。

    tol_mode:
      "pool"   目标侧 k×k 合法窗均值池化(主推,位移完全解耦、质量守恒)
      "nbmean" 邻域均匀平均 L1(对照模式;对高对比场解耦弱,见模块 docstring)
    """

    def __init__(
        self,
        trunk_dim: int,
        grid_h: int = 7,
        grid_w: int = 14,
        tol_kernel: int = 3,
        tol_mode: str = "pool",
        **kwargs,
    ):
        super().__init__(trunk_dim=trunk_dim, grid_h=grid_h, grid_w=grid_w, **kwargs)
        self.main_h = int(grid_h)
        self.main_w = int(grid_w) // 2   # 主相机 = 网格左半区(行主序取列 < w/2)
        self.tol_kernel = max(1, int(tol_kernel))
        _mode = str(tol_mode).lower()
        if _mode not in ("pool", "nbmean"):
            raise ValueError(f"tol_mode must be pool|nbmean, got {tol_mode!r}")
        self.tol_mode = _mode
        g = self.tol_kernel // 2
        # 邻域偏移表(含中心)与每格合法邻居数(常数,预计算;非 persistent,
        # 不进 ckpt)。
        self._shifts = tuple(
            (dr, dc) for dr in range(-g, g + 1) for dc in range(-g, g + 1)
        )
        cnt = torch.zeros(1, self.main_h, self.main_w, 1)
        for dr, dc in self._shifts:
            r0, r1 = max(0, -dr), self.main_h - max(0, dr)
            c0, c1 = max(0, -dc), self.main_w - max(0, dc)
            cnt[:, r0:r1, c0:c1, :] += 1.0
        self.register_buffer("neighbor_count", cnt, persistent=False)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", tol_kernel={self.tol_kernel}, tol_mode={self.tol_mode}"

    def _window_sum(self, grid: torch.Tensor) -> torch.Tensor:
        """[B,h,w,C] 上对每个格累加其 k×k 合法窗内(含自身)的值,边缘截断。"""
        acc = torch.zeros_like(grid)
        for dr, dc in self._shifts:
            pr0, pr1 = max(0, -dr), self.main_h - max(0, dr)
            pc0, pc1 = max(0, -dc), self.main_w - max(0, dc)
            acc[:, pr0:pr1, pc0:pc1, :] = acc[:, pr0:pr1, pc0:pc1, :] + \
                grid[:, pr0 + dr:pr1 + dr, pc0 + dc:pc1 + dc, :]
        return acc

    def _block_mean(self, dz_main: torch.Tensor) -> torch.Tensor:
        """目标侧池化:每格目标 = k×k 合法窗(含中心,边缘截断)内 Δz 均值。

        dz_main 处于 no_grad(父 forward 已保证),池化目标同样不进图。
        """
        b, n, c = dz_main.shape
        grid = dz_main.view(b, self.main_h, self.main_w, c)
        with torch.no_grad():
            pooled = self._window_sum(grid) / self.neighbor_count.to(grid.dtype)
        return pooled.view(b, n, c)

    def _match_loss(self, pred, dz_main, strict):
        if self.tol_kernel == 1:
            return strict
        if self.tol_mode == "pool":
            return F.l1_loss(pred.float(), self._block_mean(dz_main))
        # nbmean(对照):逐格邻域均匀平均 L1;对 pred_i 的梯度把其拉向邻域目标
        # 中位数(高对比场的中值侵蚀见模块 docstring)。
        b, n, c = pred.shape
        pred_g = pred.float().view(b, self.main_h, self.main_w, c)
        dz_g = dz_main.view(b, self.main_h, self.main_w, c)
        acc = torch.zeros_like(pred_g)
        for dr, dc in self._shifts:
            pr0, pr1 = max(0, -dr), self.main_h - max(0, dr)
            pc0, pc1 = max(0, -dc), self.main_w - max(0, dc)
            acc[:, pr0:pr1, pc0:pc1, :] = acc[:, pr0:pr1, pc0:pc1, :] + (
                pred_g[:, pr0:pr1, pc0:pc1, :]
                - dz_g[:, pr0 + dr:pr1 + dr, pc0 + dc:pc1 + dc, :]
            ).abs()
        tol_map = acc / self.neighbor_count.to(acc.dtype)
        return tol_map.mean()
