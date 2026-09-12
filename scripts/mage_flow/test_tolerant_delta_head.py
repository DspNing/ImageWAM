"""TolerantDeltaHead 单元测试(2026-09-07)。

跑:mageflow python scripts/mage_flow/test_tolerant_delta_head.py
覆盖:k=1 逐位等价 v1(两种模式)/ 池化目标暴力参考对照 / 角格分母(边缘不补零)/
位移解耦性(pool 完全解耦 vs nbmean 弱解耦的定量对照)/ 零锚无损 /
梯度流通(含零初始化 out_proj 的注意点)/ 返回契约(λ·warm、raw 恒严格)/ bf16。
"""
import os
import sys

import torch
import torch.nn.functional as F

_BB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "imagewam", "models", "backbones")
sys.path.insert(0, _BB)

import delta_dynamics as dd_mod  # noqa: E402
import tolerant_delta_head as td_mod  # noqa: E402

DeltaDynamicsHead = dd_mod.DeltaDynamicsHead
TolerantDeltaHead = td_mod.TolerantDeltaHead

H = W = 7
PASS = []


def check(name, cond, detail=""):
    PASS.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def make_head(mode, k=3, **kw):
    return TolerantDeltaHead(trunk_dim=32, latent_channels=16, hidden=64, depth=2,
                             tol_kernel=k, tol_mode=mode, **kw)


def brute_pool_target(dz, k):
    """独立参考实现:双重循环 + 显式边界判断(与被测实现不同构)。"""
    b, n, c = dz.shape
    g = k // 2
    dg = dz.view(b, H, W, c)
    out = torch.zeros_like(dg)
    for r in range(H):
        for col in range(W):
            vals = []
            for dr in range(-g, g + 1):
                for dc in range(-g, g + 1):
                    rr, cc = r + dr, col + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        vals.append(dg[:, rr, cc])
            out[:, r, col] = torch.stack(vals).mean(0)
    return out.view(b, n, c)


def make_inputs(b=3, c=16, seed=0):
    torch.manual_seed(seed)
    ref_hidden = torch.randn(b, 98, 32)
    action = torch.randn(b, 16, 7)
    ref_latent = torch.randn(b, c, H, 14)
    target_latent = torch.randn(b, c, H, 14)
    return ref_hidden, action, ref_latent, target_latent


def main():
    torch.manual_seed(0)
    v1 = DeltaDynamicsHead(trunk_dim=32, latent_channels=16, hidden=64, depth=2)
    pool1 = make_head("pool", k=1)
    pool1.load_state_dict(v1.state_dict())
    nb1 = make_head("nbmean", k=1)
    nb1.load_state_dict(v1.state_dict())
    check("main_idx 一致", torch.equal(v1.main_idx, pool1.main_idx))
    rh, ra, rl, tl = make_inputs()
    w0, r0 = v1(rh, ra, None, rl, tl)
    wp, rp = pool1(rh, ra, None, rl, tl)
    wn, rn = nb1(rh, ra, None, rl, tl)
    check("k=1 逐位等价(pool)", torch.allclose(w0, wp, atol=1e-7)
          and torch.allclose(r0, rp, atol=1e-7))
    check("k=1 逐位等价(nbmean)", torch.allclose(w0, wn, atol=1e-7)
          and torch.allclose(r0, rn, atol=1e-7))

    pool3 = make_head("pool", k=3)
    pool3.load_state_dict(v1.state_dict())
    nb3 = make_head("nbmean", k=3)
    nb3.load_state_dict(v1.state_dict())

    # ---- 池化目标对照独立暴力实现 ----
    dz = torch.randn(2, 49, 16)
    want_t = brute_pool_target(dz, 3)
    got_t = pool3._block_mean(dz)
    check("pool 目标对照暴力实现", torch.allclose(got_t, want_t, atol=1e-5),
          f"maxdiff={(got_t - want_t).abs().max().item():.2e}")

    # ---- 角尖峰手算值:角4/边6/心9 合法窗均值(pred=0 时两模式同值)----
    dz_spike = torch.zeros(1, 49, 16)
    dz_spike[:, 0] = 1.0
    pred_zero = torch.zeros(1, 49, 16)
    strict0 = F.l1_loss(pred_zero, dz_spike)
    want = (1 / 4 + 1 / 6 + 1 / 6 + 1 / 9) / 49
    gp = pool3._match_loss(pred_zero, dz_spike, strict0).item()
    gn = nb3._match_loss(pred_zero, dz_spike, strict0).item()
    check("角尖峰手算值(pool)", abs(gp - want) < 1e-6, f"{gp:.6f} vs {want:.6f}")
    check("角尖峰手算值(nbmean)", abs(gn - want) < 1e-6, f"{gn:.6f} vs {want:.6f}")

    # ---- 位移解耦性:目标 blob 偏移 1 格 ----
    dz_shift = torch.zeros(1, 49, 16)
    dz_shift[:, 4 * 7 + 4] = 1.0                     # blob 在 (4,4)
    pred_star = pool3._block_mean(dz_shift)          # 池化场 = pool 模式的可达解
    s_strict = F.l1_loss(pred_star, dz_shift)
    l_pool = pool3._match_loss(pred_star, dz_shift, s_strict).item()
    l_nb = nb3._match_loss(pred_star, dz_shift, s_strict).item()
    check("pool 位移完全解耦(pred=池化场 → 损失≈0)", l_pool < 1e-6,
          f"pool={l_pool:.2e} vs strict={s_strict.item():.5f}")
    print(f"       [对照] nbmean 同一预测下损失={l_nb:.5f}"
          f"(仅比 strict {s_strict.item():.5f} 降 {(1 - l_nb / s_strict.item()) * 100:.0f}%"
          f" → 高对比场解耦弱,故 pool 为默认)")

    # ---- 质量守恒:池化不丢 blob 总质量(内部参考,信息性)----
    mass_ratio = (pred_star.sum() / dz_shift.sum()).item()
    print(f"       [info] 池化场/blob 质量比 = {mass_ratio:.3f}(理论 ≈1,边缘格略偏)")

    # ---- 零锚无损:pred=0, dz=0 → 损失 0 且梯度 0(静态区无压力)----
    dz0 = torch.zeros(1, 49, 16)
    p0 = pred_zero.clone().requires_grad_(True)
    l0 = pool3._match_loss(p0, dz0, F.l1_loss(p0, dz0))
    l0.backward()
    check("零目标损失为 0(pool)", l0.item() < 1e-8)
    check("零目标梯度为 0(pool)", p0.grad.abs().max().item() < 1e-8)

    # ---- 梯度流通:零初始化 out_proj 时首批梯度本应为零(v1 安全设计);
    #      给 out_proj 加扰动模拟训练态后,weighted 反传须到达 ref_hidden ----
    rh_g = rh.clone().requires_grad_(True)
    with torch.no_grad():
        pool3.out_proj.weight.add_(torch.randn_like(pool3.out_proj.weight) * 0.02)
        pool3.out_proj.bias.add_(torch.randn_like(pool3.out_proj.bias) * 0.02)
    w3, _ = pool3(rh_g, ra, None, rl, tl)
    w3.backward()
    check("梯度到达 ref_hidden(训练态)", rh_g.grad is not None
          and rh_g.grad.abs().max().item() > 0)

    # ---- 返回契约:k=1 时 weighted = λ·warm·raw(warm=1);num_calls 递增 ----
    fresh = make_head("pool", k=1, loss_lambda=0.25, warmup_steps=1)
    wf, rf = fresh(rh, ra, None, rl, tl)
    check("返回契约 weighted=λ·warm·raw(k=1)", torch.allclose(wf, 0.25 * rf, atol=1e-7)
          and fresh.num_calls.item() == 1,
          f"num_calls={int(fresh.num_calls.item())}")

    # ---- raw 恒为严格逐位 L1(pool 模式也不改日志口径)----
    # 注意 trunk 是 nn.TransformerEncoderLayer(dropout=0.1 默认),train 模式下
    # 两次前向 dropout 掩码不同,必须置 eval 消随机性后再对照(v1 同款行为)。
    pool3.eval()
    with torch.no_grad():
        dz_true = (tl.float() - rl.float()).flatten(2).transpose(1, 2)
        dz_true = dz_true[:, pool3.main_idx]
        vis = pool3.feat_proj(rh[:, pool3.main_idx])
        act = pool3.action_proj(ra) + pool3.action_pos
        pred_re = pool3.out_proj(
            pool3.out_norm(pool3.blocks(torch.cat([vis, act], dim=1))[:, :49]))
        strict_re = F.l1_loss(pred_re.float(), dz_true)
    _, r4 = pool3(rh, ra, None, rl, tl)
    check("raw 恒为严格逐位 L1", torch.allclose(r4, strict_re, atol=1e-5),
          f"{r4.item():.6f} vs {strict_re.item():.6f}")
    pool3.train()

    # ---- bf16 数值路径(mot .to(dtype) 后的真实形态)----
    tolb = make_head("pool", k=3).to(torch.bfloat16)
    wb, rb = tolb(rh.bfloat16(), ra.bfloat16(), None, rl.bfloat16(), tl.bfloat16())
    check("bf16 前向有限", torch.isfinite(wb).all() and torch.isfinite(rb).all())

    # ---- 非法 mode 拒绝 ----
    try:
        make_head("softmin")
        check("非法 tol_mode 拒绝", False)
    except ValueError:
        check("非法 tol_mode 拒绝", True)

    n_fail = sum(1 for _, ok, _ in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
