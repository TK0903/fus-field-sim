# -*- coding: utf-8 -*-
"""
Rayleigh-Sommerfeld 積分の GPU 計算

--------------------------------------------------------------------------------
 計算式と時間規約
--------------------------------------------------------------------------------
  時間規約: exp(+i*omega*t)。よって外向き球面波は空間的に exp(-i*k*r')。
  剛体バッフル面上の Rayleigh-Sommerfeld 積分を球冠表面 S で数値積分する:

      p(r) = (i*rho*c*k*u0 / (2*pi)) * ∫∫_S  exp(-i*k*r') / r'  dS

  u0 : 表面法線振動速度(一様・peak)、 r' = |場の点 - 音源点|

 【近似について】
  Rayleigh 積分は本来「平面剛体バッフル」用である。曲面音源への適用は
  標準的な近似であり、obliquity factor(斜め係数)は省略している。
  F# = 0.77 の強集束だが、積分自体は近軸近似を含まない厳密形である。
--------------------------------------------------------------------------------
"""

import math
import torch


def make_prefactor(cfg, u0=1.0, device=None, dtype=torch.complex128):
    """前置係数 i*rho*c*k*u0/(2*pi) を複素テンソルで返す"""
    val = 1j * cfg.rho * cfg.c * cfg.k * u0 / (2.0 * math.pi)
    return torch.tensor(val, dtype=dtype, device=device)


def compute_pressure(field_pts, src_pts, src_dS, cfg, prefactor,
                     chunk=None, device=None, dtype=torch.complex128):
    """
    p(r) = prefactor * Σ_s [ exp(-alpha*r') * exp(-i*k*r') / r' ] * dS_s

    実装方針:
      ・Python の for ループでペアを回さない。全ペアをテンソル化 (cdist)。
      ・場の点をチャンクに切って逐次計算し、音源方向に総和する。
        → 高周波化で音源点数が増えても OOM を避けられる構造。
      ・OOM 時はチャンクを自動半減してリトライ。
      ・r'→0 の特異点処理: 場の点は z>=3mm、音源は z<=h なので実際には
        起きないが、安全のため r' を微小値でクランプする。

    Returns: p (Nf,) complex tensor
    """
    chunk = cfg.field_chunk if chunk is None else chunk
    k = cfg.k
    alpha = cfg.alpha_np_per_m
    Nf = field_pts.shape[0]
    out = torch.empty(Nf, dtype=dtype, device=device)
    r_floor = 1e-9      # 特異点クランプ [m]

    i0 = 0
    cur = int(chunk)
    while i0 < Nf:
        i1 = min(i0 + cur, Nf)
        try:
            d = torch.cdist(field_pts[i0:i1], src_pts)   # (C,Ns) 距離 r'
            d = torch.clamp(d, min=r_floor)
            amp = 1.0 / d
            if alpha > 0.0:
                amp = amp * torch.exp(-alpha * d)
            green = torch.polar(amp, -k * d)             # exp(-i k r') / r'
            out[i0:i1] = prefactor * (green * src_dS).sum(dim=1)
            i0 = i1
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur > 1:
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
                cur = max(1, cur // 2)
                print(f"  [OOM] チャンクを半減して再試行: chunk={cur}")
            else:
                raise
    return out


def gain_of(p, cfg):
    """複素音圧テンソル → 利得 |p|/(rho*c*u0) の numpy 配列"""
    return (torch.abs(p).detach().cpu().numpy() / cfg.p_surface)
