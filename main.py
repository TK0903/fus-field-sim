# -*- coding: utf-8 -*-
"""
================================================================================
 実行エントリ  —  球面集束トランスデューサ 音場計算
================================================================================

 使い方:
   python main.py 2d          2D 計算のみ (軸上/横方向/2D断面マップ)
   python main.py 3d          3D 計算のみ (焦点体積の等圧面)
   python main.py both        両方 (既定)

 条件は config.py で編集する。一時的な上書きは CLI で:
   python main.py both --freq 3.0            周波数 3.0 MHz
   python main.py both --coating 50          コーティング 50 µm
   python main.py 3d --coating 100 --tag v2  結果フォルダ名に接尾辞を付ける
   python main.py both --cpu                 GPU を使わず CPU で計算

 結果は  results/<設定がわかる名前>/  に保存される。例:
   results/f1.500MHz_coat0um_T22.0C/
     ├─ config.txt        使用した設定一式(再現用)
     ├─ run.log           コンソール出力の全文
     ├─ axial.png / lateral.png / fieldmap.png / results_2d.csv
     └─ iso3d_*.png / results_3d.csv
================================================================================
"""

import argparse
import datetime
import sys
import traceback

import torch

import config as cfgmod
from fus.plotting import setup_japanese_font


class Tee:
    """標準出力をコンソールとログファイルの両方へ流す"""
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def select_device(force_cpu=False):
    """GPU を使う。CUDA が無ければ自動的に CPU にフォールバック。"""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def print_header(cfg, device):
    print("=" * 72)
    print(" 球面集束トランスデューサ 音場計算")
    print("=" * 72)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f" デバイス         : CUDA / {prop.name}")
        print(f" VRAM 容量        : {prop.total_memory/(1024**3):.2f} GB")
    else:
        print(" デバイス         : CPU"
              + ("" if torch.cuda.is_available() else " (CUDA 非対応環境)"))
    print(f" 実行日時         : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("-" * 72)
    for line in cfg.summary_lines():
        print(f" {line}")
    print(f" 出力先           : {cfg.out_dir}")
    if cfg.warnings:
        print("-" * 72)
        for w in cfg.warnings:
            print(f" [注意] {w}")
    print("=" * 72)


def write_config_txt(cfg, device):
    """再現用に、使った設定を results フォルダへテキストで残す"""
    p = cfg.out_dir / "config.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("# 球面集束トランスデューサ 音場計算 — 使用した設定\n")
        f.write(f"# 実行日時: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# デバイス: {device.type}"
                + (f" ({torch.cuda.get_device_name(0)})"
                   if device.type == "cuda" else "") + "\n\n")
        for line in cfg.summary_lines():
            f.write(line + "\n")
        f.write("\n# --- 数値設定 ---\n")
        f.write(f"SUBDIV_PER_LAMBDA       = {cfg.subdiv_per_lambda}\n")
        f.write(f"FIELD_SUBDIV_PER_LAMBDA = {cfg.field_subdiv_per_lambda}\n")
        f.write(f"RZ_SUBDIV               = {cfg.rz_subdiv}\n")
        f.write(f"FIELD_CHUNK             = {cfg.field_chunk}\n")
        f.write("\n# --- 計算範囲 [mm] ---\n")
        f.write(f"2D  z = {cfg.axial_z_min*1e3:.3f} .. {cfg.axial_z_max*1e3:.3f}, "
                f"x = ±{cfg.lateral_half*1e3:.3f}\n")
        f.write(f"3D  r = 0 .. {cfg.r_max_3d*1e3:.3f}, "
                f"z = {cfg.z_min_3d*1e3:.3f} .. {cfg.z_max_3d*1e3:.3f}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="球面集束トランスデューサの音場計算 (Rayleigh 積分, GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="both", choices=["2d", "3d", "both"],
                    help="実行する計算 (既定: both)")
    ap.add_argument("--freq", type=float, default=None,
                    help="駆動周波数 [MHz] (config.py の FREQ_MHZ を上書き)")
    ap.add_argument("--coating", type=float, default=None,
                    help="コーティング厚 [µm] (config.py の設定を上書き)")
    ap.add_argument("--temp", type=float, default=None,
                    help="水温 [℃] (config.py の設定を上書き)")
    ap.add_argument("--coating-mode", default=None,
                    choices=["conformal", "keep_aperture"],
                    help="コーティングの形状モデル")
    ap.add_argument("--tag", default=None,
                    help="結果フォルダ名に付ける接尾辞")
    ap.add_argument("--cpu", action="store_true",
                    help="GPU を使わず CPU で計算する")
    args = ap.parse_args(argv)

    setup_japanese_font()

    # 設定の組み立て(ここで範囲チェックも走る)。失敗したら計算に入らず終了する。
    try:
        cfg = cfgmod.make_config(freq_mhz=args.freq, coating_um=args.coating,
                                 water_temp_c=args.temp,
                                 coating_mode=args.coating_mode, tag=args.tag)
    except ValueError as e:
        print("設定エラー: " + str(e), file=sys.stderr)
        print("\nconfig.py または CLI 引数を確認してください。", file=sys.stderr)
        return 2
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.cpu)

    # コンソール出力を run.log にも残す
    log_path = cfg.out_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        orig = sys.stdout
        sys.stdout = Tee(orig, fh)
        try:
            print_header(cfg, device)
            write_config_txt(cfg, device)

            if args.mode in ("2d", "both"):
                from fus import sim2d
                print("\n" + "#" * 72)
                print("#  2D 計算")
                print("#" * 72)
                sim2d.run(cfg, device)

            if args.mode in ("3d", "both"):
                from fus import sim3d
                print("\n" + "#" * 72)
                print("#  3D 計算")
                print("#" * 72)
                sim3d.run(cfg, device)

            print("\n" + "=" * 72)
            print(f" 完了。結果は次のフォルダに保存されました:")
            print(f"   {cfg.out_dir}")
            print("=" * 72)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.stdout = orig
            print(f"\nエラーで停止しました。詳細は {log_path} を参照。",
                  file=sys.stderr)
            return 1
        finally:
            sys.stdout = orig
    return 0


if __name__ == "__main__":
    sys.exit(main())
