# -*- coding: utf-8 -*-
"""
図の共通設定 (日本語フォント・バックエンド)
"""

import matplotlib
matplotlib.use("Agg")            # 画面表示不要・PNG 保存のみ
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_japanese_font():
    """
    図中の日本語が豆腐(□)にならないよう、利用可能な日本語フォントを自動選択。
    Windows なら Yu Gothic / Meiryo / MS Gothic のいずれかが通常入っている。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
                 "IPAexGothic", "TakaoGothic"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False   # マイナス記号の文字化け防止
