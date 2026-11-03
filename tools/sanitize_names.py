#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公開用の py/ から配信者の名前を消す（公開版に名前を出さない。mori1 2026-11-03「アメリー型って出てるのまずい」）。
dq10-auto から py/ に写した後に必ず実行する: python tools/sanitize_names.py
置き換えは内部の表記 → 公開用の呼び方。コメント・docstring・表示文字列のどれでも置き換える。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES = [
    (r"アメリーさん型", "定石"), (r"アメリー型", "定石"), (r"アメリーさん", "配信者A"), (r"アメリー", "配信者A"),
    (r"ソルティーさん", "配信者B"), (r"ソルティー", "配信者B"),
    (r"ヘリオッドさん", "配信者C"), (r"ヘリオッド", "配信者C"),
    (r"GJ\s*channel", "配信者D"), (r"GJ\s*さん", "配信者D"), (r"\bGJ\b", "配信者D"),
    (r"ポロル", "配信者E"), (r"ウェロア", "配信者F"), (r"フローライト氏", "配信者G"), (r"フローライト", "配信者G"),
    (r"あでゅぴ", "配信者H"), (r"ひつひつ羊", "配信者I"), (r"ノート両氏", "解説者"), (r"ヘリオッド・ノート両氏", "解説者"),
    (r"mori1\s*さん", "作者"), (r"mori1", "作者"), (r"廣實さん", "作者"),
]
TARGET_WORDS = ["アメリー", "ソルティー", "ヘリオッド", "GJ", "ポロル", "ウェロア", "フローライト", "mori1", "廣實", "あでゅぴ", "ひつひつ"]


def sanitize(text):
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    return text


def main():
    changed = 0; left = []
    for p in list((ROOT / "py").glob("*.py")) + [ROOT / "app.js", ROOT / "index.html", ROOT / "worker.js"]:
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t2 = sanitize(t)
        if t2 != t:
            p.write_text(t2, encoding="utf-8"); changed += 1
        for w in TARGET_WORDS:
            if w in t2:
                left.append(f"{p.name}: {w}")
    print(f"書き換え {changed} ファイル")
    if left:
        print("残っている:", left); sys.exit(1)
    print("配信者名なし")


if __name__ == "__main__":
    main()
