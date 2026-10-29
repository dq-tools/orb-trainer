#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鍛冶ミニゲームシミュレータ v1（武器・防具・道具共通モデル）

モデルの根拠:
- ダメージ: kaji_data.py の温度別7候補から等確率（ほしづくよ氏の計算式と同等）
- 会心: 2倍威力。理想値が射程内なら理想値ぴったりで停止（本会心）。射程外なら2倍加算（偽会心）
- 理想値 = ゾーン中央（ゆんまる動画の本会心停止値 B=248/D=185/E=185 がゾーン中央と一致）
- 倍半特性: 温度が400の倍数で威力2倍 / 400n+200 で半減（実測: 2000/1600/1200/800が倍ターン）
- 温度変化: 打つ特技は-50、熱風おろしは計-150、冷やし込み-300、火力上げ+300（動画実測）
- できのよさ: 誤差絶対値合計。ゾーン内は誤差4に丸め。ゾーン外は誤差そのまま+ペナルティ3

未確定パラメータ（assumptionsに明記、実測で更新する）:
- 会心基礎率 / ねらい系の倍率 / 大成功閾値 / 弱ねらい・左右打ちの仕様
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import kaji_data

# ---------------------------------------------------------------- 設定

USE_FAKE_INFO = False   # 偽会心（会心したのに止まらなかった）で理想値候補を絞り、推定理想値・先読みの信念・厳密解に使う（2026-10-12 実験）
K_OFFSETS = (-2, -1, 0, 1, 2)

MODORI_TRAITS = ("再生", "戻り")   # 鍛冶の地金「再生」（2026-10-20 mori1: 呼び方は「再生」に統一。旧表記「戻り」も受ける）

ASSUMPTIONS = {
    # 会心基礎率 = 光ハンマー★3(4.6%) + スキル会心アップ計0.6%(Lv30で打ち止め・大辞典確認済) + コツ(1.0%)
    # = 6.2%。フローライト氏統計モデルと整合
    "crit_base": 0.062,
    "aim_crit_mult": 7.0,     # ねらい系: 道具会心率+600% → 7倍
    "great_success_max_err": 7,  # 6マス超素材の大成功閾値（GRADE_TABLEで確定済み）
    "focus_max": 253,         # 実測: mori1さんのゲーム画面表示（消費集中力 5/253）2026-09-03
    # 必殺チャージ: 公式ガイド「加えたチカラ(合計ダメージ)が大きいと直後に起きやすい」
    # 途中チャージ: 「1打ごとに判定」モデル（DQ10フロンティア 2.1後期の実測: Lv45で5506ターン中45回=0.82%/ターン、
    #   超4連が圧倒的に出やすい=打数分の判定。mori1さん指定 2026-09-04）。1打あたり0.6%（単体0.6% / ペア1.2% / 4連2.4%）
    #   ※旧モデル(合計ダメージ比例 hissatsu_k)は廃止
    "hissatsu_per_hit": 0.006,
    "hissatsu_focus": 0,
    "quad_min_cells": 3,   # 2x2 の面として選べる最小のマス数（穴を含む面。グリーブは 3。4 にすると穴のある面は使わない＝旧仕様）      # 必殺（ヘパイトスの炎）押下で回復する集中。旧仕様 +30（helpfordqx blacksmith.js）は本物の棋譜で否定: GJさん 15 局（2026-10-20）、mori1さんも「必殺で回復はない」と確認
                              #   1100℃で必殺→集中 112 のまま（+30 なし）。kifu の check（消費の連鎖）も矛盾 0。要 mori1 確認
    "open_hissatsu": 0.10,    # 光の鍛冶ハンマー特殊効果: 開幕必殺チャージ率（約10%・できのよさ依存）
    # ねらい系の倍率: 鍛冶の実測はネアンデルタール氏124打で8.5倍示唆、ひろぐ氏は7倍仮定、
    # 裁縫は8倍説（フローライト氏は裁縫・木工中心で鍛冶固有版なし）→ 中間の8倍を採用 ⚠幅あり
    # 2026-09-04 mori1さん指定の参考: dq10-blog.blogspot.com/2017/03/blog-post.html（防具鍛冶・光ハンマー★3）
    #   ねらい打ち = 道具会心率+600%（7倍: 6.2%→43.4%）/ 会心ターン = +400%（5倍: 31%）/ 両方 = 11倍（68.2%）
    #   ※以前の8倍(50%)は高めだった（mori1さん体感）
    "aim_bonus": 6.0,         # ねらい系: 倍率に +6（→7倍）
    "event_bonus_mult": 4.0,  # 光マス/会心アップターン: 倍率に +4（→単独5倍、ねらいと併用で11倍）
    "modori_rate": 1.0,       # 再生（旧称 戻り）: 打った後に 200℃の倍数になった時、範囲外のマスがあれば毎回起きる（GJさん棋譜 3 局で確認）
    "modori_amount": (12, 16),  # 戻る量（GJさん配信 P9yRMJLHreA の緑の数字 12〜16）
    # まれに集中力回復（職人 Lv70 の特技。行動の後に集中が 10 以下だと低確率で 30 回復。画面は白く光って「+30」「なんと 集中力が 回復した！」。
    # 2026-10-29 mori1 本人の録画 よろい上 8 局目で判明。本物の棋譜の鍛冶 160 機会中 9 回 ≈ 5.6%（⚠仮。防具 3/34・道具 6/110・武器 0/16））
    "focus_recover_p": 0.056,
    "focus_recover_below": 10,
    "focus_recover_amount": 30,
    "light_rate": 1.0,        # 光特性: 200の倍数到達時にどこかが光る確率（mori1さん確認: 100%。2026-09-04）
    # 理想値（会心で止まる値）のばらつき。"zone"=基準値の範囲全体から一様（mori1さん 2026-10-17:
    # 「どこが基準値になるかはランダム」。実配信で超あまつゆA 180-190 が下端180で会心停止）。
    # 整数 n = 中央±n（旧仕様は 2。ゆんまる動画の B=248/D=185/E=185 が中央だったことからの仮定）
    "ideal_spread": "zone",
}

# できのよさ判定表（マス数 → (大成功上限, ★2上限, ★1上限, ★0上限)。それ超えは失敗）
# 出典: https://cidmogero.com/2021/02/13/shokunin-daiseiko/
#（3マス行はみいみ氏の実測 ★3:0-2/★2:3-8/★1:9-13/★0:14-23 と一致確認済み）
GRADE_TABLE = {
    2: (0, 4, 9, 17),
    3: (2, 8, 13, 23),
    4: (3, 8, 13, 23),
    5: (5, 13, 21, 33),
    6: (7, 14, 25, 39),
    7: (9, 16, 29, 44),
    8: (11, 18, 33, 49),  # ⚠外挿（実測なし。2..7の増分パターンから推定）
}


def sample_ideal(lo: int, hi: int, rng: random.Random, center: int | None = None) -> int:
    """理想値を1つ引く（ASSUMPTIONS["ideal_spread"] に従う）。center は旧仕様（中央±n）の中心。"""
    sp = ASSUMPTIONS["ideal_spread"]
    if sp == "zone":
        return rng.randint(lo, hi)
    c = (lo + hi) // 2 if center is None else center
    return max(lo, min(hi, c + rng.randint(-int(sp), int(sp))))


def ideal_offsets(lo: int, hi: int) -> tuple[int, ...]:
    """理想値の候補（中央からのずれ）。終盤ソルバーの信念に使う。"""
    sp = ASSUMPTIONS["ideal_spread"]
    c = (lo + hi) // 2
    if sp == "zone":
        return tuple(range(lo - c, hi - c + 1))
    return tuple(sorted({max(lo, min(hi, c + k)) - c for k in range(-int(sp), int(sp) + 1)}))


def grade(n_cells: int, total_err: int) -> str:
    """誤差合計からできのよさを返す。"""
    g, s2, s1, s0 = GRADE_TABLE[n_cells]
    if total_err <= g:
        return "大成功"
    if total_err <= s2:
        return "★2"
    if total_err <= s1:
        return "★1"
    if total_err <= s0:
        return "★0"
    return "失敗"

# 特技定義: (倍率キー, 消費, 対象タイプ, 会心率倍率, 温度変化)
#   対象タイプ: cell=1マス指定 / pair_v=縦2 / pair_lr=横2 / pair_diag=ななめ2 /
#              quad=2×2 / none=対象なし
SKILLS = {
    "たたく":        dict(mult="x1.0", cost=5,  targets="cell", crit=1.0, dtemp=-50),
    "てかげん打ち":   dict(mult="x0.5", cost=10, targets="cell", crit=1.0, dtemp=-50),
    "2倍打ち":       dict(mult="x2.0", cost=8,  targets="cell", crit=1.0, dtemp=-50),
    "3倍打ち":       dict(mult="x3.0", cost=11, targets="cell", crit=1.0, dtemp=-50),
    "熱風おろし":     dict(mult="x2.5", cost=6,  targets="cell", crit=1.0, dtemp=-150),
    "上下打ち":      dict(mult="x1.2", cost=8,  targets="pair_v", crit=1.0, dtemp=-50),
    "左右打ち":      dict(mult="x1.2", cost=8,  targets="pair_lr", crit=1.0, dtemp=-50),  # 消費8はmori1さん確認済み(2026-09-06)
    "ななめ打ち":     dict(mult="x1.2", cost=7,  targets="pair_diag", crit=1.0, dtemp=-50),
    "4連打ち":       dict(mult="x1.2", cost=12, targets="quad", crit=1.0, dtemp=-50),
    "超4連打ち":      dict(mult="x2.0", cost=18, targets="quad", crit=1.0, dtemp=-50),
    "ねらい打ち":     dict(mult="x1.0", cost=16, targets="cell", crit=7.0, dtemp=-50),
    "弱ねらい打ち":   dict(mult="x0.5", cost=20, targets="cell", crit=7.0, dtemp=-50),  # 消費20はmori1さん確認済み(2026-09-06)
    "上下ねらい打ち": dict(mult="x1.2", cost=25, targets="pair_v", crit=7.0, dtemp=-50),
    "みだれ打ち":     dict(mult="x0.8", cost=7,  targets="rand4", crit=1.0, dtemp=-50),  # ランダムに4回（同じマスに重なることもある＝mori1さん確認）。消費7・0.8倍は GJさんの棋譜で確認（2026-10-20）
    "火力上げ":      dict(mult=None,   cost=10, targets="none", crit=0.0, dtemp=+300),
    "冷やし込み":     dict(mult=None,   cost=12, targets="none", crit=0.0, dtemp=-300),
    "必殺":          dict(mult=None,   cost=0,  targets="none", crit=0.0, dtemp=-50)  # ヘパイトス使用ターンは温度-50（DQ10フロンティア記事）,
}


# 温度→ダメージ表を (temp, mult) で直引きできるよう前展開（高速化。挙動は同一）
_DC = {}
for _t in range(0, 2101, 50):
    _tt = max(50, min(2000, _t // 50 * 50))
    for _mk, _v in kaji_data.DAMAGE_TABLE[_tt].items():
        _DC[(_t, _mk)] = _v


def dmg_candidates(temp: int, mult_key: str) -> list[int]:
    try:
        return _DC[(temp, mult_key)]
    except KeyError:
        t = max(50, min(2000, temp // 50 * 50))
        return kaji_data.DAMAGE_TABLE[t][mult_key]


_COST_CACHE = {}


def cost_of(skill: str, trait: str, temp: int, first: bool = False) -> int:
    """first=True は局の最初の行動（開始の 1000℃）。集中力変化でも最初の手は消費が変わらない
    （超あまたね杯 樹液 5 局: 1000℃の火力上げが 253→243、画面の弱ねらい打ちも 20。2026-10-20）"""
    if first and trait == "集中力変化":
        return SKILLS[skill]["cost"]
    key = (skill, trait, temp)
    c = _COST_CACHE.get(key)
    if c is None:
        c = _cost_of(skill, trait, temp)
        _COST_CACHE[key] = c
    return c


def _cost_of(skill: str, trait: str, temp: int) -> int:
    """特技の消費集中力（集中力変化: 400の倍数で半減、400n+200 の会心率アップのターンは 1.5 倍・切り上げ）。
    1.5 倍は大会配信で実測（熱風 6→9 / ねらい打ち 16→24 / 上下ねらい打ち 25→38 / 超4連 18→27 / たたく 5→8、mori1さん確認 2026-10-18）"""
    cost = SKILLS[skill]["cost"]
    if trait == "集中力変化" and cost > 0:
        if temp % 400 == 0:
            return math.ceil(cost / 2)
        if temp % 400 == 200:
            return math.ceil(cost * 1.5)
    return cost


def is_crit_turn(trait: str, temp: int) -> bool:
    """会心率アップターン（集中力変化: 400n+200 = 200/600/1000...）。"""
    return trait == "集中力変化" and temp % 400 == 200


_TM_CACHE = {}


def turn_multiplier(trait: str, temp: int) -> float:
    key = (trait, temp)
    v = _TM_CACHE.get(key)
    if v is None:
        v = _turn_multiplier(trait, temp)
        _TM_CACHE[key] = v
    return v


def _turn_multiplier(trait: str, temp: int) -> float:
    """素材特性による威力倍率（倍半のみ実装。他特性はv2で）。

    倍半のスケジュールは全アイテム固定（mori1さん確認済み）:
    200=半減, 400=倍, 600=半減, 800=倍, 1000=半減, 1200=倍,
    1400=半減, 1600=倍, 1800=半減, 2000=倍
    """
    if trait == "倍半":
        if temp % 400 == 0:
            return 2.0
        if temp % 200 == 0:
            return 0.5
    return 1.0


# ---------------------------------------------------------------- 状態

@dataclass
class Cell:
    lo: int
    hi: int
    v: int = 0
    ideal_true: int = 0  # 隠し理想値（会心停止点）。クラフトごとに変動する（範囲は ASSUMPTIONS['ideal_spread']）
    pinned: bool = False  # 本会心で停止済み＝理想値が判明している（画面で観測できる情報）
    kmask: int = 0b11111  # 理想値の候補（中央-2..+2 のビット）。偽会心で「射程内に理想は無かった」と分かると絞る（USE_FAKE_INFO）

    ideal: int = 0  # ポリシーが狙う推定理想値 = ゾーン中央（真の理想値は見えない）。__post_init__で確定

    def __post_init__(self):
        self.ideal = (self.lo + self.hi) // 2
        if self.ideal_true == 0:
            self.ideal_true = self.ideal
        if USE_FAKE_INFO and self.kmask != 0b11111:
            self.ideal = self.ideal_from_mask()

    @property
    def center(self) -> int:
        return (self.lo + self.hi) // 2

    def ideal_from_mask(self) -> int:
        """候補の平均を推定理想値にする（偽会心の情報を反映）。"""
        ks = [k for b, k in enumerate(K_OFFSETS) if self.kmask >> b & 1]
        if not ks:
            return self.center
        c = self.center
        return max(self.lo, min(self.hi, c + round(sum(ks) / len(ks))))

    def narrow_by_fake(self, v_before: int, reach: int):
        """偽会心: 理想が (v_before, v_before+reach] に無かった → その候補を除く。全滅なら変えない。"""
        c = self.center
        m = 0
        for b, k in enumerate(K_OFFSETS):
            it = max(self.lo, min(self.hi, c + k))
            if self.kmask >> b & 1 and not (0 < it - v_before <= reach):
                m |= 1 << b
        if m:
            self.kmask = m
            self.ideal = self.ideal_from_mask()

    @property
    def r(self) -> int:
        """推定理想値までの残り（ポリシーの信念）。本会心済みなら理想値=現在値なので0。"""
        if self.pinned:
            return 0
        return self.ideal - self.v

    @property
    def in_zone(self) -> bool:
        return self.lo <= self.v <= self.hi

    @property
    def error(self) -> int:
        """できのよさ誤差は隠し理想値に対して測られる。"""
        e = abs(self.v - self.ideal_true)
        if self.in_zone:
            return min(e, 4)
        return max(e + 3, 9)  # ゾーン外は「実際のズレが 8 以下でも 9 とみなす」（tomahawkch 虹色のオーブ神髄編・魚おじ。2026-10-26 修正。旧: e+3 で 1〜4 のはみ出しが大成功に残っていた）


@dataclass
class Sim:
    cells: dict[str, Cell]
    trait: str
    temp: int = 1000
    focus: int = ASSUMPTIONS["focus_max"]
    rng: random.Random = field(default_factory=random.Random)
    log: list = field(default_factory=list)
    hissatsu_charged: bool = False   # 必殺チャージ発生中（使うと次の1打が会心確定）
    hissatsu_armed: bool = False     # 必殺使用済み→次の1打会心確定
    hissatsu_used: bool = False      # ヘパイトスの炎は1回の鍛冶で1度だけ（DQ10大辞典）
    lit_cell: str | None = None      # 光特性: 現在光っているマス（次の1行動で消える）
    no_hissatsu: bool = False        # 必殺なしモード（実盤面のチャージ抽選を止める。練習用。clone には伝播しない）
    deterministic: bool = False      # 先読み用: 中央値ダメージ固定・会心/必殺抽選なし
    started: bool = False            # 最初の行動を済ませたか（集中力変化の「最初の手は消費そのまま」用。clone にも写す。log は clone で空になるため）
    pairs_v: list = None
    pairs_lr: list = None
    pairs_diag: list = None
    quads: list = None

    # 6マス A,B/C,D/E,F 配置のデフォルト（後方互換）
    PAIRS_V = [("A", "C"), ("C", "E"), ("B", "D"), ("D", "F")]
    PAIRS_LR = [("A", "B"), ("C", "D"), ("E", "F")]
    PAIRS_DIAG = [("B", "C"), ("D", "E")]  # ななめ打ちは左下→右上の方向のみ（mori1確認 2026-09-05）。A-D/C-Fは不可
    QUADS = [("A", "B", "C", "D"), ("C", "D", "E", "F")]

    def __post_init__(self):
        if self.pairs_v is None:
            self.pairs_v = list(self.PAIRS_V)
        if self.pairs_lr is None:
            self.pairs_lr = list(self.PAIRS_LR)
        if self.pairs_diag is None:
            self.pairs_diag = list(self.PAIRS_DIAG)
        if self.quads is None:
            self.quads = list(self.QUADS)

    def can_use(self, skill: str) -> bool:
        return self.focus >= cost_of(skill, self.trait, self.temp, first=not self.started)

    def hit_one(self, name: str, mult_key: str, crit_mult: float, forced_crit=False):
        c = self.cells[name]
        z = turn_multiplier(self.trait, self.temp)
        if name == self.lit_cell:
            z *= 2.0  # 光ったマスはダメージ2倍
        cands = dmg_candidates(self.temp, mult_key)
        d = cands[3] if self.deterministic else self.rng.choice(cands)
        d_eff = math.ceil(d * z)
        # 会心率 = 基礎 × (1 + ねらい6 + 光/会心ターン4)  … dq10-blog 2017/03 の式
        mult = 1.0
        if crit_mult > 1:
            mult += ASSUMPTIONS["aim_bonus"]
        if name == self.lit_cell or is_crit_turn(self.trait, self.temp):
            mult += ASSUMPTIONS["event_bonus_mult"]
        p_crit = ASSUMPTIONS["crit_base"] * mult
        p_crit = min(0.95, p_crit)
        if forced_crit or ((not self.deterministic) and self.rng.random() < p_crit):
            reach = 2 * d_eff
            r_true = c.ideal_true - c.v
            if r_true <= 0:
                # 理想値に到達済み/通り過ぎたマスへの会心は Miss（0）。必殺は DQ10フロンティア記事、
                # 通常の会心も同じ前提にする（mori1さん 2026-10-18）
                return ("miss", 0)
            if 0 < r_true <= reach:
                c.v = c.ideal_true  # 本会心（隠し理想値で停止）
                c.pinned = True     # 理想値が判明（以後この信念を使う）
                return ("crit_pin", d_eff)
            if USE_FAKE_INFO and not self.deterministic:
                c.narrow_by_fake(c.v, reach)
            c.v += 2 * d_eff  # 偽会心
            return ("crit_fake", 2 * d_eff)
        c.v += d_eff
        return ("hit", d_eff)

    def act(self, skill: str, targets: tuple[str, ...] = ()):
        s = SKILLS[skill]
        cost = cost_of(skill, self.trait, self.temp, first=not self.started)
        assert self.focus >= cost, f"集中力不足: {skill}"
        self.started = True
        if skill == "必殺":
            assert self.hissatsu_charged and not self.hissatsu_used
            self.hissatsu_charged = False
            self.hissatsu_armed = True
            self.hissatsu_used = True
            self.focus = min(ASSUMPTIONS["focus_max"], self.focus + ASSUMPTIONS["hissatsu_focus"])
            self.temp = max(0, self.temp + s["dtemp"])  # 必殺使用ターンは温度-50
            self.log.append((skill, (), self.temp, []))
            return []
        results = []
        if s["targets"] != "none":
            forced = self.hissatsu_armed
            self.hissatsu_armed = False
            if s["targets"] == "rand4":     # みだれ打ち: 対象は選ばず、全マスから 4 回ランダム（必殺の後は 4 打とも会心＝mori1さん確認 2026-10-20）
                names = list(self.cells)
                targets = tuple(self.rng.choice(names) for _ in range(4))
            for i, name in enumerate(targets):
                results.append((name, *self.hit_one(name, s["mult"], s["crit"], forced)))
        self.focus -= cost
        # まれに集中力回復（Lv70 特技）: 行動の後に集中が focus_recover_below 以下なら低確率で +30（2026-10-29、本物の棋譜 9/160）
        if (self.focus <= ASSUMPTIONS["focus_recover_below"] and not self.deterministic
                and self.rng.random() < ASSUMPTIONS["focus_recover_p"]):
            self.focus = min(ASSUMPTIONS["focus_max"], self.focus + ASSUMPTIONS["focus_recover_amount"])
            self.focus_recovered = getattr(self, "focus_recovered", 0) + 1
        hit_temp = self.temp
        self.temp = max(0, self.temp + s["dtemp"])
        # 再生（戻り）: 行動の後の温度が 200℃の倍数になると、範囲外でいちばん割合が進んだマスが 12〜16 戻る（mori1さん 2026-10-19、GJさん棋譜 8・10・13局で確認）。
        # 冷やし込み・火力上げ（マスに当たらない行動）でも起きる（2026-10-29 mori1 指摘。本物の棋譜 7 例中 7 例＝アメリー 6・クマタロー 1 で再生あり。以前は「打撃の後だけ」としていた＝誤り）
        if self.trait in MODORI_TRAITS and self.temp % 200 == 0 and self.temp > 0 and not self.deterministic                 and self.rng.random() < ASSUMPTIONS["modori_rate"]:
            # 範囲に入っていないマスのうち、割合がいちばん進んだマス（範囲内のマスは戻らない。GJさん 8局目で全マス範囲内の 800℃は戻りなし）
            cand = [n for n in self.cells if not (self.cells[n].lo <= self.cells[n].v <= self.cells[n].hi)]
            name = max(cand, key=lambda n: self.cells[n].v / max(1, self.cells[n].center)) if cand else None
        else:
            name = None
        if name:
            c = self.cells[name]
            back = self.rng.randint(*ASSUMPTIONS["modori_amount"])
            c.v = max(0, c.v - back)
            c.pinned = False
            results.append((name, "modori", -back))
        # 光は1行動で消灯。200の倍数に到達したら新たにどこかが光る（ゾーン内は光らない）
        self.lit_cell = None
        if self.trait == "光" and not self.deterministic and self.temp % 200 == 0                 and self.temp > 0:
            eligible = [n for n, c in self.cells.items() if not c.in_zone and c.r > 0]
            if eligible and self.rng.random() < ASSUMPTIONS["light_rate"]:
                self.lit_cell = self.rng.choice(eligible)
        self.log.append((skill, targets, self.temp, results))
        # 必殺チャージ抽選: 公式ガイド準拠で「加えたチカラ」比例（決定論モードでは無し）
        # ヘパイトスの炎は1鍛冶1回まで → 使用済みなら再チャージしない
        if not self.deterministic and not self.no_hissatsu and not self.hissatsu_charged                 and not self.hissatsu_armed and not self.hissatsu_used and results:
            # 1打ごとに独立判定（対象マス数ぶん）
            p_hit = ASSUMPTIONS["hissatsu_per_hit"]
            p_charge = 1.0 - (1.0 - p_hit) ** len(results)
            if self.rng.random() < p_charge:
                self.hissatsu_charged = True
        return results

    def clone(self, rng: random.Random | None = None,
              hide_ideal: bool = False) -> "Sim":
        """先読み用の複製。乱数系列は分岐させる。

        hide_ideal=True: 隠し理想値を中央にリセット（プレイヤーは真の理想値を
        知らない前提。先読みロールアウトはこれを使い、本番採点だけが真値を使う）。
        """
        def _ideal(v):
            if v.pinned:
                return v.v  # 本会心済み: 理想値は判明している（隠す必要なし）
            return (v.lo + v.hi) // 2 if hide_ideal else v.ideal_true
        c = Sim(cells={k: Cell(v.lo, v.hi, v.v, ideal_true=_ideal(v), pinned=v.pinned, kmask=v.kmask)
                       for k, v in self.cells.items()},
                trait=self.trait, temp=self.temp, focus=self.focus,
                rng=rng or random.Random(self.rng.random()),
                pairs_v=self.pairs_v, pairs_lr=self.pairs_lr,
                pairs_diag=self.pairs_diag, quads=self.quads)
        c.hissatsu_charged = self.hissatsu_charged
        c.hissatsu_armed = self.hissatsu_armed
        c.hissatsu_used = self.hissatsu_used
        # no_hissatsu は実盤面（練習モード）だけに効かせ、先読み用クローンには引き継がない
        # （mori1: 先読みは「必殺が来るかも」の通常前提で計算する）
        c.lit_cell = self.lit_cell
        c.started = self.started
        return c

    def total_error(self) -> int:
        return sum(c.error for c in self.cells.values())

    def all_in_zone(self) -> bool:
        return all(c.in_zone for c in self.cells.values())


def make_item(item_name: str, rng: random.Random | None = None) -> Sim:
    """recipes.json の任意商材から Sim を作る（マス配置からペア/2x2を自動導出）。

    特性は倍半・光・再生（旧称 戻り。2026-10-19、いちばん割合が進んだマスが 12〜16 戻る。確率は仮）を実装。集中力変化は一部。
    """
    import recipes as _recipes
    rng = rng or random.Random()
    guild, it = _recipes.find(item_name)
    if it is None:
        raise KeyError(item_name)
    grid = _recipes.parse_grid(it["grid"])
    cells = {}
    pos = {}
    idx = 0
    for r, row in enumerate(grid):
        for cidx, cell in enumerate(row):
            if cell and "lo" in cell:
                name = chr(ord("A") + idx)
                ideal = sample_ideal(cell["lo"], cell["hi"], rng)
                cells[name] = Cell(cell["lo"], cell["hi"], ideal_true=ideal)
                pos[name] = (r, cidx)
                idx += 1
    at = {v: k for k, v in pos.items()}
    pairs_v = [(at[(r, c)], at[(r + 1, c)]) for (r, c) in pos.values()
               if (r + 1, c) in at]
    pairs_lr = [(at[(r, c)], at[(r, c + 1)]) for (r, c) in pos.values()
                if (r, c + 1) in at]
    pairs_diag = [(at[(r + 1, c)], at[(r, c + 1)]) for (r, c) in pos.values()
                  if (r + 1, c) in at and (r, c + 1) in at]  # ななめ打ちは左下→右上のみ
    quads = []
    qmin = ASSUMPTIONS.get("quad_min_cells", 3)
    rows_n = len(grid); cols_n = max(len(r) for r in grid) if grid else 0
    for r in range(rows_n - 1):
        for c in range(cols_n - 1):
            q = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            have = [at[x] for x in q if x in at]
            # 盤面の穴を含む 2x2 も選べる（機甲兵のグリーブ [[-,A],[-,B],[C,D]] の超4連は B・C・D の 3 マスに当たる。mori1 本人の棋譜 2026-10-25）
            if len(have) >= max(2, qmin) and len(have) == len(set(have)):
                quads.append(tuple(have))
    trait = it.get("trait", "")
    sim = Sim(cells=cells, trait=trait, rng=rng,
              pairs_v=sorted(set(pairs_v)), pairs_lr=sorted(set(pairs_lr)),
              pairs_diag=sorted(set(pairs_diag)), quads=sorted(set(quads)))
    if rng.random() < ASSUMPTIONS["open_hissatsu"]:
        sim.hissatsu_charged = True
    return sim


def item_layout(item_name: str):
    """商材のマス配置 → (配置キー, {マス名: (行, 列)})。

    配置キーは行ごとにマスの有無を並べた文字列（6マス='11/11/11'、2×2='11/11'、ブーツ='01/01/11' など）。
    付番は make_item と同じ（左上から行→列の順に A, B, …）。GUI の盤面の並びと画面読込の枠（roi_config の layouts）はこのキーで持つ（2026-10-22）。
    """
    import recipes as _recipes
    _, it = _recipes.find(item_name)
    if it is None:
        raise KeyError(item_name)
    pos, mask, idx = {}, [], 0
    for r, row in enumerate(_recipes.parse_grid(it["grid"])):
        m = ""
        for c, cell in enumerate(row):
            if cell and "lo" in cell:
                pos[chr(ord("A") + idx)] = (r, c)
                idx += 1
                m += "1"
            else:
                m += "0"
        mask.append(m)
    return "/".join(mask), pos


def make_amatsuyu(rng: random.Random | None = None) -> Sim:
    """超あまつゆのいと（倍半・6マス）。"""
    rng = rng or random.Random()
    zones = {"A": (180, 190), "B": (245, 251), "C": (150, 156),
             "D": (180, 190), "E": (180, 190), "F": (245, 251)}
    cells = {}
    for k, (lo, hi) in zones.items():
        cells[k] = Cell(lo, hi, ideal_true=sample_ideal(lo, hi, rng))
    sim = Sim(cells=cells, trait="倍半", rng=rng)
    # 光の鍛冶ハンマー: 開幕必殺チャージ
    if rng.random() < ASSUMPTIONS["open_hissatsu"]:
        sim.hissatsu_charged = True
    return sim


def make_kikouhei_ue(rng: random.Random | None = None) -> Sim:
    """機甲兵のよろい上（VER8.1 新防具、防具鍛冶・再生・6マス）。基準値は mori1さん提供の表（2026-10-18）。
    幅が 10 と超素材より広い。GJ channel 配信（P9yRMJLHreA）で 2 回打って どちらも大成功 5 割"""
    rng = rng or random.Random()
    zones = {"A": (235, 244), "B": (105, 113), "C": (125, 133),
             "D": (150, 159), "E": (170, 179), "F": (220, 229)}
    cells = {k: Cell(lo, hi, ideal_true=sample_ideal(lo, hi, rng)) for k, (lo, hi) in zones.items()}
    sim = Sim(cells=cells, trait="再生", rng=rng)
    if rng.random() < ASSUMPTIONS["open_hissatsu"]:
        sim.hissatsu_charged = True
    return sim


# ---------------------------------------------------------------- 補助（ポリシー用）

def safe_max(sim: Sim, name: str, mult_key: str) -> bool:
    """最大ロールでも理想値を超えないか（倍率・特性込み）。"""
    z = turn_multiplier(sim.trait, sim.temp)
    dmax = math.ceil(dmg_candidates(sim.temp, mult_key)[-1] * z)
    return dmax <= sim.cells[name].r


_AW_CACHE = {}


def aim_window(sim: Sim, mult_key: str, temp: int | None = None) -> tuple[int, int]:
    """(z*dmax, 2*z*dmin) = 会心確定レンジ。R がこの範囲なら ねらい系が安全+本会心確定。"""
    t = sim.temp if temp is None else temp
    key = (sim.trait, t, mult_key)
    w = _AW_CACHE.get(key)
    if w is None:
        z = turn_multiplier(sim.trait, t)
        cands = dmg_candidates(t, mult_key)
        w = (math.ceil(cands[-1] * z), 2 * math.ceil(cands[0] * z))
        _AW_CACHE[key] = w
    return w


def aim_window_cell(sim: Sim, name: str, mult_key: str,
                    temp: int | None = None) -> tuple[int, int]:
    """マス個別の理想釣りレンジ（mori1さん定石）。

    下限 = z*dmax + (理想値-ゾーン下限): 非会心でもゾーンに入らず「再挑戦可能」なまま
    上限 = 2*z*dmin: 会心なら本会心確定
    ※偽会心でゾーンに入ると光地金等で再度狙えなくなるため、下限にゾーン猶予を足す。
    実プレイヤーの準備数値表（ゆんまる: B準備183-199 = R52-68）と一致する計算。
    """
    lo_w, hi_w = aim_window(sim, mult_key, temp)
    grace = sim.cells[name].ideal - sim.cells[name].lo
    return lo_w + grace, hi_w
