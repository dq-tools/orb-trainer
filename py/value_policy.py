#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""評価関数（value_net）で手を選ぶ鍛冶の汎用方針と、その学習データ作り（2026-10-27）。

考え方（将棋・2048 の「探索 × 学習した評価」）: 打てる手を全部列挙し、手ごとに出目・会心・再生の分岐を
信念つきクローンで SAMPLES 本引いて、その先の局面の価値 V（大成功確率）の平均が最大の手を打つ。
終盤は厳密解（endgame_solver）をそのまま使う。V は商材に依存しない特徴（kaji_feat）から学ぶ。

学習データ: python value_policy.py gen -i 虹色のオーブ -n 2000 --seed0 0 -w 10 --eps 0.1 -o value_data/orb_t0.npz
  （--model を渡すとその評価関数の自己対局、渡さなければ商材ごとの既存方針＝先生。--eps で時々ランダムな手を混ぜ、
   その直前の局面は記録しない＝ランダムの手で汚れた結果を正解にしない）
学習:        python value_net.py train value_data/orb_t0.npz -o value_models/orb_v1.npz
ベンチ:      python value_policy.py bench -i 虹色のオーブ -n 200 --seed0 0 --model value_models/orb_v1.npz -w 10
"""
import math
import os
import random
import sys
import zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kaji_sim as KS
from kaji_sim import Sim, make_item
import kaji_sim as _KS
import policy_net as PN

SKILL_LIST = list(_KS.SKILLS) + ([] if "必殺" in _KS.SKILLS else ["必殺"])   # 手の番号表（方策の学習用。並びを変えない。kaji_sim.SKILLS に必殺が入っていれば 17 種）
_DEFAULT_POLICY = str(Path(__file__).resolve().parent / "value_models" / "kaji_pol_default.npz")
POLICY = os.environ.get("KAJI_POLICY", _DEFAULT_POLICY if Path(_DEFAULT_POLICY).exists() else "") or None   # 方策の重み。"" で切る
POLICY_TOPK = int(os.environ.get("KAJI_POLICY_TOPK", "6"))   # 方策の上位 K 手だけ評価値で比べる（0 で絞らない）
_POLICY_NET = None


def _policy_net():
    global _POLICY_NET
    if _POLICY_NET is None:
        _POLICY_NET = PN.NumpyPolicy(POLICY)
    return _POLICY_NET


def encode_move(sim, mv) -> np.ndarray:
    """選んだ手 → [技の番号, 対象マスのビット（list(sim.cells) の並び）]（2026-11-01、方策の学習用）。まとめ手は最初の 1 手で記録する"""
    mv = first_step(mv)
    sk = mv[0]; tg = mv[1] if len(mv) > 1 else ()
    idx = SKILL_LIST.index(sk) if sk in SKILL_LIST else -1
    names = list(sim.cells); mask = 0
    for t in (tg if isinstance(tg, (tuple, list, frozenset, set)) else (tg,)):
        if t in names:
            mask |= 1 << names.index(t)
    return np.array([idx, mask], dtype=np.int16)
import kaji_feat as KF
import value_net as VN
import finish_lookahead as FL
import endgame_solver as ES
import run_amatsuyu as R

SAMPLES = 12          # 手ごとの分岐の本数（信念つきクローン。手の間で同じ乱数列＝共通乱数）
USE_SOLVER = True     # 終盤は厳密解
EXACT_SINGLE = True   # 1 マスの技は出目 7 通り × 会心（本会心/偽会心/Miss）を確率つきで全部数える（標本の偶然で「はみ出す目」を見落とさない。2026-10-27）
EXACT_PAIR = True     # 2 マスの技（上下ねらい・上下打ち・左右・ななめ）も各マスの分岐の直積で列挙（確率 EXACT_PRUNE 未満の枝は捨てる）
EXACT_PRUNE = 0.004
DEPTH = 1             # 2 = 上位 TOPK 手は、分岐先でもう 1 手読む（分岐 SAMPLES2 本・その先は V）
TOPK = 5
SAMPLES2 = 4
DEPTH2_EXACT = True   # 2 手読みの 1 手目の分岐も全列挙（1〜2 マスの技）。False で旧来の標本 samples 本
DEPTH2_MAX_BR = 12    # 全列挙の枝を確率の高い順にこの本数まで（上下ねらい 147 本をそのまま読むと重い）
# 深さ 3 以上の探索（value_deep、2026-10-30）。鍵は「何手目の手か」（1=今の手、2=次の手…）
BRANCH_CAP = {1: 12}               # その手の分岐（全列挙の枝）を確率の高い順にこの本数まで（無い段は全部）。1 手目は DEPTH2_MAX_BR と同じ。3 手読みは {1: 6, 2: 6} などを VALUE_SET で
SCREEN_K = {}                      # その段の候補手を篩で上位 K 手に絞る（0 = 絞らない）。篩＝「いちばん確率の高い出目 1 本」の先を評価関数で見た値。
                                   # 実測（オーブ・あまつゆ 2 手目、1000 分岐先）: K=8 で葉 3 割・真の最善手を含む 95〜96%・取りこぼし平均 0.001
SUPER = {"超あまつゆのいと", "超かがやきの樹液", "超ようせいのひだね"}

_NET = None
_MODEL_PATH = None


def _apply_env():
    """環境変数 VALUE_SET="DEPTH=2,ES.ALLOW_LIGHT=True"（ES./KF./FL./VN.（value_net） で他モジュールの定数も上書き）"""
    import re as _re
    for kv in filter(None, _re.split(r",(?![^{}\[\]]*[}\]])", os.environ.get("VALUE_SET", ""))):   # 波括弧・角括弧の中のカンマでは切らない（BRANCH_CAP={1:6,2:6} 用。2026-10-30）
        k, v = kv.split("=", 1)
        k = k.strip()
        if "." in k:
            mod, name = k.split(".", 1)
            setattr({"ES": ES, "KF": KF, "FL": FL, "VN": VN}[mod], name, eval(v))
        else:
            globals()[k] = eval(v)


_apply_env()


class Ensemble:
    """重みを '+' でつないだ複数モデルの平均（1 つのモデルの穴を他が埋める）"""

    def __init__(self, paths):
        from value_net import load_model
        self.nets = [load_model(p) for p in paths]

    def forward(self, X, M, G):
        return sum(n.forward(X, M, G) for n in self.nets) / len(self.nets)

    def forward_cells(self, X, G):
        return sum(n.forward_cells(X, G) for n in self.nets) / len(self.nets)


def load_net(path: str):
    global _NET, _MODEL_PATH
    if path != _MODEL_PATH:
        from value_net import load_model
        _NET = Ensemble(path.split("+")) if "+" in path else load_model(path); _MODEL_PATH = path   # 段階ごとの重み（stage_bounds）も読める
    return _NET


def teacher_for(item: str):
    """商材ごとの既存方針（先生）"""
    if item == "虹色のオーブ":
        import orb_policy as OP
        return OP.next_move
    if item == "超あまつゆのいと":
        return FL.next_move
    if item == "超かがやきの樹液":
        import jueki
        return jueki.next_move_default
    if item == "超ようせいのひだね":
        import hidane
        return hidane.next_move
    import kaji_generic as KG
    return lambda s: KG.next_move(s)


def _seeds(sim: Sim, n: int):
    base = random.Random(zlib.crc32(repr(FL._state_key(sim)).encode("utf-8")))
    return [base.randrange(1 << 30) for _ in range(n)]


def _terminal(s: Sim) -> bool:
    return s.temp <= 0 or (s.focus < 5 and not s.hissatsu_charged)   # 0 ℃は終了（2026-10-31）


def apply_move(s: Sim, mv):
    """手を Sim に打つ（必殺・まとめ手 火力上げ×k も解釈）"""
    if mv[0] == "必殺":
        s.act("必殺")
    elif "×" in mv[0]:
        skill, k = mv[0].split("×")
        for _ in range(int(k)):
            s.act(skill, ())
    else:
        s.act(mv[0], mv[1])


def first_step(mv):
    """まとめ手は最初の 1 回ぶんにする（実際に打つ手）"""
    if mv is not None and "×" in mv[0]:
        return (mv[0].split("×")[0], ())
    return mv


_CB_CACHE = {}


def cell_branches(sim: Sim, n: str, mult_key: str, aim: bool):
    """1 マスに 1 打を当てた時の結果 [(v, pinned, prob)]（出目 7 通り × 非会心/本会心/偽会心/Miss、理想値は範囲で一様）。
    鍵（マスの状態・温度・地金・光・必殺の装填・技）で memo（3 手読みでは同じ鍵が何千回も来る。2026-10-30）。戻り値のリストは書き換えないこと"""
    c = sim.cells[n]
    key = (c.lo, c.hi, c.v, c.pinned, sim.temp, sim.trait, n == sim.lit_cell, sim.hissatsu_armed, mult_key, aim)
    r = _CB_CACHE.get(key)
    if r is None:
        r = _cell_branches(sim, n, mult_key, aim)
        if len(_CB_CACHE) > 500000:
            _CB_CACHE.clear()
        _CB_CACHE[key] = r
    return r


def _cell_branches(sim: Sim, n: str, mult_key: str, aim: bool):
    c = sim.cells[n]
    z = KS.turn_multiplier(sim.trait, sim.temp) * (2.0 if n == sim.lit_cell else 1.0)
    if sim.hissatsu_armed:
        p_crit = 1.0
    else:
        mult = 1.0 + (KS.ASSUMPTIONS["aim_bonus"] if aim else 0.0)
        if n == sim.lit_cell or KS.is_crit_turn(sim.trait, sim.temp):
            mult += KS.ASSUMPTIONS["event_bonus_mult"]
        p_crit = min(0.95, KS.ASSUMPTIONS["crit_base"] * mult)
    w = c.hi - c.lo + 1
    if c.pinned:
        p_miss_all = 1.0
    elif c.v < c.lo:
        p_miss_all = 0.0
    else:
        p_miss_all = min(1.0, (c.v - c.lo + 1) / w)
    branches = {}

    def add(v, pinned, pr):
        if pr <= 0:
            return
        k = (v, pinned)
        branches[k] = branches.get(k, 0.0) + pr

    for d in KS.dmg_candidates(sim.temp, mult_key):
        de = math.ceil(d * z)
        add(c.v + de, c.pinned, (1.0 - p_crit) / 7)
        if p_crit <= 0:
            continue
        pc = p_crit / 7
        add(c.v, c.pinned, pc * p_miss_all)
        if not c.pinned and c.v < c.hi:
            lo_i = max(c.lo, c.v + 1)
            hi_i = min(c.hi, c.v + 2 * de)
            p_pin = max(0, hi_i - lo_i + 1) / w
            if p_pin > 0:
                add((lo_i + hi_i) // 2, True, pc * p_pin)
            if c.v + 2 * de < c.hi:
                p_fake = max(0, c.hi - max(c.lo - 1, c.v + 2 * de)) / w
                add(c.v + 2 * de, False, pc * p_fake)
    return [(v, pinned, pr) for (v, pinned), pr in branches.items()]


_DUMMY_RNG = random.Random(0)


def _after_state(sim: Sim, assign, cost: int, t2: int, k: int) -> Sim:
    """assign={マス: (v, pinned)} を当てはめた後の局面（集中・温度・再生（戻り 14 固定）・光（候補を順に割り当て）を反映）"""
    s = sim.clone(_DUMMY_RNG, hide_ideal=True)     # 葉の局面は乱数を使わない（打つ時は _belief_clone で別の rng）。Random(0) を毎回作ると 48 万回で 3 秒（2026-10-30）
    s.started = True
    s.hissatsu_armed = False
    for n, (v, pinned) in assign.items():
        s.cells[n].v = v
        s.cells[n].pinned = pinned
        if pinned:
            s.cells[n].ideal_true = v
    s.focus -= cost
    s.temp = t2
    if s.trait in KS.MODORI_TRAITS and t2 % 200 == 0 and t2 > 0:
        cand = [m for m in s.cells if not (s.cells[m].lo <= s.cells[m].v <= s.cells[m].hi)]
        if cand:
            tgt = max(cand, key=lambda m: KS.modori_key(s.cells[m]))
            s.cells[tgt].v = max(0, s.cells[tgt].v - 14)
            s.cells[tgt].pinned = False
    s.lit_cell = None
    if s.trait == "光" and t2 % 200 == 0 and t2 > 0:
        elig = [m for m, cc in s.cells.items() if not cc.in_zone and cc.r > 0]
        if elig:
            s.lit_cell = elig[k % len(elig)]
    return s


def exact_outcomes(sim: Sim, mv):
    """1 マス／2 マスの技の分岐を確率つきで列挙 [(after_sim, prob)]。2 マスは各マスの分岐の直積（確率 EXACT_PRUNE 未満の枝は捨てて正規化）。
    必殺のチャージ抽選は無視（2026-10-27）。"""
    skill, targets = mv
    sk = KS.SKILLS[skill]
    aim = sk["crit"] > 1
    per = [cell_branches(sim, n, sk["mult"], aim) for n in targets]
    combos = [({}, 1.0)]
    for n, br in zip(targets, per):
        combos = [(dict(a, **{n: (v, pinned)}), pr * q) for a, pr in combos for v, pinned, q in br]
        if len(targets) > 1:
            combos = [(a, pr) for a, pr in combos if pr >= EXACT_PRUNE]
    tot = sum(pr for _, pr in combos)
    if tot <= 0:
        return []
    cost = KS.cost_of(skill, sim.trait, sim.temp, first=not sim.started)
    t2 = max(0, sim.temp + sk["dtemp"])
    return [(_after_state(sim, a, cost, t2, k), pr / tot) for k, (a, pr) in enumerate(combos)]


def single_outcomes(sim: Sim, mv):
    return exact_outcomes(sim, mv)


def _branches(sim: Sim, mv, seeds, samples: int, max_br: int | None = None, use_exact: bool = True):
    """手 mv の分岐 [(打った後の局面 or None, 確率)]。1〜2 マスの技は全列挙（EXACT_SINGLE / EXACT_PAIR。max_br なら確率の高い枝に絞って正規化）、
    他は信念つきクローン samples 本（AssertionError＝打てない分岐は None）"""
    if use_exact:
        tg = KS.SKILLS[mv[0]]["targets"] if mv[0] in KS.SKILLS else None
        exact = (EXACT_SINGLE and tg == "cell" and len(mv[1]) == 1) or (EXACT_PAIR and tg in ("pair_v", "pair_lr", "pair_diag") and len(mv[1]) == 2)
        if exact:
            outs = exact_outcomes(sim, mv)
            if outs:
                if max_br is not None and len(outs) > max_br:
                    outs = sorted(outs, key=lambda x: -x[1])[:max_br]
                    z = sum(pr for _, pr in outs)
                    outs = [(s, pr / z) for s, pr in outs]
                return outs
    brs = []
    for sd in seeds:
        s = FL._belief_clone(sim, random.Random(sd))
        try:
            apply_move(s, mv)
        except AssertionError:
            brs.append((None, 1.0 / samples))
            continue
        brs.append((s, 1.0 / samples))
    return brs


def _leaf_values(sims, net) -> np.ndarray:
    """葉の局面の値を一括で（終局＝集中 5 未満は厳密な大成功確率、他は評価関数）。featurize_many で 1 局面ずつの 9 倍速（2026-10-30）"""
    if not sims:
        return np.zeros(0)
    meta = KF.ItemMeta(sims[0])
    return _leaf_values_arrays(meta, KF.arrays_from_sims(meta, sims), net)


def _leaf_values_arrays(meta, A: dict, net) -> np.ndarray:
    """配列表現の葉の値。終局は p_great、他は評価関数（詰め物なしの forward_cells）"""
    B = len(A["temp"])
    out = np.zeros(B)
    if B == 0:
        return out
    term = KF.terminal_mask(A)
    if term.any():
        sub = {k: x[term] for k, x in A.items()}
        out[term] = KF.p_great_of_arrays(meta, sub)
    live = ~term
    if live.any():
        sub = {k: x[live] for k, x in A.items()}
        X, G = KF.featurize_arrays(meta, sub)
        out[live] = net.forward_cells(X, G) if hasattr(net, "forward_cells") else net.forward(*KF.pad_many(X), G)
    return out


def _leaf_arrays_from(sim: Sim, meta, targets, vs, pins, probs, cost, t2, ks):
    """対象マスごとの値の列 vs[i] / pins[i]（長さ B）を当てはめた葉 B 局面を配列で。ks は光の候補の番目（_after_state の k）"""
    B = len(probs); idx = meta.idx
    v0 = np.array([sim.cells[k].v for k in meta.names], dtype=np.float64)
    p0 = np.array([sim.cells[k].pinned for k in meta.names], dtype=bool)
    v = np.repeat(v0[None, :], B, axis=0); pinned = np.repeat(p0[None, :], B, axis=0)
    for nm, vv, pp in zip(targets, vs, pins):
        v[:, idx[nm]] = vv; pinned[:, idx[nm]] = pp
    lo = meta.lo[None, :]; hi = meta.hi[None, :]
    if meta.regen and t2 % 200 == 0 and t2 > 0:
        outz = ~((lo <= v) & (v <= hi))
        ratio = np.where(outz, v / np.maximum(1.0, meta.center)[None, :], -1.0)
        tgt = ratio.argmax(1)
        rows = np.where(outz.any(1))[0]
        v[rows, tgt[rows]] = np.maximum(0.0, v[rows, tgt[rows]] - 14)
        pinned[rows, tgt[rows]] = False
    lit = np.full(B, -1, dtype=np.int64)
    if sim.trait == "光" and t2 % 200 == 0 and t2 > 0:
        elig = (~pinned) & ~((lo <= v) & (v <= hi)) & (v < meta.center[None, :])     # Cell.r > 0 ＝ 中央（hide_ideal 後の理想）より手前
        for k in range(B):
            e = np.where(elig[k])[0]
            if len(e):
                lit[k] = e[ks[k] % len(e)]
    return dict(temp=np.full(B, t2, dtype=np.int64), focus=np.full(B, float(sim.focus - cost)), v=v, pinned=pinned, lit=lit,
                hc=np.full(B, sim.hissatsu_charged, dtype=bool), ha=np.zeros(B, dtype=bool), hu=np.full(B, sim.hissatsu_used, dtype=bool),
                started=np.ones(B, dtype=bool))


def exact_leaf_arrays(sim: Sim, mv, meta):
    """exact_outcomes(sim, mv) の葉を Sim を作らず配列で: (A dict, probs [B])。_after_state と同じ規則
    （集中・温度・再生 14 固定・光は候補を k 番目に・必殺の装填は解除・started=True）。1 局面あたりの費用が clone の 1/20（2026-10-30）。
    枝の並びは exact_outcomes と同じ（1 マス目の枝 × 2 マス目の枝、確率 EXACT_PRUNE 未満の組は捨てて正規化）"""
    skill, targets = mv
    sk = KS.SKILLS[skill]
    aim = sk["crit"] > 1
    per = [cell_branches(sim, n, sk["mult"], aim) for n in targets]
    cost = KS.cost_of(skill, sim.trait, sim.temp, first=not sim.started)
    t2 = max(0, sim.temp + sk["dtemp"])
    if len(targets) == 1:
        br = per[0]
        vs = [np.array([x[0] for x in br], dtype=np.float64)]
        pins = [np.array([x[1] for x in br], dtype=bool)]
        probs = np.array([x[2] for x in br], dtype=np.float64)
    elif len(targets) == 2:
        b0, b1 = per
        pr = np.array([x[2] for x in b0])[:, None] * np.array([y[2] for y in b1])[None, :]
        keep = pr >= EXACT_PRUNE
        i0, i1 = np.nonzero(keep)                      # 行優先＝exact_outcomes の並び
        if len(i0) == 0:
            return None, None
        vs = [np.array([x[0] for x in b0], dtype=np.float64)[i0], np.array([y[0] for y in b1], dtype=np.float64)[i1]]
        pins = [np.array([x[1] for x in b0], dtype=bool)[i0], np.array([y[1] for y in b1], dtype=bool)[i1]]
        probs = pr[i0, i1]
    else:
        return None, None
    tot = probs.sum()
    if tot <= 0:
        return None, None
    probs = probs / tot
    return _leaf_arrays_from(sim, meta, targets, vs, pins, probs, cost, t2, np.arange(len(probs))), probs


def most_probable_leaf_arrays(sim: Sim, mv, meta):
    """篩用: いちばん確率の高い枝 1 本だけの葉（マスごとに最頻の枝を取る＝組の確率も最大）。光の候補の番目は 0 で近似"""
    skill, targets = mv
    sk = KS.SKILLS[skill]
    aim = sk["crit"] > 1
    vs, pins = [], []
    for n in targets:
        br = cell_branches(sim, n, sk["mult"], aim)
        if not br:
            return None
        v, pnd, _ = max(br, key=lambda x: x[2])
        vs.append(np.array([v], dtype=np.float64)); pins.append(np.array([pnd], dtype=bool))
    cost = KS.cost_of(skill, sim.trait, sim.temp, first=not sim.started)
    t2 = max(0, sim.temp + sk["dtemp"])
    return _leaf_arrays_from(sim, meta, targets, vs, pins, np.ones(1), cost, t2, np.zeros(1, dtype=np.int64))


def _is_exact(mv) -> bool:
    tg = KS.SKILLS[mv[0]]["targets"] if mv[0] in KS.SKILLS else None
    return (EXACT_SINGLE and tg == "cell" and len(mv[1]) == 1) or (EXACT_PAIR and tg in ("pair_v", "pair_lr", "pair_diag") and len(mv[1]) == 2)


def value_of_moves(sim: Sim, net, moves, samples: int | None = None):
    """手ごとの期待価値。1〜2 マスの技は分岐を確率つきで列挙（EXACT_SINGLE / EXACT_PAIR）、他は信念つきクローン samples 本の平均。終局した分岐は厳密な大成功確率。
    葉は全部集めて featurize_many → forward 1 回（2026-10-30）"""
    samples = SAMPLES if samples is None else samples
    seeds = _seeds(sim, samples)
    leaves, slots = [], []          # slots: (手の番号, 葉の番号 or None, 厳密値, 重み)
    for mi, mv in enumerate(moves):
        for s, pr in _branches(sim, mv, seeds, samples):
            if s is None:
                slots.append((mi, None, 0.0, pr))
            else:
                slots.append((mi, len(leaves), None, pr)); leaves.append(s)
    lv = _leaf_values(leaves, net)
    vals = np.zeros(len(moves))
    for mi, li, ex, wt in slots:
        vals[mi] += (ex if li is None else lv[li]) * wt
    return vals


def _most_probable_leaf(meta, A: dict, probs) -> dict:
    k = int(np.argmax(probs))
    return {key: x[k:k + 1] for key, x in A.items()}


def _expand_level(sims, net, meta, ply: int, remaining: int) -> np.ndarray:
    """局面の列 sims（同じ商材）の値を返す。remaining=0 なら葉（終局は厳密値・他は評価関数）。
    remaining>=1 なら各局面で ply 手目の候補（SCREEN_K[ply] で篩）を BRANCH_CAP[ply] 本ずつ分岐して 1 段深く読み、
    値 = max(今終えた時の厳密値, 候補ごとの期待値の最大)。葉は配列で作り、1 段分をまとめて評価する"""
    n_nodes = len(sims)
    if n_nodes == 0:
        return np.zeros(0)
    if remaining == 0:
        return _leaf_values(sims, net)
    stop = KF.p_great_many(sims)
    vals = stop.copy()
    term = KF.terminal_mask(KF.arrays_from_sims(meta, sims))
    cands = []                                   # (node, m2)
    for b, s in enumerate(sims):
        if term[b]:
            cands.append((b, []))
            continue
        cands.append((b, KF.legal_moves(s)))
    K = SCREEN_K.get(ply, 0)
    cap = BRANCH_CAP.get(ply, None)
    # --- 篩: いちばん確率の高い出目 1 本の先を評価関数で見て、候補を上位 K 手に ---
    if K and any(len(m2) > K for _, m2 in cands):
        parts, owner = [], []
        for b, m2 in cands:
            if len(m2) <= K:
                continue
            s = sims[b]
            seeds_s = _seeds(s, 1)
            for j, mv2 in enumerate(m2):
                if _is_exact(mv2):
                    A = most_probable_leaf_arrays(s, mv2, meta)
                    if A is not None:
                        parts.append(A); owner.append((b, j)); continue
                one = _branches(s, mv2, seeds_s, 1, use_exact=False)
                if one and one[0][0] is not None:
                    parts.append(KF.arrays_from_sims(meta, [one[0][0]])); owner.append((b, j))
        if parts:
            sv = _leaf_values_arrays(meta, KF.concat_arrays(parts), net)
            score = {}
            for (b, j), v in zip(owner, sv):
                score.setdefault(b, {})[j] = float(v)
            for i, (b, m2) in enumerate(cands):
                if len(m2) > K and b in score:
                    keep = sorted(score[b], key=lambda j: -score[b][j])[:K]
                    cands[i] = (b, [m2[j] for j in sorted(keep)])
    # --- 分岐の展開 ---
    acc = {}                                     # (node, 候補の番号) -> 期待値
    if remaining == 1:
        parts, cells, leaves, nleaf = [], [], [], 0     # 葉は配列（exact）か Sim（標本）
        for b, m2 in cands:
            s = sims[b]
            seeds_s = _seeds(s, SAMPLES2)
            for j, mv2 in enumerate(m2):
                if _is_exact(mv2):
                    A, probs = exact_leaf_arrays(s, mv2, meta)
                    if A is not None:
                        if cap is not None and len(probs) > cap:
                            top = np.argsort(-probs)[:cap]
                            A = {key: x[top] for key, x in A.items()}; probs = probs[top] / probs[top].sum()
                        k0 = nleaf; nleaf += len(probs)
                        parts.append(A)
                        cells.extend((b, j, k0 + k, float(pr)) for k, pr in enumerate(probs))
                        continue
                for s2, pr2 in _branches(s, mv2, seeds_s, SAMPLES2, use_exact=False):
                    if s2 is None:
                        cells.append((b, j, -1, pr2))
                    else:
                        cells.append((b, j, nleaf, pr2)); leaves.append(s2); nleaf += 1; parts.append(None)
        arrs = []; si = 0
        for q in parts:
            if q is None:
                arrs.append(KF.arrays_from_sims(meta, [leaves[si]])); si += 1
            else:
                arrs.append(q)
        lv = _leaf_values_arrays(meta, KF.concat_arrays(arrs), net) if arrs else np.zeros(0)
        for b, j, li, wt in cells:
            acc[(b, j)] = acc.get((b, j), 0.0) + (0.0 if li < 0 else float(lv[li])) * wt
    else:
        children, cinfo = [], []                 # 子は Sim（さらに手を読むので）
        for b, m2 in cands:
            s = sims[b]
            seeds_s = _seeds(s, SAMPLES2)
            for j, mv2 in enumerate(m2):
                for s2, pr2 in _branches(s, mv2, seeds_s, SAMPLES2, max_br=cap):
                    if s2 is None:
                        acc[(b, j)] = acc.get((b, j), 0.0)
                    else:
                        children.append(s2); cinfo.append((b, j, pr2))
        cv = _expand_level(children, net, meta, ply + 1, remaining - 1)
        for (b, j, pr2), v in zip(cinfo, cv):
            acc[(b, j)] = acc.get((b, j), 0.0) + pr2 * float(v)
    for (b, j), v in acc.items():
        if v > vals[b]:
            vals[b] = v
    return vals


def value_deep(sim: Sim, net, moves, samples: int, depth: int):
    """手ごとの期待値を depth 手読みで（depth=2 が従来の value_depth2）。1 手目の分岐は全列挙（BRANCH_CAP[1] 本）か標本 samples 本、
    その先は _expand_level（段ごとに篩 SCREEN_K と枝の上限 BRANCH_CAP）"""
    seeds = _seeds(sim, samples)
    meta = KF.ItemMeta(sim)
    bsims, binfo = [], []
    for mi, mv in enumerate(moves):
        for s, pr in _branches(sim, mv, seeds, samples, max_br=BRANCH_CAP.get(1, DEPTH2_MAX_BR) if DEPTH2_EXACT else None, use_exact=DEPTH2_EXACT):
            if s is None:
                continue
            bsims.append(s); binfo.append((mi, pr))
    if not bsims:
        return np.zeros(len(moves))
    bv = _expand_level(bsims, net, meta, 2, depth - 1)
    vals = np.zeros(len(moves))
    for (mi, pr), v in zip(binfo, bv):
        vals[mi] += pr * float(v)
    return vals


def value_depth2(sim: Sim, net, moves, samples: int):
    """2 手読み（value_deep の depth=2。2026-10-29 ④ → 2026-10-30 に一般化）"""
    return value_deep(sim, net, moves, samples, 2)


SCRIPT_EXT = False         # あまつゆの台本を 8〜11 手目まで伸ばす（ソルティーさんの型: 1950 熱風 F → 1800 たたく → 1750 熱風 → 1600 熱風 B）。1000 局で決める（2026-11-03）
OPENING_SCRIPT = True      # 倍半の商材は学習型の前に手書きの開幕台本（kaji_generic.baihan_opening＝GJ さん・mori1 の型: 火力上げ×2 → 1600 熱風 → 火力上げ×2 → 2050 → 2000 超4連）を使う（2026-11-03）


HIDANE_SCRIPT_MIN_TEMP = 1600   # ひだね（光・6 マス・必要量 ≥1200）: この温度以上は手書き hidane の手（火力上げ×4 → 2200 → 光ったマスへの応答）。None で使わない


_ZONE_SIG = {}


def _is_item(sim: Sim, item: str) -> bool:
    """Sim に商材名が無いので、範囲の組（lo, hi の並び）で見分ける"""
    sig = _ZONE_SIG.get(item)
    if sig is None:
        try:
            ref = make_item(item, random.Random(0))
            sig = tuple(sorted((c.lo, c.hi) for c in ref.cells.values()))
        except Exception:
            sig = ()
        _ZONE_SIG[item] = sig
    return bool(sig) and tuple(sorted((c.lo, c.hi) for c in sim.cells.values())) == sig


def hidane_script_move(sim: Sim):
    if HIDANE_SCRIPT_MIN_TEMP is None or sim.trait != "光" or len(sim.cells) != 6 or sim.temp < HIDANE_SCRIPT_MIN_TEMP:
        return None
    if not _is_item(sim, "超ようせいのひだね"):
        return None
    try:
        import hidane as HD
        mv = HD.next_move(sim)
    except Exception:
        return None
    if mv is None:
        return None
    sk, tg = mv[0], tuple(mv[1]) if len(mv) > 1 and mv[1] else ()
    if sk == "必殺":
        return ("必殺", ()) if (sim.hissatsu_charged and not sim.hissatsu_used) else None
    if not sim.can_use(sk):
        return None
    return (sk, tg)


JUEKI_SCRIPT_MIN_TEMP = 2000    # 樹液（集中力変化・6 マス・必要量 ≥1200）: この温度以上は手書き jueki の手（開幕 A ＋ 会心ターンの面作り）。None で使わない


def jueki_script_move(sim: Sim):
    if JUEKI_SCRIPT_MIN_TEMP is None or sim.trait != "集中力変化" or len(sim.cells) != 6:
        return None
    if not _is_item(sim, "超かがやきの樹液"):
        return None
    try:
        import jueki as JK
        mv = JK.opening_step(sim, JK.OPENINGS[JK.DEFAULT_OPENING])     # 開幕の台本が続く間は温度に関わらず（1000→1300 の火力上げも）
        if mv is None:
            if sim.temp < JUEKI_SCRIPT_MIN_TEMP:
                return None
            mv = JK.next_move_default(sim)
    except Exception:
        return None
    if mv is None:
        return None
    sk, tg = mv[0], tuple(mv[1]) if len(mv) > 1 and mv[1] else ()
    if sk == "必殺":
        return ("必殺", ()) if (sim.hissatsu_charged and not sim.hissatsu_used) else None
    if not sim.can_use(sk):
        return None
    return (sk, tg)


ORB_SCRIPT_MIN_TEMP = 1500      # 虹色のオーブ: この温度以上は手書き orb_policy（アメリーさん型）の手。None で使わない


def orb_script_move(sim: Sim):
    if ORB_SCRIPT_MIN_TEMP is None or sim.trait not in ("再生", "戻り") or len(sim.cells) != 6 or sim.temp < ORB_SCRIPT_MIN_TEMP:
        return None
    if not _is_item(sim, "虹色のオーブ"):
        return None
    try:
        import orb_policy as OP
        mv = OP.next_move(sim)
    except Exception:
        return None
    if mv is None:
        return None
    sk, tg = mv[0], tuple(mv[1]) if len(mv) > 1 and mv[1] else ()
    if sk == "必殺":
        return ("必殺", ()) if (sim.hissatsu_charged and not sim.hissatsu_used) else None
    if not sim.can_use(sk):
        return None
    return (sk, tg)


def opening_script_move(sim: Sim):
    """開幕台本の手（該当しなければ None）。倍半だけ。mori1 の型（2026-11-03 の 3 局）:
    火力上げ×2 → 1600（倍ターン）は「2 倍の熱風が範囲に入るマス」があればそこへ熱風（あまつゆ A: 180〜190 に +183〜187）、無ければ必要量最大のマス →
    火力上げ×2 → 2050 は ななめ打ち（−50 で 2000 に乗せる。超4連を打つ面の中の、残りが大きい 2 マス）→ 2000（倍ターン）超4連。以後は学習型"""
    if sim.trait != "倍半" or not sim.quads:
        return None
    cells = sim.cells
    touched = [n for n, c in cells.items() if c.v > 0]
    temp = sim.temp
    if not touched:
        if temp < 1600:
            return ("火力上げ", ()) if sim.can_use("火力上げ") else None
        if temp == 1600 and sim.can_use("熱風おろし"):
            # 本人の選び方: 非会心（倍ターンの出目）でははみ出さず、会心（さらに 2 倍）なら範囲に届いて本会心で止まるマス。
            # 届く目の割合が最大 → 範囲の幅が広い → 名前順（あまつゆは A。3 局中 2 局が実際に会心で +183／+187）
            z = KS.turn_multiplier(sim.trait, temp)
            ds = [math.ceil(x * z) for x in KS.dmg_candidates(temp, KS.SKILLS["熱風おろし"]["mult"])]
            best = None
            for n, c in cells.items():
                if any(c.v + d > c.hi for d in ds):
                    continue
                reach = sum(1 for d in ds if c.v + 2 * d >= c.lo) / len(ds)
                if reach < 5 / 7:
                    continue
                key = (c.hi - c.lo, c.r, -ord(n[0]))        # 幅が広い（会心で止まる余地）→ 必要量が大きい → 名前順
                if best is None or key > best[0]:
                    best = (key, n)
            if best:
                return ("熱風おろし", (best[1],))
            n = max(cells, key=lambda k: cells[k].r)
            ds = [math.ceil(x * z) for x in KS.dmg_candidates(temp, KS.SKILLS["熱風おろし"]["mult"])]
            if cells[n].v + max(ds) <= cells[n].hi:
                return ("熱風おろし", (n,))
        return None
    if len(touched) == 1 and 1400 <= temp < 2050 and temp % 400 != 0:
        return ("火力上げ", ()) if sim.can_use("火力上げ") else None
    if len(touched) == 1 and temp == 2050:
        # 超4連を打つ面（触っていないマスだけの面で、必要量の合計が最大）の中の ななめ 2 マス（残りが大きい組）
        quads = [q for q in sim.quads if all(cells[n].v == 0 for n in q)]
        if quads and sim.can_use("ななめ打ち"):
            q = max(quads, key=lambda q: sum(cells[n].r for n in q))
            pairs = [p for p in (sim.pairs_diag or []) if set(p) <= set(q)]
            if pairs:
                ds = [math.ceil(x * KS.turn_multiplier(sim.trait, temp)) for x in KS.dmg_candidates(temp, KS.SKILLS["ななめ打ち"]["mult"])]
                safe = [p for p in pairs if all(cells[n].v + max(ds) <= cells[n].hi - 60 for n in p)]
                if safe:
                    p = max(safe, key=lambda p: sum(cells[n].r for n in p))
                    return ("ななめ打ち", tuple(p))
        if sim.can_use("4連打ち"):
            try:
                import kaji_generic as KG
                mv = KG.baihan_opening(sim)
                if mv is not None and mv[0] == "4連打ち":
                    return ("4連打ち", tuple(mv[1]))
            except Exception:
                pass
        return None
    if temp == 2000 and sim.can_use("超4連打ち"):
        try:
            import kaji_generic as KG
            mv = KG.baihan_opening(sim)
            if mv is not None and mv[0] == "超4連打ち":
                return ("超4連打ち", tuple(mv[1]))
        except Exception:
            pass
    if SCRIPT_EXT and len(touched) >= 4 and 1600 <= temp <= 1950:
        # ソルティーさんの 8〜11 手目（21 局）: 1950 熱風 F → 1800 たたく → 1750 熱風 → 1600（倍）熱風 B（触っていない大マスを倍ターンまで温存）
        untouched = [n for n, c in cells.items() if c.v == 0]
        z = KS.turn_multiplier(sim.trait, temp)

        def safe(sk, n):
            ds = [math.ceil(x * z) for x in KS.dmg_candidates(temp, KS.SKILLS[sk]["mult"])]
            return cells[n].v + max(ds) <= cells[n].hi and cells[n].r > max(ds) + 20
        if temp == 1600 and len(untouched) == 1 and sim.can_use("熱風おろし") and safe("熱風おろし", untouched[0]):
            return ("熱風おろし", (untouched[0],))
        if temp in (1950, 1750, 1800) and untouched:
            cand = [n for n in touched if not cells[n].pinned]
            if cand:
                n = max(cand, key=lambda k: cells[k].r)
                sk = "たたく" if temp == 1800 else "熱風おろし"
                if sim.can_use(sk) and safe(sk, n):
                    return (sk, (n,))
    return None


def next_move(sim: Sim, net=None, samples: int | None = None):
    samples = SAMPLES if samples is None else samples
    net = net or _NET
    if OPENING_SCRIPT:
        mv = opening_script_move(sim)
        if mv is not None:
            return mv
    mv = hidane_script_move(sim) or jueki_script_move(sim) or orb_script_move(sim)
    if mv is not None:
        return mv
    if USE_SOLVER and ES.applicable(sim):
        mv, _, _ = ES.solve(sim)
        if mv != "toobig":
            return mv
    moves = KF.legal_moves(sim)
    if not moves:
        return None
    stop_v = KF.p_great_now(sim)
    if POLICY is not None and POLICY_TOPK > 0 and len(moves) > POLICY_TOPK:
        # 方策（policy_net）の上位 K 手だけ評価値で比べる（2026-11-01、裁縫で +2〜4）。まとめ手は最初の 1 手で採点
        X, g = KF.featurize(sim); Xp, m = KF.pad(X)
        sc = PN.score_moves(_policy_net(), Xp, m, g, [first_step(mv) for mv in moves], list(sim.cells), SKILL_LIST)
        keep = sorted(np.argsort(-sc)[:POLICY_TOPK])
        moves = [moves[i] for i in keep]
    vals = value_of_moves(sim, net, moves, samples)
    if DEPTH >= 2 and len(moves) > 1:
        top = [int(i) for i in np.argsort(-vals)[:TOPK]]
        v2 = value_deep(sim, net, [moves[i] for i in top], samples, DEPTH)
        vals = np.full(len(moves), -1.0)
        for i, v in zip(top, v2):
            vals[i] = v
    bi = int(np.argmax(vals))
    if vals[bi] <= stop_v:
        return None
    return first_step(moves[bi])


def play(item: str, seed: int, net=None, samples: int | None = None) -> Sim:
    sim = make_item(item, random.Random(seed))
    for _ in range(100):
        if _terminal(sim):
            break
        mv = next_move(sim, net, samples)
        if mv is None:
            break
        sim.act(mv[0], mv[1]) if mv[0] != "必殺" else sim.act("必殺")
    return sim


def is_great(sim: Sim) -> bool:
    return sim.total_error() <= KF.great_thr(sim)


RAW_LEN = 4 + 4 * KF.MAX_CELLS + 8


def raw_state(sim: Sim) -> np.ndarray:
    """特徴を作り直せるよう、局面そのものも小さく記録する（温度・集中・各マスの lo/hi/v/pinned・地金・光・必殺・started）。
    将来 kaji_feat の特徴を変えた時に、学習データを作り直さずに済む（2026-10-27）"""
    r = np.zeros(RAW_LEN, dtype=np.float32)
    r[0] = sim.temp; r[1] = sim.focus; r[2] = len(sim.cells); r[3] = KF.TRAITS.index(KF._trait_name(sim.trait)) if KF._trait_name(sim.trait) in KF.TRAITS else 0
    names = list(sim.cells)
    for i, n in enumerate(names[:KF.MAX_CELLS]):
        c = sim.cells[n]
        r[4 + i * 4: 8 + i * 4] = (c.lo, c.hi, c.v, 1.0 if c.pinned else 0.0)
    b = 4 + 4 * KF.MAX_CELLS
    r[b] = names.index(sim.lit_cell) if sim.lit_cell in names else -1
    r[b + 1] = sim.hissatsu_charged; r[b + 2] = sim.hissatsu_armed; r[b + 3] = sim.hissatsu_used; r[b + 4] = sim.started
    return r


# ---------------------------------------------------------------- 学習データ

def gen_game(item: str, seed: int, eps: float, model: str | None, start=None):
    """1 局打って (X[n,9,FC], M[n,9], G[n,FG], y) を返す。
    start=(割合, "teacher"|"random", 最長手数): その割合の局は、別の打ち手（先生 or ランダム）で 1〜最長手数 打った途中局面から始める
    （carousel shaping。自分の 1 手読みの棋譜にしか無い局面に過学習するのを崩す、2026-10-29）。前置きの手は記録しない"""
    rng = random.Random(seed * 7919 + 17)
    sim = make_item(item, random.Random(seed))
    teach = teacher_for(item)

    def teach_pol(s):
        try:
            return teach(s)
        except Exception:
            return FL.choose(s)
    if model:
        net = load_net(model)
        pol = lambda s: next_move(s, net)
    else:
        pol = teach_pol
    rows = []
    prefix = 0
    if start and rng.random() < start[0]:
        prefix = rng.randint(1, int(start[2]))
    for _ in range(100):
        if _terminal(sim):
            break
        record = True
        if prefix > 0:
            prefix -= 1
            if start[1] == "random":
                lm = KF.legal_moves(sim)
                if not lm:
                    break
                mv = first_step(rng.choice(lm))
            else:
                mv = teach_pol(sim)
            if mv is None:
                break
            sim.act(mv[0], mv[1]) if mv[0] != "必殺" else sim.act("必殺")
            continue
        if eps > 0 and rng.random() < eps:
            lm = KF.legal_moves(sim)
            if not lm:
                break
            mv = first_step(rng.choice(lm)); record = False
        else:
            mv = pol(sim)
        if mv is None:
            break
        if record:
            X, g = KF.featurize(sim)
            Xp, m = KF.pad(X)
            rows.append((Xp, m, g, raw_state(sim), encode_move(sim, mv)))   # 選んだ手も記録（方策の学習用、2026-11-01）
        sim.act(mv[0], mv[1]) if mv[0] != "必殺" else sim.act("必殺")
    y = 1.0 if is_great(sim) else 0.0
    return rows, y


def save_rows(path, res):
    """gen の結果（[(rows, y)]）を npz に（X, M, G, y, R, A）。rows の 5 列目（A）が無い古い形も読める"""
    Xs, Ms, Gs, ys, Rs, As = [], [], [], [], [], []
    greats = 0
    for rows, y in res:
        greats += y
        for row in rows:
            Xp, m, g_, r_ = row[:4]
            Xs.append(Xp); Ms.append(m); Gs.append(g_); ys.append(y); Rs.append(r_)
            As.append(row[4] if len(row) > 4 else np.array([-1, 0], dtype=np.int16))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.array(Xs, dtype=np.float32), M=np.array(Ms, dtype=np.float32), G=np.array(Gs, dtype=np.float32),
                        y=np.array(ys, dtype=np.float32), R=np.array(Rs, dtype=np.float32), A=np.array(As, dtype=np.int16))
    return len(ys), greats


def _gen_one(a):
    item, seed, eps, model = a
    rows, y = gen_game(item, seed, eps, model)
    return rows, y


def _bench_one(a):
    item, seed, model, samples = a
    net = load_net(model)
    s = play(item, seed, net, samples)
    return is_great(s), s.total_error(), sum(1 for c in s.cells.values() if c.pinned), s.focus


def _teacher_one(a):
    item, seed = a
    rows, y = gen_game(item, seed, 0.0, None)
    return y


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gen"); g.add_argument("-i", "--items", nargs="+", required=True); g.add_argument("-n", type=int, default=1000)
    g.add_argument("--seed0", type=int, default=0); g.add_argument("-w", type=int, default=10); g.add_argument("--eps", type=float, default=0.1)
    g.add_argument("--model", default=None); g.add_argument("-o", required=True)
    b = sub.add_parser("bench"); b.add_argument("-i", "--items", nargs="+", required=True); b.add_argument("-n", type=int, default=200)
    b.add_argument("--seed0", type=int, default=0); b.add_argument("-w", type=int, default=10); b.add_argument("--model", required=True)
    b.add_argument("--samples", type=int, default=SAMPLES)
    t = sub.add_parser("trace"); t.add_argument("-i", "--item", required=True); t.add_argument("--seed", type=int, default=0); t.add_argument("--model", required=True)
    a = ap.parse_args()
    if a.cmd == "gen":
        jobs = [(it, sd, a.eps, a.model) for it in a.items for sd in range(a.seed0, a.seed0 + a.n)]
        with Pool(a.w) as pool:
            res = pool.map(_gen_one, jobs, chunksize=4)
        n_rows, greats = save_rows(a.o, res)
        print(f"{a.o}: {len(jobs)} 局 → {n_rows} 局面  大成功 {greats / len(jobs) * 100:.1f}%  （{' '.join(a.items)}, eps {a.eps}, {'自己対局 ' + a.model if a.model else '先生'}）")
    elif a.cmd == "bench":
        for it in a.items:
            jobs = [(it, sd, a.model, a.samples) for sd in range(a.seed0, a.seed0 + a.n)]
            with Pool(a.w) as pool:
                res = pool.map(_bench_one, jobs, chunksize=2)
            gr = sum(r[0] for r in res)
            print(f"## {it} value_policy({Path(a.model).name}, samples {a.samples}{', ' + os.environ['VALUE_SET'] if os.environ.get('VALUE_SET') else ''}) seeds{a.seed0}-{a.seed0 + a.n - 1}: "
                  f"大成功 {gr / a.n * 100:.1f}%  本会心 {sum(r[2] for r in res) / a.n:.2f}/局  平均誤差 {sum(r[1] for r in res) / a.n:.1f}  残集中 {sum(r[3] for r in res) / a.n:.0f}")
    elif a.cmd == "trace":
        net = load_net(a.model)
        s = make_item(a.item, random.Random(a.seed))
        while not _terminal(s):
            moves = KF.legal_moves(s)
            vals = value_of_moves(s, net, moves)
            order = np.argsort(-vals)[:5]
            mv = next_move(s, net)
            print(f"{s.temp}℃ 集中 {s.focus} {{{', '.join(f'{n}:{c.v}' + ('★' if c.pinned else '') for n, c in s.cells.items())}}} 今終えると {KF.p_great_now(s):.2f}")
            print("   候補:", "  ".join(f"{moves[i][0]}{''.join(moves[i][1])}={vals[i]:.3f}" for i in order), "→", mv)
            if mv is None:
                break
            s.act(mv[0], mv[1]) if mv[0] != "必殺" else s.act("必殺")
        print("誤差", s.total_error(), "大成功" if is_great(s) else "×")
    else:
        ap.print_help()
