#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鍛冶の局面 → 評価関数（value_net）の入力。商材に依存しない「まとめた状態認識」（2026-10-27、mori1 の案）。

数値をそのまま見せず、各マスを「今の温度の出目で何が起きるか」で表す:
  倍率ごと（x0.5/x0.8/x1.0/x1.2/x2.0/x2.5/x3.0）に
    非会心ではみ出す目の割合 ／ 範囲に入る目の割合 ／ 手前に落ちる目の割合 ／
    会心なら本会心（理想で止まる）になる確率（理想値は範囲で一様）／ 下限まで何打か
  （残り 10 と 11 は同じ出目の括りに入るので同じ特徴になる）
盤面全体は 温度・集中・必殺・地金・次の 200℃ の倍数までの距離・今すぐ終えた時の大成功確率（厳密）など。

featurize(sim) -> (cell_feats [n_cells, FC], glob [FG])
p_great_now(sim) -> 今の盤面で終えた時の大成功確率（信念: 理想値は範囲で一様、はみ出しは 9 扱い）
legal_moves(sim) -> 打てる手の一覧 [(skill, targets)]（本会心済み・はみ出したマスは叩かない。ねらいは会心で届く時だけ）
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kaji_sim as KS
from kaji_sim import Sim, SKILLS, ASSUMPTIONS, dmg_candidates, turn_multiplier, cost_of, is_crit_turn

MULTS = ["x0.5", "x0.8", "x1.0", "x1.2", "x2.0", "x2.5", "x3.0"]
TRAITS = ["", "倍半", "光", "再生", "集中力変化"]
PER_MULT = 5
BASE_CELL = 14
FC = BASE_CELL + PER_MULT * len(MULTS)
FG = 26
MAX_CELLS = 9
AIM_WIDE_ONLY = False  # 広い（平均幅より広い）未確定マスに単体のねらいが打てる時は、狭いマスへの単体ねらいを外す（mori1 2026-10-28「狙う場所で差がある」の実験）
MACRO_TEMP = True   # legal_moves に 火力上げ×k / 冷やし込み×k（k=2..4）のまとめ手を入れる

_DICE = {}


def dice(temp: int, mk: str, z: float):
    key = (temp, mk, z)
    d = _DICE.get(key)
    if d is None:
        d = np.array([math.ceil(x * z) for x in dmg_candidates(temp, mk)], dtype=np.float64)
        _DICE[key] = d
    return d


def _trait_name(t: str) -> str:
    return "再生" if t in KS.MODORI_TRAITS else t


def great_thr(sim: Sim) -> int:
    return KS.GRADE_TABLE[len(sim.cells)][0]


def regen_target(sim: Sim):
    """次に 200℃ の倍数で再生が起きた時に戻るマス（範囲外で割合が最大）。無ければ None。"""
    if _trait_name(sim.trait) != "再生":
        return None
    cand = [n for n, c in sim.cells.items() if not (c.lo <= c.v <= c.hi)]
    if not cand:
        return None
    return max(cand, key=lambda n: sim.cells[n].v / max(1, sim.cells[n].center))


def cell_error_dist(c) -> np.ndarray:
    """マスの誤差の分布（理想値は [lo,hi] で一様。本会心済みは 0）。添字=誤差（0..12、12 は 12 以上）。"""
    out = np.zeros(13)
    if c.pinned:
        out[0] = 1.0
        return out
    w = c.hi - c.lo + 1
    for ideal in range(c.lo, c.hi + 1):
        e = abs(c.v - ideal)
        if c.lo <= c.v <= c.hi:
            e = min(e, 4)
        else:
            e = max(e + 3, 9)
        out[min(e, 12)] += 1.0 / w
    return out


def p_great_now(sim: Sim) -> float:
    thr = great_thr(sim)
    tot = np.zeros(1); tot[0] = 1.0
    for c in sim.cells.values():
        tot = np.convolve(tot, _edist(c))
    return float(tot[:thr + 1].sum())


_TABLE = {}


def mult_table(temp: int) -> np.ndarray:
    """[len(MULTS), 7] の素の出目（倍率を掛ける前）"""
    tb = _TABLE.get(temp)
    if tb is None:
        tb = np.array([dmg_candidates(temp, mk) for mk in MULTS], dtype=np.float64)
        _TABLE[temp] = tb
    return tb


_ED = {}


def _edist(c) -> np.ndarray:
    key = (c.lo, c.hi, c.v, c.pinned)
    d = _ED.get(key)
    if d is None:
        d = cell_error_dist(c)
        _ED[key] = d
    return d


def featurize(sim: Sim):
    cells = sim.cells
    names = list(cells)
    t = sim.temp
    trait = _trait_name(sim.trait)
    zt = turn_multiplier(sim.trait, t)
    crit_turn = is_crit_turn(sim.trait, t)
    rt = regen_target(sim)
    n = len(names)
    X = np.zeros((n, FC))
    vpart = {k: [] for k in names}
    for a, b in sim.pairs_v:
        vpart[a].append(b); vpart[b].append(a)
    lo = np.array([cells[k].lo for k in names], dtype=np.float64)
    hi = np.array([cells[k].hi for k in names], dtype=np.float64)
    v = np.array([cells[k].v for k in names], dtype=np.float64)
    pinned = np.array([cells[k].pinned for k in names])
    z = np.array([zt * (2.0 if k == sim.lit_cell else 1.0) for k in names])
    w = hi - lo
    glo = lo - v
    ghi = hi - v
    in_zone = (v >= lo) & (v <= hi)
    over = v > hi
    # 倍率ごとの出目 D[i, j, k]
    D = np.ceil(mult_table(t)[None, :, :] * z[:, None, None])
    p_over = (D > ghi[:, None, None]).mean(2)
    p_in = ((D >= glo[:, None, None]) & (D <= ghi[:, None, None])).mean(2)
    p_under = (D < glo[:, None, None]).mean(2)
    lo_i = np.maximum(lo, v + 1)
    cnt = np.clip(np.minimum(hi[:, None, None], v[:, None, None] + 2 * D) - lo_i[:, None, None] + 1, 0, None)
    p_pin = (cnt / (w + 1)[:, None, None]).mean(2)
    p_pin = np.where((v < hi)[:, None], p_pin, 0.0)
    need = np.minimum(6.0, np.maximum(0.0, glo)[:, None] / np.maximum(1.0, D.mean(2))) / 6.0
    need = np.where((glo > 0)[:, None], need, 0.0)
    dead = (pinned | over)[:, None]
    p_over = np.where(dead, 1.0, p_over); p_in = np.where(dead, 0.0, p_in); p_under = np.where(dead, 0.0, p_under)
    p_pin = np.where(dead, 0.0, p_pin); need = np.where(dead, 0.0, need)
    X[:, BASE_CELL:] = np.stack([p_over, p_in, p_under, p_pin, need], axis=2).reshape(n, -1)
    # P(ideal <= v) = 会心が Miss になる確率
    p_miss = np.where(pinned, 1.0, np.where(v >= lo, np.clip((v - lo + 1) / (w + 1), 0, 1), 0.0))
    X[:, 0] = pinned
    X[:, 1] = in_zone
    X[:, 2] = over
    X[:, 3] = np.where(in_zone, (v - lo) / np.maximum(1, w), 0.0)
    X[:, 4] = np.where(over, np.minimum(v - hi, 20) / 20.0, 0.0)
    X[:, 5] = np.minimum(w, 20) / 20.0
    X[:, 6] = np.maximum(0, glo) / 300.0
    X[:, 9] = p_miss
    idx = {nm: i for i, nm in enumerate(names)}
    if sim.lit_cell in idx:
        X[idx[sim.lit_cell], 7] = 1.0
    if rt is not None:
        X[idx[rt], 8] = 1.0
    j12 = MULTS.index("x1.2"); j20 = MULTS.index("x2.0")
    for nm in names:
        i = idx[nm]
        if vpart[nm]:
            X[i, 10] = 1.0
            X[i, 11] = max((1.0 - p_over[idx[p], j12]) * p_pin[idx[p], j12] for p in vpart[nm])
        qs = [q for q in sim.quads if nm in q]
        if qs:
            X[i, 12] = sum(1.0 for q in qs if all(p_over[idx[m], j12] == 0.0 for m in q)) / len(qs)
            X[i, 13] = sum(1.0 for q in qs if all(p_over[idx[m], j20] == 0.0 for m in q)) / len(qs)
    live = [c for c in cells.values() if not c.pinned]
    g = np.zeros(FG)
    g[0] = t / 2000.0
    g[1] = (t % 200) / 200.0
    g[2] = (t % 400) / 400.0
    g[3] = 1.0 if (t - 50) % 200 == 0 and t > 50 else 0.0      # 次の一打で 200℃ の倍数（再生/光/倍半のターンが来る）
    g[4] = 1.0 if (t - 150) % 200 == 0 and t > 150 else 0.0    # 熱風で 200℃ の倍数
    g[5] = zt / 2.0
    g[6] = turn_multiplier(sim.trait, max(0, t - 50)) / 2.0
    g[7] = 1.0 if crit_turn else 0.0
    g[8] = sim.focus / 253.0
    g[9] = 1.0 if sim.hissatsu_charged else 0.0
    g[10] = 1.0 if sim.hissatsu_armed else 0.0
    g[11] = 1.0 if sim.hissatsu_used else 0.0
    ti = TRAITS.index(trait) if trait in TRAITS else 0
    g[12 + ti] = 1.0                                             # 12..16 地金
    g[17] = n / 9.0
    g[18] = len(live) / 9.0
    g[19] = float(((~pinned) & in_zone).sum()) / 9.0
    g[20] = float(over.sum()) / 9.0
    g[21] = great_thr(sim) / 11.0
    g[22] = float(np.maximum(0, glo)[~pinned].sum()) / 600.0
    g[23] = cost_of("たたく", sim.trait, t, first=not sim.started) / 5.0
    g[24] = p_great_now(sim)
    g[25] = 1.0 if sim.lit_cell else 0.0
    return X, g


def pad(X: np.ndarray):
    """[n, FC] → [MAX_CELLS, FC] と mask [MAX_CELLS]"""
    n = X.shape[0]
    out = np.zeros((MAX_CELLS, FC)); out[:n] = X
    m = np.zeros(MAX_CELLS); m[:n] = 1.0
    return out, m


def legal_moves(sim: Sim):
    """打てる手。本会心済み・はみ出したマスは叩かない（必殺の装填中は会心確定なので本会心済みも Miss で無害＝許す）。
    ねらい系は会心で下限に届く時だけ。非会心の全部の目がはみ出す単体打ちは外す。"""
    out = []
    cells = sim.cells
    zt = turn_multiplier(sim.trait, sim.temp)
    armed = sim.hissatsu_armed

    def ok_cell(n):
        c = cells[n]
        if c.v > c.hi:
            return False
        return armed or not c.pinned

    def z(n):
        return zt * (2.0 if n == sim.lit_cell else 1.0)

    for skill, s in SKILLS.items():
        if skill == "必殺":
            if sim.hissatsu_charged and not sim.hissatsu_used:
                out.append(("必殺", ()))
            continue
        if not sim.can_use(skill):
            continue
        tg = s["targets"]
        if tg == "none":
            if skill == "火力上げ" and sim.temp >= 2000:
                continue
            if skill == "冷やし込み" and sim.temp < 300:
                continue
            out.append((skill, ()))
            # まとめ手（火力上げ×k / 冷やし込み×k）: 温度を目当ての所へ持っていく「投資」を 1 手読みでも比べられるようにする（2026-10-27）。
            # 返す手は最初の 1 回ぶんで、評価の時だけ k 回続けて打つ（value_policy が解釈）
            if MACRO_TEMP:
                step = s["dtemp"]; cost = cost_of(skill, sim.trait, sim.temp, first=not sim.started)
                for k in range(2, 5):
                    t2 = sim.temp + step * k
                    if skill == "火力上げ" and t2 > 2200:
                        break
                    if skill == "冷やし込み" and t2 < 0:
                        break
                    if sim.focus < cost * k + 5:
                        break
                    out.append((f"{skill}×{k}", ()))
            continue
        if tg == "rand4":
            if any(c.pinned for c in cells.values()) and not armed:
                continue
            out.append((skill, ()))
            continue
        aim = s["crit"] > 1
        if tg == "cell":
            for n in cells:
                if not ok_cell(n):
                    continue
                c = cells[n]
                d = dice(sim.temp, s["mult"], z(n))
                glo = c.lo - c.v; ghi = c.hi - c.v
                if aim and not armed and 2 * d[-1] < glo:
                    continue           # 会心でも下限に届かない
                if not armed and d[0] > ghi:
                    continue           # 全部の目がはみ出す
                out.append((skill, (n,)))
            continue
        groups = {"pair_v": sim.pairs_v, "pair_lr": sim.pairs_lr, "pair_diag": sim.pairs_diag, "quad": sim.quads}[tg]
        for grp in groups:
            if not all(ok_cell(n) for n in grp):
                continue
            bad = False
            for n in grp:
                c = cells[n]
                d = dice(sim.temp, s["mult"], z(n))
                if not armed and d[0] > c.hi - c.v:
                    bad = True; break
                if aim and not armed and 2 * d[-1] < c.lo - c.v:
                    bad = True; break
            if not bad:
                out.append((skill, tuple(grp)))
    if AIM_WIDE_ONLY and not armed:
        ws = {n: c.hi - c.lo for n, c in cells.items() if not c.pinned and c.v <= c.hi}
        if ws:
            wbar = sum(ws.values()) / len(ws)
            def single_aim(mv):
                return mv[0] in SKILLS and SKILLS[mv[0]]["crit"] > 1 and SKILLS[mv[0]]["targets"] == "cell"
            if any(single_aim(mv) and ws.get(mv[1][0], wbar) > wbar for mv in out):
                out = [mv for mv in out if not (single_aim(mv) and ws.get(mv[1][0], wbar) < wbar)]
    return out


if __name__ == "__main__":
    import random
    s = KS.make_item("虹色のオーブ", random.Random(0))
    X, g = featurize(s)
    print(X.shape, g.shape, "p_great_now", p_great_now(s))
    print(len(legal_moves(s)), legal_moves(s)[:8])
    import time
    t0 = time.perf_counter()
    for _ in range(200):
        featurize(s)
    print("featurize ms", (time.perf_counter() - t0) / 200 * 1000)
