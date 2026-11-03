#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鍛冶の局面 → 評価関数（value_net）の入力。商材に依存しない「まとめた状態認識」（2026-10-27、作者 の案）。

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
AIM_CRIT_TURN_ONLY = __import__("os").environ.get("KAJI_AIM_CRIT_TURN", "0") == "1"   # 集中力変化の少マスで ねらい系を会心ターンに限る（試験）
AIM_CRIT_TURN_MAX_CELLS = 3
AIM_WIDE_ONLY = False  # 広い（平均幅より広い）未確定マスに単体のねらいが打てる時は、狭いマスへの単体ねらいを外す（作者 2026-10-28「狙う場所で差がある」の実験）
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
    return KS.great_of(sim)


def regen_target(sim: Sim):
    """次に 200℃ の倍数で再生が起きた時に戻るマス（範囲外で割合が最大）。無ければ None。"""
    if _trait_name(sim.trait) != "再生":
        return None
    cand = [n for n, c in sim.cells.items() if not (c.lo <= c.v <= c.hi)]
    if not cand:
        return None
    from kaji_sim import modori_key
    return max(cand, key=lambda n: modori_key(sim.cells[n]))


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


NO_TEMP_REVERSAL = True   # 直前の温度だけの手を打ち消す手（冷やし込み→火力上げ、火力上げ→冷やし込み）は候補にしない（集中を捨てる往復の禁止。2026-11-03）


def _last_temp_move(sim):
    """直前の手が温度だけの手（火力上げ／冷やし込み）ならその名前、違えば None"""
    if not getattr(sim, "log", None):
        return None
    last = sim.log[-1]
    return last[0] if last[0] in ("火力上げ", "冷やし込み") else None


def legal_moves(sim):
    """打てる手。本会心済み・はみ出したマスは叩かない（必殺の装填中は会心確定なので本会心済みも Miss で無害＝許す）。
    ねらい系は会心で下限に届く時だけ。非会心の全部の目がはみ出す単体打ちは外す。"""
    out = _legal_moves_raw(sim)
    if NO_TEMP_REVERSAL:
        prev = _last_temp_move(sim)
        if prev is not None:
            ban = "火力上げ" if prev == "冷やし込み" else "冷やし込み"
            out = [mv for mv in out if not str(mv[0]).startswith(ban)]
    return out


def _legal_moves_raw(sim):
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
            if skill == "冷やし込み" and sim.temp <= 300:     # 300 → 0 ℃は終了になるので候補にしない（2026-10-31 作者）
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
                    if skill == "冷やし込み" and t2 <= 0:
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
        groups = {"pair_v": list(sim.pairs_v) + list(getattr(sim, "pairs_v_solo", None) or []), "pair_lr": sim.pairs_lr, "pair_diag": sim.pairs_diag, "quad": sim.quads}[tg]
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
    if AIM_CRIT_TURN_ONLY and not armed and _trait_name(sim.trait) == "集中力変化" and len(cells) <= AIM_CRIT_TURN_MAX_CELLS:
        # 集中力変化の少マス（こて等、誤差 2）: ねらい系は会心率アップの手番（400n+200）だけ（2026-10-31 作者「少ないマスは会心でぴったり」。
        # こて 100 局: 失敗局は会心ターン以外のねらい系が多い 1.39 対 0.96）
        from kaji_sim import is_crit_turn
        if not is_crit_turn(sim.trait, sim.temp):
            out = [mv for mv in out if not (mv[0] in SKILLS and SKILLS[mv[0]]["crit"] > 1)]
    if AIM_WIDE_ONLY and not armed:
        ws = {n: c.hi - c.lo for n, c in cells.items() if not c.pinned and c.v <= c.hi}
        if ws:
            wbar = sum(ws.values()) / len(ws)
            def single_aim(mv):
                return mv[0] in SKILLS and SKILLS[mv[0]]["crit"] > 1 and SKILLS[mv[0]]["targets"] == "cell"
            if any(single_aim(mv) and ws.get(mv[1][0], wbar) > wbar for mv in out):
                out = [mv for mv in out if not (single_aim(mv) and ws.get(mv[1][0], wbar) < wbar)]
    return out


# ---------------------------------------------------------------------------
# 一括の特徴化（2026-10-30、2 手読みの高速化）。featurize(sim) を B 局面まとめて同じ値を返す。
# 1 局面ずつだと小さな numpy 演算 40 回の呼び出しの固定費が 0.23ms/局面を占め、2 手読み 1 決定で 3 万局面を読むと 7 秒かかる。
# 同じ商材（マスの名前・ペア・面・地金が同じ）の局面に限る。featurize と同じ値になることは scratchpad の check_many.py で確認した。
# ---------------------------------------------------------------------------
def _gather(sims, names):
    """[B, n] の lo / hi / v / pinned（1 回の走査で取り出す。Python の属性参照が featurize_many の主な費用）"""
    arr = np.array([[(c.lo, c.hi, c.v, c.pinned) for c in s.cells.values()] for s in sims], dtype=np.float64)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3] > 0.5


def edist_many(lo, hi, v, pinned) -> np.ndarray:
    """cell_error_dist を [B, n] の配列でまとめて: [B, n, 13]（添字=誤差 0..12、12 は 12 以上。本会心済みは 0 に全部）"""
    B, n = lo.shape
    out = np.zeros((B, n, 13))
    w = hi - lo
    in_zone = (v >= lo) & (v <= hi) & ~pinned
    outz = ~in_zone & ~pinned
    # 範囲内: |v-ideal| を 4 で頭打ち
    out[:, :, 0] = np.where(in_zone, 1.0, 0.0)
    for e in (1, 2, 3):
        out[:, :, e] = np.where(in_zone, ((v - e) >= lo).astype(np.float64) + ((v + e) <= hi).astype(np.float64), 0.0)
    out[:, :, 4] = np.where(in_zone, (w + 1) - out[:, :, :4].sum(2), 0.0)
    # 範囲外: e = 距離 + k + 3（k = 0..w）を max(e, 9) にして 12 で頭打ち
    base = np.where(v < lo, lo - v, v - hi) + 3
    out[:, :, 9] = np.where(outz, np.clip(9 - base + 1, 0, w + 1), 0.0)
    for e in (10, 11):
        out[:, :, e] = np.where(outz & (e - base >= 0) & (e - base <= w), 1.0, 0.0)
    out[:, :, 12] = np.where(outz, np.clip(w + 1 - np.maximum(0, 12 - base), 0, w + 1), 0.0)
    out /= (w + 1)[:, :, None]
    out[:, :, 0] = np.where(pinned, 1.0, out[:, :, 0])
    return out


def p_great_arrays(lo, hi, v, pinned, thr) -> np.ndarray:
    """p_great_now を配列で: 誤差分布の畳み込み（12 で打ち切り）の 0..thr の和。thr は [B]"""
    B, n = lo.shape
    # (lo, hi, v, pinned) が同じマスはまとめて 1 回（2 手読みの葉ではマスの状態の種類は数千）
    key = ((lo.astype(np.int64) * 1024 + hi.astype(np.int64)) * 2 + pinned) * 4096 + np.clip(v.astype(np.int64), 0, 4095)
    uk, first, cinv = np.unique(key.ravel(), return_index=True, return_inverse=True)
    edu = edist_many(lo.ravel()[first][None, :], hi.ravel()[first][None, :], v.ravel()[first][None, :], pinned.ravel()[first][None, :])[0]   # [U, 13]
    cinv = cinv.reshape(B, n)
    L = int(min(13, np.max(thr) + 1))                 # 誤差 thr までしか要らないので畳み込みをそこで打ち切る（2026-10-30）
    tot = np.zeros((B, L)); tot[:, 0] = 1.0
    for i in range(n):
        e = edu[cinv[:, i]]
        new = np.zeros((B, L))
        for c in range(L):
            new[:, c:] += tot[:, :L - c] * e[:, c:c + 1]
        tot = new
    cs = np.cumsum(tot, axis=1)
    return cs[np.arange(B), np.clip(thr.astype(np.int64), 0, L - 1)]


def p_great_many(sims) -> np.ndarray:
    """p_great_now を B 局面まとめて（配列計算。1 局面ずつの convolve の約 20 倍速）"""
    if not sims:
        return np.zeros(0)
    names = list(sims[0].cells)
    lo, hi, v, pinned = _gather(sims, names)
    thr = np.array([great_thr(s) for s in sims], dtype=np.float64)
    return p_great_arrays(lo, hi, v, pinned, thr)


class ItemMeta:
    """同じ商材の局面に共通の情報（マスの名前・範囲・中央・ペア・面・地金・判定の閾値）。featurize_arrays / 2 手読みの葉の配列生成で使う"""
    __slots__ = ("names", "idx", "n", "trait_raw", "trait", "lo", "hi", "center", "pairs_v", "quads", "thr", "PM", "Q", "cntq", "regen")

    def __init__(self, s0: Sim):
        self.names = list(s0.cells); self.n = len(self.names)
        self.idx = {nm: i for i, nm in enumerate(self.names)}
        self.trait_raw = s0.trait
        self.trait = _trait_name(s0.trait)
        self.regen = s0.trait in KS.MODORI_TRAITS
        self.lo = np.array([s0.cells[k].lo for k in self.names], dtype=np.float64)
        self.hi = np.array([s0.cells[k].hi for k in self.names], dtype=np.float64)
        self.center = np.array([s0.cells[k].center for k in self.names], dtype=np.float64)
        self.pairs_v = list(s0.pairs_v); self.quads = list(s0.quads or [])
        self.thr = great_thr(s0)
        n = self.n
        self.PM = np.zeros((n, n), dtype=bool)
        for x, y in self.pairs_v:
            self.PM[self.idx[x], self.idx[y]] = True; self.PM[self.idx[y], self.idx[x]] = True
        self.Q = np.zeros((len(self.quads), n), dtype=bool)
        for qi, q in enumerate(self.quads):
            for m in q:
                self.Q[qi, self.idx[m]] = True
        self.cntq = self.Q.sum(0).astype(np.float64)


def arrays_from_sims(meta: "ItemMeta", sims) -> dict:
    """Sim の列 → featurize_arrays の入力（temp / focus / v / pinned / lit / hc / ha / hu / started、長さ B）"""
    B = len(sims)
    arr = np.array([[(c.v, c.pinned) for c in s.cells.values()] for s in sims], dtype=np.float64) if B else np.zeros((0, meta.n, 2))
    return dict(
        temp=np.array([s.temp for s in sims], dtype=np.int64),
        focus=np.array([s.focus for s in sims], dtype=np.float64),
        v=arr[:, :, 0], pinned=arr[:, :, 1] > 0.5,
        lit=np.array([meta.idx.get(s.lit_cell, -1) if s.lit_cell is not None else -1 for s in sims], dtype=np.int64),
        hc=np.array([s.hissatsu_charged for s in sims], dtype=bool),
        ha=np.array([s.hissatsu_armed for s in sims], dtype=bool),
        hu=np.array([s.hissatsu_used for s in sims], dtype=bool),
        started=np.array([s.started for s in sims], dtype=bool),
    )


def concat_arrays(parts: list) -> dict:
    out = {}
    for k in parts[0]:
        out[k] = np.concatenate([q[k] for q in parts])
    return out


def terminal_mask(A: dict) -> np.ndarray:
    """value_policy._terminal と同じ: 0 ℃、または 集中 5 未満で必殺のチャージが無い"""
    return (A["temp"] <= 0) | ((A["focus"] < 5) & ~A["hc"])


def p_great_of_arrays(meta: "ItemMeta", A: dict) -> np.ndarray:
    B = len(A["temp"])
    lo = np.broadcast_to(meta.lo, (B, meta.n)); hi = np.broadcast_to(meta.hi, (B, meta.n))
    return p_great_arrays(lo, hi, A["v"], A["pinned"], np.full(B, meta.thr, dtype=np.float64))


def featurize_many(sims):
    """[featurize(s) for s in sims] と同じ値を一括で返す: (X [B, n, FC], G [B, FG])。sims は同じ商材の局面（空なら (None, None)）"""
    if not sims:
        return None, None
    meta = ItemMeta(sims[0])
    return featurize_arrays(meta, arrays_from_sims(meta, sims))


def _cell_block(meta: "ItemMeta", ci, t, z, v, pinned):
    """マス単位の特徴（列 0〜7, 9 と 14 以降）を 1-D の配列（長さ U）で。ci=マス番号, t=温度, z=倍率（光なら 2 倍）, v, pinned。
    戻り値 (Xu [U, FC], p_over [U, M], p_pin [U, M])"""
    U = len(ci)
    lo = meta.lo[ci]; hi = meta.hi[ci]
    w = hi - lo
    glo = lo - v
    ghi = hi - v
    in_zone = (v >= lo) & (v <= hi)
    over = v > hi
    ut, inv = np.unique(t, return_inverse=True)
    tabs = np.stack([mult_table(int(x)) for x in ut])[inv]                   # [U, M, 7]
    D = np.ceil(tabs * z[:, None, None])
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
    Xu = np.zeros((U, FC))
    Xu[:, BASE_CELL:] = np.stack([p_over, p_in, p_under, p_pin, need], axis=2).reshape(U, -1)
    Xu[:, 0] = pinned
    Xu[:, 1] = in_zone
    Xu[:, 2] = over
    Xu[:, 3] = np.where(in_zone, (v - lo) / np.maximum(1, w), 0.0)
    Xu[:, 4] = np.where(over, np.minimum(v - hi, 20) / 20.0, 0.0)
    Xu[:, 5] = np.minimum(w, 20) / 20.0
    Xu[:, 6] = np.maximum(0, glo) / 300.0
    Xu[:, 7] = (z > 1.0) & (meta.trait == "光")        # 光っているマス（z は 光のマスだけ 2 倍。倍半の倍ターンは全マス同じ z なので別に判定）
    Xu[:, 9] = np.where(pinned, 1.0, np.where(v >= lo, np.clip((v - lo + 1) / (w + 1), 0, 1), 0.0))
    return Xu, p_over, p_pin


def featurize_arrays(meta: "ItemMeta", A: dict):
    """配列表現の B 局面 → (X [B, n, FC], G [B, FG])。featurize(sim) と同じ値（check_many.py で確認）。
    マス単位の計算は (マス, 温度, 光, v, pinned) が同じ行をまとめて 1 回だけ行う（2 手読みの葉 48 万局面でマスの状態は数千通り。2026-10-30）"""
    B = len(A["temp"]); n = meta.n
    trait_raw = meta.trait_raw; trait = meta.trait
    t = A["temp"]; focus = A["focus"]; v = A["v"]; pinned = A["pinned"]; lit = A["lit"]
    ut, inv = np.unique(t, return_inverse=True)
    zt = np.array([turn_multiplier(trait_raw, int(x)) for x in ut])[inv]
    zt_next = np.array([turn_multiplier(trait_raw, max(0, int(x) - 50)) for x in ut])[inv]
    crit = np.array([is_crit_turn(trait_raw, int(x)) for x in ut], dtype=np.float64)[inv]
    cost_u = np.array([cost_of("たたく", trait_raw, int(x)) for x in ut], dtype=np.float64)[inv]
    cost = np.where(A["started"], cost_u, float(KS.SKILLS["たたく"]["cost"]) if trait_raw == "集中力変化" else cost_u)
    lo = meta.lo[None, :]; hi = meta.hi[None, :]
    lit_here = lit[:, None] == np.arange(n)[None, :]
    in_zone = (v >= lo) & (v <= hi)
    over = v > hi
    glo = lo - v
    # --- マス単位の重複を除く: 鍵 = (マス番号, 温度/50, 光, pinned, v) ---
    key = (((np.arange(n)[None, :] * 64 + (t // 50)[:, None]) * 2 + lit_here) * 2 + pinned) * 4096 + np.clip(v.astype(np.int64), 0, 4095)
    uk, first, cinv = np.unique(key.ravel(), return_index=True, return_inverse=True)
    fb = first // n; fi = first % n
    zu = zt[fb] * np.where(lit_here.ravel()[first], 2.0, 1.0)
    Xu, pou, ppu = _cell_block(meta, fi, t[fb], zu, v.ravel()[first], pinned.ravel()[first])
    cinv = cinv.reshape(B, n)
    X = Xu[cinv]                                                     # [B, n, FC]
    j12 = MULTS.index("x1.2"); j20 = MULTS.index("x2.0")
    po12 = pou[:, j12][cinv]; pp12 = ppu[:, j12][cinv]; po20 = pou[:, j20][cinv]   # [B, n]
    bi = np.arange(B)
    if trait == "再生":
        ratio = np.where(~in_zone, v / np.maximum(1.0, meta.center)[None, :], -1.0)
        rt = ratio.argmax(1)
        has_rt = (~in_zone).any(1)
        X[bi[has_rt], rt[has_rt], 8] = 1.0
    PM = meta.PM
    has_p = PM.any(1)
    if has_p.any():
        val = (1.0 - po12) * pp12
        X[:, :, 10] = has_p[None, :].astype(np.float64)
        X[:, :, 11] = np.where(has_p[None, :], np.where(PM[None, :, :], val[:, None, :], -np.inf).max(2), 0.0)
    if len(meta.quads):
        Q = meta.Q; cntq = meta.cntq
        safe12 = ((po12 == 0.0)[:, None, :] | ~Q[None, :, :]).all(2)
        safe20 = ((po20 == 0.0)[:, None, :] | ~Q[None, :, :]).all(2)
        Qf = Q.astype(np.float64)
        X[:, :, 12] = np.where(cntq[None, :] > 0, (safe12.astype(np.float64) @ Qf) / np.maximum(cntq, 1.0)[None, :], 0.0)
        X[:, :, 13] = np.where(cntq[None, :] > 0, (safe20.astype(np.float64) @ Qf) / np.maximum(cntq, 1.0)[None, :], 0.0)
    has_lit = lit >= 0
    G = np.zeros((B, FG))
    G[:, 0] = t / 2000.0
    G[:, 1] = (t % 200) / 200.0
    G[:, 2] = (t % 400) / 400.0
    G[:, 3] = (((t - 50) % 200 == 0) & (t > 50)).astype(np.float64)
    G[:, 4] = (((t - 150) % 200 == 0) & (t > 150)).astype(np.float64)
    G[:, 5] = zt / 2.0
    G[:, 6] = zt_next / 2.0
    G[:, 7] = crit
    G[:, 8] = focus / 253.0
    G[:, 9] = A["hc"]; G[:, 10] = A["ha"]; G[:, 11] = A["hu"]
    ti = TRAITS.index(trait) if trait in TRAITS else 0
    G[:, 12 + ti] = 1.0
    G[:, 17] = n / 9.0
    G[:, 18] = (~pinned).sum(1) / 9.0
    G[:, 19] = ((~pinned) & in_zone).sum(1) / 9.0
    G[:, 20] = over.sum(1) / 9.0
    G[:, 21] = meta.thr / 11.0
    G[:, 22] = np.where(~pinned, np.maximum(0, glo), 0.0).sum(1) / 600.0
    G[:, 23] = cost / 5.0
    G[:, 24] = p_great_arrays(np.broadcast_to(meta.lo, (B, n)), np.broadcast_to(meta.hi, (B, n)), v, pinned, np.full(B, meta.thr, dtype=np.float64))
    G[:, 25] = has_lit.astype(np.float64)
    return X, G


def pad_many(X: np.ndarray):
    """[B, n, FC] → ([B, MAX_CELLS, FC], mask [B, MAX_CELLS])"""
    B, n = X.shape[0], X.shape[1]
    out = np.zeros((B, MAX_CELLS, FC)); out[:, :n] = X
    m = np.zeros((B, MAX_CELLS)); m[:, :n] = 1.0
    return out, m


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
