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
import kaji_feat as KF
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
SUPER = {"超あまつゆのいと", "超かがやきの樹液", "超ようせいのひだね"}

_NET = None
_MODEL_PATH = None


def _apply_env():
    """環境変数 VALUE_SET="DEPTH=2,ES.ALLOW_LIGHT=True"（ES./KF./FL. で他モジュールの定数も上書き）"""
    for kv in filter(None, os.environ.get("VALUE_SET", "").split(",")):
        k, v = kv.split("=", 1)
        k = k.strip()
        if "." in k:
            mod, name = k.split(".", 1)
            setattr({"ES": ES, "KF": KF, "FL": FL}[mod], name, eval(v))
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
    return s.focus < 5 and not s.hissatsu_charged


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


def cell_branches(sim: Sim, n: str, mult_key: str, aim: bool):
    """1 マスに 1 打を当てた時の結果 [(v, pinned, prob)]（出目 7 通り × 非会心/本会心/偽会心/Miss、理想値は範囲で一様）"""
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


def _after_state(sim: Sim, assign, cost: int, t2: int, k: int) -> Sim:
    """assign={マス: (v, pinned)} を当てはめた後の局面（集中・温度・再生（戻り 14 固定）・光（候補を順に割り当て）を反映）"""
    s = sim.clone(random.Random(0), hide_ideal=True)
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
            tgt = max(cand, key=lambda m: s.cells[m].v / max(1, s.cells[m].center))
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


def value_of_moves(sim: Sim, net, moves, samples: int | None = None):
    """手ごとの期待価値。1〜2 マスの技は分岐を確率つきで列挙（EXACT_SINGLE / EXACT_PAIR）、他は信念つきクローン samples 本の平均。終局した分岐は厳密な大成功確率。"""
    samples = SAMPLES if samples is None else samples
    seeds = _seeds(sim, samples)
    Xs, Ms, Gs = [], [], []
    slots = []          # (手の番号, 行番号 or None, 厳密値, 重み)
    for mi, mv in enumerate(moves):
        tg = KS.SKILLS[mv[0]]["targets"] if mv[0] in KS.SKILLS else None
        exact = (EXACT_SINGLE and tg == "cell" and len(mv[1]) == 1) or (EXACT_PAIR and tg in ("pair_v", "pair_lr", "pair_diag") and len(mv[1]) == 2)
        if exact:
            outs = exact_outcomes(sim, mv)
            if outs:
                for s, pr in outs:
                    if _terminal(s):
                        slots.append((mi, None, KF.p_great_now(s), pr))
                        continue
                    X, g = KF.featurize(s)
                    Xp, m = KF.pad(X)
                    slots.append((mi, len(Xs), None, pr))
                    Xs.append(Xp); Ms.append(m); Gs.append(g)
                continue
        for sd in seeds:
            s = FL._belief_clone(sim, random.Random(sd))
            try:
                apply_move(s, mv)
            except AssertionError:
                slots.append((mi, None, 0.0, 1.0 / samples))
                continue
            if _terminal(s):
                slots.append((mi, None, KF.p_great_now(s), 1.0 / samples))
                continue
            X, g = KF.featurize(s)
            Xp, m = KF.pad(X)
            slots.append((mi, len(Xs), None, 1.0 / samples))
            Xs.append(Xp); Ms.append(m); Gs.append(g)
    if Xs:
        p = net.forward(np.stack(Xs), np.stack(Ms), np.stack(Gs))
    vals = np.zeros(len(moves))
    for mi, row, ex, wt in slots:
        vals[mi] += (ex if row is None else p[row]) * wt
    return vals


def value_depth2(sim: Sim, net, moves, samples: int):
    """各手の分岐先で、さらに最善手（1 手読み・SAMPLES2 本）を取った値の平均。分岐先が終局なら厳密値"""
    seeds = _seeds(sim, samples)
    vals = np.zeros(len(moves))
    for mi, mv in enumerate(moves):
        tot = 0.0
        for sd in seeds:
            s = FL._belief_clone(sim, random.Random(sd))
            try:
                apply_move(s, mv)
            except AssertionError:
                continue
            if _terminal(s):
                tot += KF.p_great_now(s); continue
            m2 = KF.legal_moves(s)
            stop = KF.p_great_now(s)
            if not m2:
                tot += stop; continue
            v2 = value_of_moves(s, net, m2, SAMPLES2)
            tot += max(stop, float(v2.max()))
        vals[mi] = tot / samples
    return vals


def next_move(sim: Sim, net=None, samples: int | None = None):
    samples = SAMPLES if samples is None else samples
    net = net or _NET
    if USE_SOLVER and ES.applicable(sim):
        mv, _, _ = ES.solve(sim)
        if mv != "toobig":
            return mv
    moves = KF.legal_moves(sim)
    if not moves:
        return None
    stop_v = KF.p_great_now(sim)
    vals = value_of_moves(sim, net, moves, samples)
    if DEPTH >= 2 and len(moves) > 1:
        top = [int(i) for i in np.argsort(-vals)[:TOPK]]
        v2 = value_depth2(sim, net, [moves[i] for i in top], samples)
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
            rows.append((Xp, m, g, raw_state(sim)))
        sim.act(mv[0], mv[1]) if mv[0] != "必殺" else sim.act("必殺")
    y = 1.0 if is_great(sim) else 0.0
    return rows, y


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
        Xs, Ms, Gs, ys, Rs = [], [], [], [], []
        greats = 0
        for rows, y in res:
            greats += y
            for Xp, m, g_, r_ in rows:
                Xs.append(Xp); Ms.append(m); Gs.append(g_); ys.append(y); Rs.append(r_)
        Path(a.o).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.o, X=np.array(Xs, dtype=np.float32), M=np.array(Ms, dtype=np.float32), G=np.array(Gs, dtype=np.float32), y=np.array(ys, dtype=np.float32), R=np.array(Rs, dtype=np.float32))
        print(f"{a.o}: {len(jobs)} 局 → {len(ys)} 局面  大成功 {greats / len(jobs) * 100:.1f}%  （{' '.join(a.items)}, eps {a.eps}, {'自己対局 ' + a.model if a.model else '先生'}）")
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
