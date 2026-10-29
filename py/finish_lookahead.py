#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""終盤先読み（超あまつゆ）: finish局面で候補手ごとに終局ロールアウトし、
大成功確率が最大の手を選ぶ。run_amatsuyu.policy の finish() を置き換える。

- 隠し理想値は「中央±2の一様分布」という信念でロールアウトごとにサンプル（カンニング無し）
- 乱数は局面ハッシュで固定 → 同じ局面なら常に同じ手（アドバイザーと一致）
"""
import math
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kaji_sim import ASSUMPTIONS, SKILLS, Sim, dmg_candidates, turn_multiplier
import kaji_sim as KS
import run_amatsuyu as R
import endgame_solver as ES

GREAT = ASSUMPTIONS["great_success_max_err"]


def great_of(sim: Sim) -> int:
    """大成功の誤差上限（マス数で変わる。6マス=7）。"""
    return KS.GRADE_TABLE.get(len(sim.cells), (GREAT,))[0]
ADAPTIVE = False      # 上位手が僅差の時にロールアウトを3倍にして再評価
ADAPTIVE_GAP = 0.06
ADAPTIVE_TOP = 3
USE_SOLVER = True     # 終盤(残集中<=ES.ENDGAME_FOCUS)は厳密解ソルバー（あまつゆ+2.7pt/樹液+4.3pt、2026-09-04）
PHASE_MIN_TEMP = 800  # この温度以上はphase式、未満は先読み（実験用に可変）
REGEN_CAND = [0]
REGEN_OVERSHOOT = False  # 再生の地金で「はみ出して戻す」を候補に入れる（2026-10-20 実験）
AIM_MAX_TEMP = None      # ねらい系を使う温度の上限（None=制限なし）
AIM_WIDE_BONUS = 0.0     # 先読みの評価に、ねらい系の対象マスの「幅の広さ」(幅−平均幅)/平均幅 × この値を足す（広いマスの本会心は残りの誤差を減らす。mori1 2026-10-28「狙う場所で差がある」）
LOWTEMP_AIM_ONLY = None  # 温度がこの値以下の時、ねらい打ちが候補にあるマスへの たたく/てかげん/2倍 を候補から外す（低温の寄せは小さい出目のねらいで＝ソルティーさんの型。2026-10-28 実験、None=無効）
AIM_WIDE_ONLY = False    # 広い（平均幅より広い）未確定マスに単体のねらいが打てる時は、狭いマスへの単体ねらいを候補から外す
AIM_NEEDS_REACH = True  # ねらい系は会心で理想に届く時だけ候補にする（2026-10-12）
AIM_RESET = False        # 再生の地金: 範囲内（下限 −4 まで）のマスへ、打った後に 200℃ の倍数になる時のねらい（全部の目がはみ出し ≤11、再生で戻る）を候補に（アメリーさん型、2026-10-26）
AIM_RESCUE = False       # 再生の地金: 非会心の大きい目が上限を 1〜AIM_RESCUE_MAX_OVER 超えるねらい系も候補に入れる（はみ出しは次の 200℃ の倍数の再生 −12〜16 で救える。アメリーさん 50 局で再生 159 回のうち 63 回がはみ出し中のマスに来て 27 回が範囲内へ。2026-10-26）
AIM_RESCUE_MAX_OVER = 8  # 許すはみ出しの最大（再生の最小 12 で上限以下に戻る量）
AIM_RESCUE_MIN_TEMP = 250  # これより低い温度では次の 200℃ の倍数が来ないので許さない
SINGLE = ["たたく", "てかげん打ち", "2倍打ち", "3倍打ち", "熱風おろし", "ねらい打ち", "弱ねらい打ち"]


def _state_key(sim: Sim):
    return (tuple((n, c.v) for n, c in sorted(sim.cells.items())), sim.temp, sim.focus,
            sim.hissatsu_charged, sim.hissatsu_used)


def candidates(sim: Sim):
    """安全（ゾーン上限を超えない）な候補手を列挙。"""
    out = []
    z = turn_multiplier(sim.trait, sim.temp)
    live = [n for n, c in sim.cells.items() if c.r > 0 and c.v <= c.hi]

    def zc(n):  # 光ったマスはダメージ2倍（hit_one と同じ）
        return z * (2.0 if n == sim.lit_cell else 1.0)

    reserved = R.hissatsu_reserved(sim)   # 必殺の面で確定するマスにはねらい系を使わない（mori1 2026-09-06）
    for skill in SINGLE:
        if not sim.can_use(skill):
            continue
        cand = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
        aim = SKILLS[skill]["crit"] > 1
        for n in live:
            c = sim.cells[n]
            if aim and n in reserved:
                continue
            if aim and AIM_NEEDS_REACH and 2 * math.ceil(cand[-1] * zc(n)) < c.r:
                continue  # 会心しても理想に届かないねらいは候補にしない
            if aim and AIM_MAX_TEMP is not None and sim.temp > AIM_MAX_TEMP:
                continue  # 高温ではねらわない（ソルティーさんの樹液: ねらい系は 650℃以下。2026-10-20 実験）（mori1 2026-10-12: 届かないAへの弱ねらい推奨の指摘）
            if math.ceil(cand[-1] * zc(n)) <= c.hi - c.v:
                out.append((skill, (n,)))
            elif aim and AIM_RESCUE and sim.trait in KS.MODORI_TRAITS and sim.temp >= AIM_RESCUE_MIN_TEMP                     and 1 <= math.ceil(cand[-1] * zc(n)) - (c.hi - c.v) <= AIM_RESCUE_MAX_OVER:
                out.append((skill, (n,)))  # はみ出す目があっても再生で救える範囲なら候補（採否はロールアウト）
            elif n == sim.lit_cell and SKILLS[skill]["crit"] > 1 and c.r >= 4:
                import policy_hidane as _PH
                if _PH.light_aim_ok(sim, c, SKILLS[skill]["mult"]):
                    out.append((skill, (n,)))  # 光ったマスへのねらい: はみ出す目 2/7 まで許容（mori1）
    # 再生（戻り）の地金: わざと範囲を 7〜12 だけ超えて、次の 200℃ の倍数で 12〜16 戻るのを受ける（GJさん 14局 1100℃ F 211→235→1000℃で 220）。
    # 戻るのは「範囲外で割合がいちばん進んだマス」なので、超えたマスが必ず戻る。戻りが 1〜2 手以内に来る温度（200n+50/100）の時だけ候補に入れる
    if REGEN_OVERSHOOT and sim.trait in KS.MODORI_TRAITS and sim.temp % 200 in (50, 100):
        lo_k, hi_k = ASSUMPTIONS["modori_amount"]
        for skill in ("たたく", "てかげん打ち", "2倍打ち", "熱風おろし"):
            if not sim.can_use(skill):
                continue
            if skill == "熱風おろし" and sim.temp % 200 != 100:
                continue   # 熱風は −150: 200n+100 からなら 200(n−1)+150… 倍数に乗らない → 使わない
            cand = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
            for n in live:
                c = sim.cells[n]
                if c.pinned or c.in_zone:
                    continue
                kmin = c.v + math.ceil(cand[0] * zc(n)) - c.hi
                kmax = c.v + math.ceil(cand[-1] * zc(n)) - c.hi
                if kmin >= 1 and kmax <= hi_k and kmax - (c.hi - c.lo) <= lo_k:   # 全目が範囲を超え、戻り最小でも上限以下・戻り最大でも下限以上の目が残る
                    out.append((skill, (n,)))
                    REGEN_CAND[0] += 1
    if AIM_RESET and sim.trait in KS.MODORI_TRAITS and sim.can_use("ねらい打ち") and (sim.temp - 50) % 200 == 0 and sim.temp >= 250:
        cand = dmg_candidates(sim.temp, "x1.0")
        for n, c in sim.cells.items():
            if c.pinned or not (c.lo - 4 <= c.v <= c.hi) or n in reserved:
                continue
            dmin = math.ceil(cand[0] * zc(n)); dmax = math.ceil(cand[-1] * zc(n))
            if c.v + dmin <= c.hi or c.v + dmax - c.hi > 11:
                continue
            ratio = (c.v + dmin) / c.center
            if any(sim.cells[m].v / sim.cells[m].center >= ratio for m in sim.cells if m != n and not (sim.cells[m].lo <= sim.cells[m].v <= sim.cells[m].hi)):
                continue
            out.append(("ねらい打ち", (n,)))
    if sim.can_use("上下ねらい打ち"):
        cand = dmg_candidates(sim.temp, "x1.2")
        for a, b in sim.pairs_v:
            ca, cb = sim.cells[a], sim.cells[b]
            if reserved & {a, b}:
                continue
            if AIM_NEEDS_REACH and (2 * math.ceil(cand[-1] * zc(a)) < ca.r or 2 * math.ceil(cand[-1] * zc(b)) < cb.r):
                continue
            if AIM_MAX_TEMP is not None and sim.temp > AIM_MAX_TEMP:
                continue
            if ca.r > 0 and cb.r > 0 and math.ceil(cand[-1] * zc(a)) <= ca.hi - ca.v                     and math.ceil(cand[-1] * zc(b)) <= cb.hi - cb.v:
                out.append(("上下ねらい打ち", (a, b)))
            elif AIM_RESCUE and sim.trait in KS.MODORI_TRAITS and sim.temp >= AIM_RESCUE_MIN_TEMP and ca.r > 0 and cb.r > 0                     and max(math.ceil(cand[-1] * zc(a)) - (ca.hi - ca.v), math.ceil(cand[-1] * zc(b)) - (cb.hi - cb.v)) <= AIM_RESCUE_MAX_OVER                     and min(math.ceil(cand[-1] * zc(a)) - (ca.hi - ca.v), math.ceil(cand[-1] * zc(b)) - (cb.hi - cb.v)) <= 0:
                out.append(("上下ねらい打ち", (a, b)))   # 片方だけ少しはみ出しうる上下ねらい（再生で救える範囲）
    # ペア/面技（非ねらい）: 安い集中で複数マスを同時に運べる（別シード1000局で+2.6pt）
    for skill, pools in (("上下打ち", sim.pairs_v), ("左右打ち", sim.pairs_lr),
                         ("ななめ打ち", sim.pairs_diag), ("4連打ち", sim.quads), ("超4連打ち", sim.quads)):
        if not sim.can_use(skill):
            continue
        cand = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
        for g in pools:
            if all(sim.cells[n].r > 0 and math.ceil(cand[-1] * zc(n)) <= sim.cells[n].hi - sim.cells[n].v
                   for n in g):
                out.append((skill, tuple(g)))
    if sim.temp >= 350 and sim.can_use("冷やし込み"):
        out.append(("冷やし込み", ()))
    # 火力上げは400℃以下のみ（+300しても800未満 → 「温度>=800は中盤」の局面判定を壊さない）
    if sim.temp <= 400 and sim.can_use("火力上げ") and any(not c.in_zone for c in sim.cells.values()):
        out.append(("火力上げ", ()))
    elif sim.trait == "光" and sim.temp <= 1700 and (sim.temp + 300) % 200 == 0             and sim.can_use("火力上げ") and any(not c.in_zone for c in sim.cells.values()):
        out.append(("火力上げ", ()))  # 光: 200の倍数に乗せて発光を誘う（mori1「集中に余裕があれば火力上げ/下げで光らせて狙う」）
    if sim.hissatsu_charged and R.hissatsu_target(sim) is not None:
        out.append(("必殺", ()))  # 4マス本会心ルールを満たす時のみ
    return out


def _belief_clone(sim: Sim, rng: random.Random) -> Sim:
    """隠し理想値を信念（ASSUMPTIONS["ideal_spread"]）で引き直したクローン。"""
    s = sim.clone(rng, hide_ideal=True)
    for c in s.cells.values():
        if c.pinned:
            continue  # 本会心済み: 理想値=現在値で確定
        if KS.USE_FAKE_INFO:
            ks = [k for b, k in enumerate(KS.K_OFFSETS) if c.kmask >> b & 1] or [0]
            c.ideal_true = max(c.lo, min(c.hi, c.center + rng.choice(ks)))
        else:
            c.ideal_true = KS.sample_ideal(c.lo, c.hi, rng, c.ideal)
    return s


def _rollout_value(s: Sim, rollout_policy=None) -> float:
    """ロールアウト方針: None=run_amatsuyu.finish。商材ごとに差し替え可（ひだねは hidane_step ループ）"""
    if rollout_policy is None:
        R.finish(s)
    else:
        rollout_policy(s)
    err = s.total_error()
    return (1.0 if err <= great_of(s) else 0.0) - err / 1000.0


def choose(sim: Sim, rollouts: int = 12, rollout_policy=None):
    """最良手を返す。None = 打たずに終える。rollout_policy=None なら run_amatsuyu.finish。"""
    if sim.hissatsu_armed:  # 会心確定が装填中 → 必殺対象を撃つ（phase/finish共通の規則）
        tg = R.hissatsu_target(sim, focus_bonus=0)
        if tg and sim.can_use(tg[0]):
            return tg
    if USE_SOLVER and ES.applicable(sim):
        mv, _, _ = ES.solve(sim)
        if mv != "toobig":
            return mv  # 厳密解（None=しあげる）
    cands = candidates(sim)
    # 現行finishの第一手を必ず候補に含める（退化防止）
    probe = sim.clone(random.Random(0), hide_ideal=True)
    n0 = len(probe.log)
    if rollout_policy is None:
        R.finish(probe)
    else:
        rollout_policy(probe)
    if len(probe.log) > n0:
        hm = (probe.log[n0][0], tuple(probe.log[n0][1]))
        if hm not in cands:
            cands.append(hm)
    if not cands:
        return None
    # プロセス間でも同じ手になるよう安定ハッシュ（hash()は文字列でプロセス毎に変わる）
    base = random.Random(zlib.crc32(repr(_state_key(sim)).encode("utf-8")))
    seeds = [base.randrange(1 << 30) for _ in range(rollouts)]
    best, best_v = None, None
    # 「何もしない」= 今の状態で終える（隠し理想値の信念で期待値化）
    stop_v = 0.0
    for sd in seeds:
        s = _belief_clone(sim, random.Random(sd))
        err = s.total_error()
        stop_v += (1.0 if err <= great_of(s) else 0.0) - err / 1000.0
    stop_v /= rollouts
    def evaluate(mv, seeds_):
        tot = 0.0
        for sd in seeds_:
            s = _belief_clone(sim, random.Random(sd))
            if mv[0] == "必殺":
                s.act("必殺")
            else:
                s.act(mv[0], mv[1])
            tot += _rollout_value(s, rollout_policy)
        return tot / len(seeds_)
    if LOWTEMP_AIM_ONLY is not None and sim.temp <= LOWTEMP_AIM_ONLY:
        aimed = {mv[1][0] for mv in cands if mv[0] == "ねらい打ち" and len(mv[1]) == 1}
        if aimed:
            cands = [mv for mv in cands if not (mv[0] in ("たたく", "てかげん打ち", "2倍打ち") and len(mv[1]) == 1 and mv[1][0] in aimed)]
    wb = {}
    if AIM_WIDE_BONUS or AIM_WIDE_ONLY:
        ws = {n: c.hi - c.lo for n, c in sim.cells.items() if not c.pinned and c.v <= c.hi}
        wbar = sum(ws.values()) / max(1, len(ws))
        wb = {n: (w - wbar) / max(1.0, wbar) for n, w in ws.items()}
        if AIM_WIDE_ONLY and any(SKILLS[mv[0]]["crit"] > 1 and len(mv[1]) == 1 and wb.get(mv[1][0], 0) > 0 for mv in cands if mv[0] in SKILLS):
            cands = [mv for mv in cands if not (mv[0] in SKILLS and SKILLS[mv[0]]["crit"] > 1 and len(mv[1]) == 1 and wb.get(mv[1][0], 0) < 0)]

    def bonus(mv):
        if not AIM_WIDE_BONUS or mv[0] not in SKILLS or SKILLS[mv[0]]["crit"] <= 1:
            return 0.0
        return AIM_WIDE_BONUS * sum(wb.get(n, 0.0) for n in mv[1])
    scored = sorted(((evaluate(mv, seeds) + bonus(mv), mv) for mv in cands), key=lambda x: -x[0])
    if ADAPTIVE and len(scored) >= 2 and scored[0][0] - scored[1][0] < ADAPTIVE_GAP:
        # 上位が僅差 → 同じ乱数系列を延長して再評価（ノイズで手を取り違えない）
        more = seeds + [base.randrange(1 << 30) for _ in range(rollouts * 2)]
        top = [mv for _, mv in scored[:ADAPTIVE_TOP]]
        scored = sorted(((evaluate(mv, more) + bonus(mv), mv) for mv in top), key=lambda x: -x[0])
    best_v, best = scored[0]
    if best_v <= stop_v:
        return None
    return best


def finish_lookahead(sim: Sim, rollouts: int = 12, max_moves: int = 40):
    for _ in range(max_moves):
        if sim.focus < 5 and not sim.hissatsu_charged:  # 必殺(集中+30)が残っていれば続行
            break
        mv = choose(sim, rollouts)
        if mv is None:
            break
        skill, tg = mv
        if skill == "必殺":
            sim.act("必殺")
        else:
            sim.act(skill, tg)


def policy(sim: Sim, rollouts: int = 12):
    """run_amatsuyu.policy の finish を先読み版に差し替えたもの。"""
    R.opening(sim)
    R.plan_pairs(sim)
    R.phase_to_turn(sim, 1200)
    if sim.temp == 1200:
        R.fire_turn(sim)
    win8 = R.aim_window(sim, "x1.2", temp=800)
    p8 = R.choose_pair(sim, win8)
    if p8:
        sim.plan["p8"] = p8
    R.phase_to_turn(sim, 800)
    if sim.temp == 800:
        R.fire_turn(sim)
    finish_lookahead(sim, rollouts)


def next_move(sim: Sim, rollouts: int = 32):
    """任意の局面から次の1手を返す（アドバイザー/GUI/バックテスト共通の基準ポリシー）。
    開幕スクリプト → phase式(1200/800の上下ねらいまで) → 先読みfinish。None=しあげる。"""
    mv = R.opening_step(sim)
    if mv:
        return mv
    if sim.hissatsu_armed:  # 必殺押下済み → その対象の特技（phase/finish共通）
        tg = R.hissatsu_target(sim, focus_bonus=0)
        if tg and sim.can_use(tg[0]):
            return tg
    if sim.temp >= PHASE_MIN_TEMP and sim.focus >= 5:
        probe = sim.clone(random.Random(0), hide_ideal=True)
        if hasattr(sim, "plan"):
            probe.plan = dict(sim.plan)
        n0 = len(probe.log)
        if R.phase_step(probe) and len(probe.log) > n0:
            pplan = getattr(probe, "plan", None)  # 1200ちょうどの fire_turn は plan を作らない
            if pplan is not None:
                if not hasattr(sim, "plan"):
                    sim.plan = pplan
                else:
                    sim.plan.update(pplan)
            return (probe.log[n0][0], tuple(probe.log[n0][1]))
    return choose(sim, rollouts)


def policy_ref(sim: Sim, rollouts: int = 32, max_moves: int = 80):
    """next_move を打ち続ける基準ポリシー（バックテスト用）。"""
    for _ in range(max_moves):
        mv = next_move(sim, rollouts)
        if mv is None:
            break
        sk, tg = mv
        if sk == "必殺":
            sim.act("必殺")
        else:
            sim.act(sk, tg)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--rollouts", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    g0 = g1 = g2 = 0
    for i in range(a.n):
        s0 = R.make_amatsuyu(random.Random(i)); R.policy(s0)
        s1 = R.make_amatsuyu(random.Random(i)); policy(s1, a.rollouts)
        s2 = R.make_amatsuyu(random.Random(i)); policy_ref(s2, a.rollouts)
        g0 += s0.total_error() <= GREAT
        g1 += s1.total_error() <= GREAT
        g2 += s2.total_error() <= GREAT
        l1 = [(m[0], tuple(m[1])) for m in s1.log]; l2 = [(m[0], tuple(m[1])) for m in s2.log]
        if l1 != l2:
            mism = globals().setdefault("MISM", [])
            mism.append(i)
    print(f"n={a.n} rollouts={a.rollouts}  現行finish {g0/a.n*100:.1f}%  先読みfinish {g1/a.n*100:.1f}%  "
          f"next_move基準 {g2/a.n*100:.1f}%  ({time.time()-t0:.0f}s)")
    mism = globals().get("MISM", [])
    print(f"棋譜不一致(ハイブリッド vs next_move): {len(mism)}/{a.n}  例: {mism[:8]}")
