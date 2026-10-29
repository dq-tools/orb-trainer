#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超あまつゆのいと 打ちポリシー + モンテカルロ評価。

v1.1: 倍ターン(1200/800)を軸にした温度スロット計画型ポリシー。
- 固定オープニング（倍ターン1600/2000に熱風・超4連）
- 1200/800到達前に縦ペアのRを「倍ターン上下ねらいの会心確定レンジ」へ運ぶ
- 倍ターンで上下ねらい打ち（会心=2マス誤差0）
- 残りは通常温度のねらい打ち・半減ターンのてかげんで仕上げ
- 温度合わせに冷やし込み(-300)・熱風(-150)を使い、半減温度での打撃を避ける

使い方: python run_amatsuyu.py [-n 10000] [--seed 1] [--verbose]
"""
import argparse
import math
import random
import sys
from collections import Counter

from kaji_sim import (ASSUMPTIONS, SKILLS, Sim, aim_window, aim_window_cell, cost_of,
                      dmg_candidates, grade, make_amatsuyu, safe_max,
                      turn_multiplier)


def opening(sim: Sim):
    sim.act("火力上げ")
    sim.act("火力上げ")                        # →1600(倍)
    sim.act("熱風おろし", ("A",))              # @1600倍 →1450
    sim.act("火力上げ")
    sim.act("火力上げ")                        # →2050
    sim.act("ななめ打ち", ("D", "E"))          # @2050 →2000
    sim.act("超4連打ち", ("C", "D", "E", "F"))  # @2000倍 →1950
    sim.act("たたく", ("B",))                  # →1900
    sim.act("熱風おろし", ("B",))              # →1750
    name = min(("B", "F"), key=lambda n: sim.cells[n].v)
    sim.act("熱風おろし", (name,))             # →1600(倍)
    name = min(("B", "F"), key=lambda n: sim.cells[n].v)
    sim.act("熱風おろし", (name,))             # @1600倍 →1450


# 開幕スクリプト（opening と同一）。(温度, 集中) の軌跡で現在の手番を特定できる
OPENING_SCRIPT = [
    ("火力上げ", ()), ("火力上げ", ()), ("熱風おろし", ("A",)),
    ("火力上げ", ()), ("火力上げ", ()), ("ななめ打ち", ("D", "E")),
    ("超4連打ち", ("C", "D", "E", "F")), ("たたく", ("B",)), ("熱風おろし", ("B",)),
    ("熱風おろし", "minBF"), ("熱風おろし", "minBF"),
]


def opening_step(sim: Sim):
    """開幕スクリプトの途中なら次の1手を返す（温度・集中が軌跡と一致する時のみ）。"""
    temp, focus = 1000, ASSUMPTIONS["focus_max"]
    for skill, tg in OPENING_SCRIPT:
        if sim.temp == temp and sim.focus == focus:
            if tg == "minBF":
                tg = (min(("B", "F"), key=lambda n: sim.cells[n].v),)
            return (skill, tuple(tg))
        focus -= SKILLS[skill]["cost"]
        temp = max(0, temp + SKILLS[skill]["dtemp"])
    return None


PHASE_TURNS = (1200, 800)   # 上下ねらいを撃つ倍ターン。(1600, 1200, 800) で ひつひつ羊式（1600で左ペア）を試せる


def phase_step(sim: Sim) -> bool:
    """開幕後〜800度上下ねらいまでの1手（局面だけから決まる）。打てたら True。"""
    for t in PHASE_TURNS:
        if t <= 1200:
            break
        if sim.temp > t:
            return phase_iter(sim, t)
        if sim.temp == t:
            n0 = len(sim.log); fire_turn(sim)
            if len(sim.log) > n0:
                return True
    if sim.temp > 1200:
        return phase_iter(sim, 1200)
    if sim.temp == 1200:
        n0 = len(sim.log); fire_turn(sim)
        if len(sim.log) > n0:
            return True
        # 1200で撃つ手が無い → そのまま800フェーズへ（policy() と同じ流れ）
    plan = getattr(sim, "plan", None)
    if plan is None:
        plan_pairs(sim); plan = sim.plan
    if not plan.get("p8_fixed"):
        p8 = choose_pair(sim, aim_window(sim, "x1.2", temp=800))
        if p8:
            plan["p8"] = p8
        plan["p8_fixed"] = True
    if sim.temp > 800:
        return phase_iter(sim, 800)
    if sim.temp == 800:
        n0 = len(sim.log); fire_turn(sim); return len(sim.log) > n0
    return False


def haul_into(sim: Sim, name: str, floor: int, slots_left: int = 99, allow_tekagen: bool = True) -> bool:
    """R が floor を下回らない範囲で運ぶ。集中効率優先: 熱風(6)>3倍(11)>2倍(8)>たたく(5)。

    熱風は温度を150下げる（スロット3消費）ので slots_left>=3 の時だけ使う。
    """
    r = sim.cells[name].r
    z = turn_multiplier(sim.trait, sim.temp)
    order = []
    if z <= 0.5:
        # 半減ターン（威力半分）: 高い技を使っても半分しか進まない → 安い技で刻む（mori1 2026-09-06「半減なのに2倍打ちは無駄」）。
        # 大きく運ぶ必要がある時（残りが大きい）だけ熱風/3倍を後ろに残す
        order = ["たたく"] + (["てかげん打ち"] if allow_tekagen else []) + (["熱風おろし"] if slots_left >= 3 else []) + ["3倍打ち", "2倍打ち"]
    else:
        if slots_left >= 3:
            order.append("熱風おろし")
        order += ["3倍打ち", "2倍打ち", "たたく"] + (["てかげん打ち"] if allow_tekagen else [])
    for skill in order:
        if not sim.can_use(skill):
            continue
        c = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
        dmax = math.ceil(c[-1] * z)
        if dmax <= r - floor:
            sim.act(skill, (name,))
            return True
    return False


def zone_deficit_cost(sim: Sim, exclude: tuple[str, ...] = ()) -> int:
    """全マスをゾーン下限まで運ぶのに必要な集中力の概算（予約分）。"""
    total = 0
    for n, c in sim.cells.items():
        if n in exclude or c.v >= c.lo:
            continue
        need = c.lo - c.v
        # 入場の実コスト: 小さい不足はてかげん(10)×1-2回、大きい不足は運搬+入場
        if need <= 10:
            total += 12
        else:
            total += int(need / 3) + 12
    return total


def can_afford_luxury(sim: Sim, cost: int, exclude: tuple[str, ...] = ()) -> bool:
    """予約(全マスゾーン入れ予算)を残した上で cost を払えるか。"""
    return sim.focus - cost >= zone_deficit_cost(sim, exclude)


WINDOW_MODE = "sim"   # "sim"=会心確定域そのまま(理想=中央±2前提) / "wero"=ウェロア表の窓（2026-09-06 比較用）


def cell_win(sim: Sim, name: str, win: tuple[int, int]) -> tuple[int, int]:
    """マスごとの上下ねらい窓（残り r の範囲）。
    "wero": ウェロア氏の準備数値表（動画 Xor_WC_ZQKs）を r に直したもの。
      下限 = 非会心最大 + (11幅ゾーンなら 理想-下限: 非会心でゾーンに入らずもう一度狙える) / 7幅は真ん中まで許容
      上限 = 会心最低値 - (上端-理想): 理想がゾーン上端にあっても会心が届く（理想位置に頼らない）"""
    if WINDOW_MODE != "wero":
        return win
    c = sim.cells[name]
    lo, hi = win
    if c.hi - c.lo >= 10:
        lo += c.ideal - c.lo
    return (lo, hi - (c.hi - c.ideal))


def choose_pair(sim: Sim, win: tuple[int, int]):
    """開いている縦ペアのうち、両マスとも win に運べそうなものを選ぶ。"""
    def openc(n):
        return not sim.cells[n].in_zone and sim.cells[n].r > 0
    reserved = hissatsu_reserved(sim)   # 必殺の面のマスはペアに使わない（集中を使わない）
    best = None
    for a, b in sim.pairs_v:
        if reserved & {a, b}:
            continue
        if openc(a) and openc(b):
            ra, rb = sim.cells[a].r, sim.cells[b].r
            (la, ha), (lb, hb) = cell_win(sim, a, win), cell_win(sim, b, win)
            if ra >= la and rb >= lb:  # まだ運べる（下回っていない）
                score = -(abs(ra - (la + ha) // 2) + abs(rb - (lb + hb) // 2))
                if best is None or score > best[0]:
                    best = (score, (a, b))
    return best[1] if best else None


def use_hissatsu(sim: Sim) -> bool:
    """必殺チャージの最適消費: 会心確定なので理想値が射程内のマスに撃てば誤差0確定。"""
    if not sim.hissatsu_charged:
        return False
    tg = hissatsu_target(sim)
    if tg is None:
        return False
    sim.act("必殺")
    if sim.can_use(tg[0]):
        sim.act(tg[0], tg[1])
    return True


def use_armed(sim: Sim) -> bool:
    """必殺押下済み(会心確定が装填中)なら、その対象特技を撃つ。"""
    if not sim.hissatsu_armed:
        return False
    tg = hissatsu_target(sim, focus_bonus=0)
    if tg and sim.can_use(tg[0]):
        sim.act(tg[0], tg[1])
        return True
    return False


HISSATSU_QUAD_ONLY = True  # 必殺は超4連で4マス本会心の時のみ（面が両方塞がった時だけペア/単体に妥協）


def hissatsu_target(sim: Sim, focus_bonus: int | None = None):
    """必殺で撃つ (特技, 対象) を局面から決める。None=今は撃たない。
    focus_bonus: 必殺押下で増える集中（未押下なら30、押下済み(armed)なら0を渡す）。"""
    import math as _m
    if focus_bonus is None:
        focus_bonus = ASSUMPTIONS["hissatsu_focus"]
    focus = sim.focus  # 判定は押下前の集中（元実装と同じ）
    # 未押下なら押下で温度-50してから打つので、射程は押下後の温度で計算
    t_hit = sim.temp if sim.hissatsu_armed else max(0, sim.temp + SKILLS["必殺"]["dtemp"])
    z = turn_multiplier(sim.trait, t_hit)
    reach12 = 2 * _m.ceil(dmg_candidates(t_hit, "x1.2")[0] * z)
    # 対象条件: 理想値より確実に手前(ブレ±2込み)。会心は届けば理想値で停止・
    # 届かなくても手前に落ちるだけなので上限は不要（mori1さん検証）
    def ok(c, reach):
        # 理想の手前(ブレ±2込みで r>=4)なら対象。ゾーン内でも可（必殺は会心確定なので偽会心の心配なし）
        return (not c.pinned) and c.v <= c.hi and c.r >= 4
    # 定石: 超4連で4マス同時に本会心（mori1: 必ず4か所が本会心になるタイミングで使う。2026-09-04）
    reach20 = 2 * _m.ceil(dmg_candidates(t_hit, "x2.0")[0] * z)
    quad_possible = False  # いずれかの面がまだ4マス本会心を狙える（全マス未完成）か
    fb = focus + focus_bonus   # 押下後に使える集中（回復は ASSUMPTIONS hissatsu_focus。集中力変化の会心ターンは消費 1.5 倍なので cost_of で見る）
    if fb >= cost_of("超4連打ち", sim.trait, t_hit):
        best_quad = None
        for quad in sim.quads:
            cs = [sim.cells[n] for n in quad]
            # 本会心済みマスは Miss で無害。それ以外は全て理想手前(ok)でないと面は成立しない
            if not all(c.pinned or ok(c, reach20) for c in cs):
                continue
            live = [c for c in cs if not c.pinned]
            if not live:
                continue
            quad_possible = True
            n_pin = sum(1 for c in live if c.r <= reach20 - 4)
            if n_pin == len(live) or (not HISSATSU_QUAD_ONLY and n_pin >= 2):
                score = (len(cs) - len(live) + n_pin, sum(min(c.r, reach20) for c in live))
                if best_quad is None or score > best_quad[0]:
                    best_quad = (score, quad)
        if best_quad:
            return ("超4連打ち", tuple(best_quad[1]))
    if HISSATSU_QUAD_ONLY and quad_possible:
        return None  # 4マス本会心の面がまだ作れる → 温存して配置を待つ
    # 縦ペア両マスが射程内 → 上下ねらいでダブル誤差0
    for a, b in sim.pairs_v:
        ca, cb = sim.cells[a], sim.cells[b]
        if ok(ca, reach12) and ok(cb, reach12)                 and ca.r <= reach12 - 4 and cb.r <= reach12 - 4 and fb >= cost_of("上下ねらい打ち", sim.trait, t_hit):
            return ("上下ねらい打ち", (a, b))
    reach10 = 2 * _m.ceil(dmg_candidates(t_hit, "x1.0")[0] * z)
    best = None
    for n, c in sim.cells.items():
        reach_n = reach10 * (2 if n == sim.lit_cell else 1)  # 光ったマスはダメージ2倍→射程2倍
        if ok(c, reach_n) and c.r <= reach_n - 4                 and (best is None or c.r > sim.cells[best].r):
            best = n
    if best and fb >= cost_of("ねらい打ち", sim.trait, t_hit):
        return ("ねらい打ち", (best,))
    return None


def hissatsu_quad_plan(sim: Sim):
    """必殺チャージ中に「4マス本会心」を狙える面（超4連の対象）を返す。None=面は作れない。
    mori1（2026-09-06）: 面のマスは必殺で確定するので集中を使わない。射程に足りないマスだけ軽く運び、
    残りの集中は面の外のペアに使う。戻り値: (面のタプル, 射程に足りないマス名のリスト, reach20)"""
    import math as _m
    if not sim.hissatsu_charged or sim.hissatsu_armed:
        return None
    t_hit = max(0, sim.temp + SKILLS["必殺"]["dtemp"])
    if sim.trait == "倍半":
        # 必殺は倍ターン（400の倍数）で撃つ前提: 押下後に到達できる次の倍ターンで射程を見積もる
        # （押下直後の温度で見ると 1000/600 の半減ターンで射程が半分になり面が化ける）
        t_hit = max(400, (t_hit // 400) * 400)
    z = turn_multiplier(sim.trait, t_hit)
    reach20 = 2 * _m.ceil(dmg_candidates(t_hit, "x2.0")[0] * z)
    best = None
    for quad in sim.quads:
        cs = [sim.cells[n] for n in quad]
        if not all(c.pinned or (c.v <= c.hi and c.r >= 4) for c in cs):
            continue
        live = [(n, sim.cells[n]) for n in quad if not sim.cells[n].pinned]
        if not live:
            continue
        short = [n for n, c in live if c.r > reach20 - 4]
        score = (len(live) - len(short), -sum(sim.cells[n].r - (reach20 - 4) for n in short))
        if best is None or score > best[0]:
            best = (score, tuple(quad), short, reach20)
    return best[1:] if best else None


def hissatsu_reserved(sim: Sim) -> set:
    """必殺の面で確定予定のマス（ねらい系や釣りで集中を使わない）。"""
    plan = hissatsu_quad_plan(sim)
    return set(plan[0]) if plan else set()


def phase_to_turn(sim: Sim, turn_temp: int):
    """turn_temp(倍ターン)まで、計画ペアを会心確定レンジへ運びつつ温度を消化する。"""
    while sim.temp > turn_temp and sim.focus >= 5:
        if not phase_iter(sim, turn_temp):
            break


QUAD_HAUL = False      # 中盤の運搬に 4連打ち（4打=必殺チャージ判定4回）を混ぜる（2026-10-12 実験: 打数を増やして途中チャージ率を上げる）
QUAD_HAUL_MIN_FOCUS = 120


def quad_haul(sim: Sim, floors: dict, slots: int):
    """4マスとも floor を割らずに打てる 2x2 があれば 4連打ち（余裕があれば 超4連）。打てたら True。"""
    if sim.focus < QUAD_HAUL_MIN_FOCUS or slots < 1:
        return False
    z = turn_multiplier(sim.trait, sim.temp)
    best = None
    for skill in ("超4連打ち", "4連打ち"):
        if not sim.can_use(skill):
            continue
        dmax = math.ceil(dmg_candidates(sim.temp, SKILLS[skill]["mult"])[-1] * z)
        for quad in sim.quads:
            ok = True; tot = 0
            for n in quad:
                c = sim.cells[n]
                if c.in_zone or c.r <= 0 or c.r - dmax < floors.get(n, 12):
                    ok = False; break
                tot += c.r
            if ok:
                score = (tot, skill == "超4連打ち")
                if best is None or score > best[0]:
                    best = (score, skill, tuple(quad))
    if best:
        sim.act(best[1], best[2])
        return True
    return False


def phase_iter(sim: Sim, turn_temp: int) -> bool:
    """phase_to_turn の1反復（1手）。打てたら True。局面だけから決まる（アドバイザー共用）。"""
    win = aim_window(sim, "x1.2", temp=turn_temp)
    win8 = aim_window(sim, "x1.2", temp=800)
    plan = getattr(sim, "plan", None)
    if plan is None:
        plan_pairs(sim)
        plan = sim.plan
    if True:
        if use_armed(sim) or use_hissatsu(sim):
            return True
        hq = hissatsu_quad_plan(sim)
        if hq:
            quad, short, reach20 = hq
            # 面に足りないマス（射程超え）を安全な最小限の打撃で射程に入れる → 次の手で必殺
            for n in sorted(short, key=lambda n: sim.cells[n].r):
                need = sim.cells[n].r - (reach20 - 4)
                for skill, mk in (("たたく", "x1.0"), ("てかげん打ち", "x0.5"), ("2倍打ち", "x2.0"), ("熱風おろし", "x2.5")):
                    if not sim.can_use(skill):
                        continue
                    z_ = turn_multiplier(sim.trait, sim.temp)
                    cand = dmg_candidates(sim.temp, mk)
                    dmin_, dmax_ = math.ceil(cand[0] * z_), math.ceil(cand[-1] * z_)
                    if dmin_ >= need and dmax_ <= sim.cells[n].r - 4:
                        sim.act(skill, (n,))
                        return True
        pair = plan.get("p12") if turn_temp == 1200 else None
        if pair and hissatsu_reserved(sim) & set(pair):
            pair = None
        if pair is None or not all(not sim.cells[n].in_zone and sim.cells[n].r > 0
                                   for n in pair):
            pair = choose_pair(sim, win)
        pair_cells = set(pair) if pair else set()
        # 800用ペアは win8 の下限を割らないように保護（運びすぎ禁止）
        protect8 = set(plan.get("p8") or ()) if turn_temp == 1200 else set()
        acted = False
        halved = turn_multiplier(sim.trait, sim.temp) == 0.5

        slots = (sim.temp - turn_temp) // 50
        # 1) ペアのマスを win へ運ぶ（半減ターンは微調整に最適）
        # 倍ターン（威力2倍）はてかげんの微調整に使わず、遠いマスへの大きい一撃に回す（mori1 2026-09-06:
        # 「倍ターンにCへてかげんは効率が悪い。Fを伸ばすべき」）。調整は残りの通常/半減ターンで行う
        doubled = turn_multiplier(sim.trait, sim.temp) >= 2.0 and slots >= 2
        if QUAD_HAUL and not halved and slots >= 2:
            floors = {}
            for n in pair_cells:
                floors[n] = cell_win(sim, n, win)[0]
            for n in protect8:
                floors[n] = max(floors.get(n, 0), win8[0])
            for n in hissatsu_reserved(sim):
                floors[n] = 10 ** 6   # 必殺の面のマスには触らない
            if quad_haul(sim, floors, slots):
                return True
        if pair:
            for n in sorted(pair_cells, key=lambda n: (-sim.cells[n].r, n)):  # 同点はマス名順（set順は非決定）
                wl, wh = cell_win(sim, n, win)
                if sim.cells[n].r > wh:
                    if haul_into(sim, n, wl, slots, allow_tekagen=not doubled):
                        acted = True
                        break
        # 2) ペア外の開きマスを運ぶ（800用ペアは win8 下限を保護。必殺の面のマスは運ばない=集中を使わない）
        reserved = hissatsu_reserved(sim)
        if hq:
            reserved -= set(hq[1])  # 射程に足りないマスは上で運ぶ
        if not acted:
            for n in sorted(sim.cells,
                            key=lambda n: -(sim.cells[n].lo - sim.cells[n].v)):
                c = sim.cells[n]
                if n in pair_cells or c.in_zone or c.r <= 0 or n in reserved:
                    continue
                floor = max(12, win8[0]) if n in protect8 else 12
                if c.r > max(24, floor + 8) and haul_into(sim, n, floor, slots):
                    acted = True
                    break
        # 3) やることがない: 温度だけ消化する
        if not acted and sim.temp - 150 >= turn_temp and sim.can_use("熱風おろし"):
            # 熱風は3スロット消化しつつ大きく運べる最効率バーナー
            for n in sorted(sim.cells, key=lambda n: -sim.cells[n].r):
                c = sim.cells[n]
                if not c.in_zone and c.r > 0 and safe_max(sim, n, "x2.5"):
                    sim.act("熱風おろし", (n,))
                    acted = True
                    break
        if not acted:
            if sim.temp - 300 >= turn_temp and sim.can_use("冷やし込み"):
                sim.act("冷やし込み")
                acted = True
            elif not halved:
                # ねらい打ちで仕上げ域のマスを拾う（通常倍率ターンのみ）
                lo10, hi10 = aim_window(sim, "x1.0")
                if sim.can_use("ねらい打ち"):
                    for n in sim.cells:
                        c = sim.cells[n]
                        if not c.in_zone and lo10 <= c.r <= hi10                                 and can_afford_luxury(sim, 16 + 6, (n,)):
                            sim.act("ねらい打ち", (n,))
                            acted = True
                            break
            if not acted:
                # てかげんで一番遠いマスを刻んで温度を1つ進める
                for n in sorted(sim.cells, key=lambda n: -sim.cells[n].r):
                    c = sim.cells[n]
                    if not c.in_zone and c.r > 0 and sim.can_use("てかげん打ち") \
                            and safe_max(sim, n, "x0.5"):
                        sim.act("てかげん打ち", (n,))
                        acted = True
                        break
        return acted


AIM_TURNS = (1200, 800)   # 倍ターンでねらい系を使う温度。(800,) にすると 1200 は削るだけ（mori1案 2026-09-06 の比較用）


def fire_turn(sim: Sim):
    """倍ターン当日: 上下ねらい > ねらい > 大きく安全に削る。"""
    if sim.temp not in AIM_TURNS:
        # ねらい系を使わず、倍ダメージで大きく安全に削る
        for n in sorted(sim.cells, key=lambda n: -sim.cells[n].r):
            c = sim.cells[n]
            if not c.in_zone and c.r > 0 and haul_into(sim, n, 0):
                return
        if sim.can_use("てかげん打ち"):
            for n in sim.cells:
                c = sim.cells[n]
                if not c.in_zone and c.r > 0 and safe_max(sim, n, "x0.5"):
                    sim.act("てかげん打ち", (n,))
                    return
        return
    win = aim_window(sim, "x1.2")
    plan = getattr(sim, "plan", {})
    key = "p12" if sim.temp == 1200 else ("p8" if sim.temp == 800 else None)
    pair = plan.get(key) if key else None
    if pair is None or not all(not sim.cells[n].in_zone and
                               win[0] <= sim.cells[n].r for n in pair):
        pair = choose_pair(sim, win)
    reserved = hissatsu_reserved(sim)
    if pair and reserved & set(pair):
        pair = None
    if pair and sim.can_use("上下ねらい打ち") \
            and cell_win(sim, pair[0], win)[0] <= sim.cells[pair[0]].r <= cell_win(sim, pair[0], win)[1] \
            and cell_win(sim, pair[1], win)[0] <= sim.cells[pair[1]].r <= cell_win(sim, pair[1], win)[1]:
        sim.act("上下ねらい打ち", pair)
        return
    lo10, hi10 = aim_window(sim, "x1.0")
    if sim.can_use("ねらい打ち"):
        for n in sorted(sim.cells, key=lambda n: -sim.cells[n].r):
            c = sim.cells[n]
            if n in reserved:
                continue  # 必殺の面で確定するマスには集中を使わない（mori1）
            if not c.in_zone and lo10 <= c.r <= hi10                     and can_afford_luxury(sim, 16 + 6, (n,)):
                sim.act("ねらい打ち", (n,))
                return
    hq = hissatsu_quad_plan(sim)
    keep = reserved - set(hq[1]) if hq else reserved
    for n in sorted(sim.cells, key=lambda n: -sim.cells[n].r):
        c = sim.cells[n]
        if n in keep:
            continue
        if not c.in_zone and c.r > 0 and haul_into(sim, n, 0):
            return
    if sim.can_use("てかげん打ち"):
        for n in sim.cells:
            c = sim.cells[n]
            if n in keep:
                continue
            if not c.in_zone and c.r > 0 and safe_max(sim, n, "x0.5"):
                sim.act("てかげん打ち", (n,))
                return


_ZE_CACHE = {}


def zone_entry_skill(sim: Sim, name: str) -> str | None:
    """V < lo のマスをゾーンに入れる特技。ゾーン上限(hi)を超えない中で最善を選ぶ。"""
    c = sim.cells[name]
    key = (sim.trait, sim.temp, sim.focus, c.lo - c.v, c.hi - c.v)
    r = _ZE_CACHE.get(key, 0)
    if r != 0:
        return r
    r = _zone_entry_skill(sim, c.lo - c.v, c.hi - c.v)
    _ZE_CACHE[key] = r
    return r


def _zone_entry_skill(sim: Sim, need: int, room: int) -> str | None:
    import math as _m
    z = turn_multiplier(sim.trait, sim.temp)
    best = None
    for skill in ("たたく", "てかげん打ち", "2倍打ち", "熱風おろし"):
        if not sim.can_use(skill):
            continue
        cand = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
        dmin, dmax = _m.ceil(cand[0] * z), _m.ceil(cand[-1] * z)
        if dmax > room:
            continue  # はみ出しリスクあり
        guaranteed = dmin >= need
        score = (guaranteed, dmin)  # ゾーン内確定 > より深く入る
        if best is None or score > best[0]:
            best = (score, skill)
    return best[1] if best else None


_PF_CACHE = {}


def pin_fish_skill(sim: Sim, name: str) -> str | None:
    """ゾーン内(or直前)のマスで、外してもゾーンを出ない誤差0釣りができる特技。"""
    c = sim.cells[name]
    if c.r <= 0:
        return None
    key = (sim.trait, sim.temp, sim.focus, c.hi - c.v, c.r)
    r = _PF_CACHE.get(key, 0)
    if r != 0:
        return r
    r = _pin_fish_skill(sim, c)
    _PF_CACHE[key] = r
    return r


def _pin_fish_skill(sim: Sim, c) -> str | None:
    import math as _m
    z = turn_multiplier(sim.trait, sim.temp)
    for skill in ("ねらい打ち", "弱ねらい打ち"):
        if not sim.can_use(skill):
            continue
        cand = dmg_candidates(sim.temp, SKILLS[skill]["mult"])
        dmin, dmax = _m.ceil(cand[0] * z), _m.ceil(cand[-1] * z)
        # 会心なら本会心(誤差0)、非会心でも理想超過が現誤差を超えない(dmax<=2r)＝
        #   外しても悪化しない。hi-vだけだと非会心で理想を大きく踏み越え固着する(mori1)
        if dmax <= c.hi - c.v and dmax <= 2 * c.r and c.r <= 2 * dmin:
            return skill
    return None


CENTER_HIT = False   # 仕上げ: 釣れない時、ゾーン内の未確定マスを安い一撃で中央へ寄せる（期待誤差を下げる。2026-10-12 実験）
CENTER_MIN_GAIN = 0.5


def center_hit(sim: Sim, names):
    """ゾーン内（か直前）の未確定マスを、全目がゾーン内に収まる安い一撃で理想（中央±2の信念）へ寄せる。
    期待誤差の減りが最大の (特技, マス) を返す。無ければ None。"""
    import math as _m
    z = turn_multiplier(sim.trait, sim.temp)
    best = None
    for n in names:
        c = sim.cells[n]
        if c.pinned or c.v > c.hi:
            continue
        from kaji_sim import ideal_offsets
        ideals = [max(c.lo, min(c.hi, c.center + k)) for k in ideal_offsets(c.lo, c.hi)]   # 理想値の信念（範囲で一様）

        def exp_err(v, c=c, ideals=ideals):
            tot = 0.0
            for it in ideals:
                e = abs(v - it)
                tot += min(e, 4) if c.lo <= v <= c.hi else e + 3
            return tot / len(ideals)
        cur = exp_err(c.v)
        for skill in ("てかげん打ち", "たたく"):
            if not sim.can_use(skill):
                continue
            cand = [_m.ceil(d * z) for d in dmg_candidates(sim.temp, SKILLS[skill]["mult"])]
            if cand[-1] > c.hi - c.v:
                continue
            after = sum(exp_err(c.v + d) for d in cand) / len(cand)
            gain = cur - after
            if gain >= CENTER_MIN_GAIN:
                score = gain / cost_of(skill, sim.trait, sim.temp)
                if best is None or score > best[0]:
                    best = (score, skill, n)
    return (best[1], (best[2],)) if best else None


def finish(sim: Sim):
    """仕上げ: ねらい打ち→ゾーン入れ→誤差0釣り の順で安全に詰める。"""
    # 温度 50 でも打てる。以前は temp > 50 で抜けていたため、熱風おろしで 50 に落ちると集中力が残っていても
    # 打ち切り、下の「火力上げで延命」にも届かなかった（2026-10-18、局B 450℃ の局面で発見。大成功 0/40）
    while sim.focus >= 5:
        if sim.temp < 50:
            # 温度切れ: 未完成のマスがあり集中力が足りるなら火力上げで戻す。無ければ終わり
            if any(not c.pinned and (not c.in_zone or c.r > 0) for c in sim.cells.values())                     and sim.can_use("火力上げ") and sim.focus >= 15:
                sim.act("火力上げ")
                continue
            break
        if use_armed(sim) or use_hissatsu(sim):
            continue
        opens = [n for n, c in sim.cells.items() if not c.in_zone and 0 < c.r]
        z = turn_multiplier(sim.trait, sim.temp)
        acted = False
        lo10, hi10 = aim_window(sim, "x1.0")
        # 0) ゾーン未達マスの入場確定。ただし「窓に運んで釣る」予算があるマスは
        #    釣り優先（外してもゾーン内着地=入場を兼ねるため）
        if sim.temp <= 900:
            for n in sorted(opens, key=lambda n: sim.cells[n].r):
                c = sim.cells[n]
                # 完璧な釣り位置: 会心なら誤差0、外しても必ずゾーン内着地
                zone_grace = c.ideal - c.lo
                fishable_later = lo10 <= c.r <= min(hi10, lo10 + zone_grace)                     and can_afford_luxury(sim, 16, (n,))
                if fishable_later:
                    continue
                skill = zone_entry_skill(sim, n)
                if skill:
                    sim.act(skill, (n,))
                    acted = True
                    break
        # 1) 会心確定レンジ（グレース付き=外してもゾーン手前で再挑戦可）にねらい打ち
        if not acted and z != 2.0 and sim.can_use("ねらい打ち"):
            for n in sorted(opens, key=lambda n: -sim.cells[n].r):
                wlo, whi = aim_window_cell(sim, n, "x1.0")
                if wlo <= sim.cells[n].r <= whi                         and can_afford_luxury(sim, 16 + 6, (n,)):
                    sim.act("ねらい打ち", (n,))
                    acted = True
                    break
        # 2) まだ遠いマスを運ぶ
        if not acted:
            for n in sorted(opens,
                            key=lambda n: -(sim.cells[n].lo - sim.cells[n].v)):
                if sim.cells[n].r > hi10 and haul_into(sim, n, lo10):
                    acted = True
                    break
        # 3) ゾーン未満の近いマスをゾーンに入れる
        if not acted:
            for n in sorted(opens, key=lambda n: sim.cells[n].r):
                skill = zone_entry_skill(sim, n)
                if skill:
                    sim.act(skill, (n,))
                    acted = True
                    break
        # 4a) ゾーン内縦ペアへの上下ねらい釣り（リスクゼロの2連釣り）
        if not acted and sim.can_use("上下ねらい打ち"):
            import math as _m
            z2 = turn_multiplier(sim.trait, sim.temp)
            cand = dmg_candidates(sim.temp, "x1.2")
            dmax12 = _m.ceil(cand[-1] * z2)
            for a, b in sim.pairs_v:
                ca, cb = sim.cells[a], sim.cells[b]
                if ca.in_zone and cb.in_zone and ca.r > 0 and cb.r > 0                         and dmax12 <= ca.hi - ca.v and dmax12 <= cb.hi - cb.v                         and dmax12 <= 2 * ca.r and dmax12 <= 2 * cb.r:
                    sim.act("上下ねらい打ち", (a, b))
                    acted = True
                    break
        # 4) ゾーン内マスの誤差0釣り（外してもゾーンを出ない時だけ）
        if not acted:
            fishables = [n for n, c in sim.cells.items() if c.in_zone and c.r > 0]
            for n in sorted(fishables, key=lambda n: -sim.cells[n].r):
                skill = pin_fish_skill(sim, n)
                if skill:
                    sim.act(skill, (n,))
                    acted = True
                    break
        # 4c) 釣れない時: ゾーン内の未確定マスを安い一撃で中央へ寄せる（期待誤差を下げる）
        if not acted and CENTER_HIT:
            mv = center_hit(sim, [n for n, c in sim.cells.items() if not c.pinned])
            if mv:
                sim.act(mv[0], mv[1])
                acted = True
        # 5) 手詰まりでも温度を下げれば精密な手が生まれる → 冷やし込みで降温
        if not acted and sim.temp >= 350 and sim.can_use("冷やし込み") and                 (opens or any(c.in_zone and c.r > 0 for c in sim.cells.values())):
            sim.act("冷やし込み")
            acted = True
        # 6) 温度切れでゾーン未達が残っているなら火力上げで延命
        if not acted and sim.temp <= 100 and opens and sim.can_use("火力上げ")                 and sim.focus >= 15:
            sim.act("火力上げ")
            acted = True
        if not acted:
            break


def plan_pairs(sim: Sim):
    """1200用と800用の互いに素な縦ペアを割り当てる。"""
    win12 = aim_window(sim, "x1.2", temp=1200)
    win8 = aim_window(sim, "x1.2", temp=800)
    mid12 = (win12[0] + win12[1]) // 2
    mid8 = (win8[0] + win8[1]) // 2
    def openc(n):
        c = sim.cells[n]
        return not c.in_zone and c.r > 0
    best = None
    for p1 in Sim.PAIRS_V:
        if not all(openc(n) for n in p1):
            continue
        for p2 in Sim.PAIRS_V:
            if set(p1) & set(p2) or not all(openc(n) for n in p2):
                continue
            # どちらのペアも「窓より手前(R大きめ)」であるほど運びやすい
            c1 = sum(abs(sim.cells[n].r - mid12) for n in p1)
            c2 = sum(abs(sim.cells[n].r - mid8) for n in p2)
            feasible = all(sim.cells[n].r >= win12[0] for n in p1) and                        all(sim.cells[n].r >= win8[0] for n in p2)
            score = (feasible, -(c1 + c2))
            if best is None or score > best[0]:
                best = (score, (p1, p2))
    if best:
        sim.plan = {"p12": best[1][0], "p8": best[1][1]}
    else:
        sim.plan = {"p12": None, "p8": None}


def policy(sim: Sim):
    opening(sim)
    plan_pairs(sim)
    phase_to_turn(sim, 1200)
    if sim.temp == 1200:
        fire_turn(sim)
    # 1200の結果を踏まえて800用ペアを現状のRで選び直す
    win8 = aim_window(sim, "x1.2", temp=800)
    p8 = choose_pair(sim, win8)
    if p8:
        sim.plan["p8"] = p8
    phase_to_turn(sim, 800)
    if sim.temp == 800:
        fire_turn(sim)
    win4 = aim_window(sim, "x1.2", temp=400)
    if choose_pair(sim, win4) and sim.focus >= 25 + zone_deficit_cost(sim):
        phase_to_turn(sim, 400)
        if sim.temp == 400:
            fire_turn(sim)
    finish(sim)


def run_one(seed: int, verbose=False) -> dict:
    sim = make_amatsuyu(random.Random(seed))
    policy(sim)
    err = sim.total_error()
    if verbose:
        for skill, tg, temp, res in sim.log:
            print(f"  {skill}{list(tg)} → {temp}℃ {res}")
        for n, c in sim.cells.items():
            print(f"  {n}: {c.v} (zone {c.lo}-{c.hi}, err {c.error})")
        print(f"  合計誤差 {err} / 残集中 {sim.focus}")
    return dict(err=err, in_zone=sum(c.in_zone for c in sim.cells.values()),
                focus_left=sim.focus,
                great=err <= ASSUMPTIONS["great_success_max_err"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        run_one(args.seed, verbose=True)
        return 0
    stats = [run_one(args.seed + i) for i in range(args.n)]
    errs = Counter(s["err"] for s in stats)
    zone6 = sum(s["in_zone"] == 6 for s in stats) / len(stats)
    grades = Counter(grade(6, s["err"]) for s in stats)
    print(f"試行 {args.n}回")
    print("できのよさ:", {k: f"{v/len(stats):.1%}" for k, v in
          sorted(grades.items(), key=lambda kv: -kv[1])})
    print(f"全マスゾーンイン率: {zone6:.1%}")
    print(f"平均誤差: {sum(s['err'] for s in stats)/len(stats):.2f} / "
          f"平均残集中: {sum(s['focus_left'] for s in stats)/len(stats):.1f}")
    print("誤差分布:", dict(sorted(errs.items())[:15]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
