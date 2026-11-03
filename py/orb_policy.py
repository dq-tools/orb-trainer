#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""虹色のオーブ専用の方針（アメリーさん型、2026-10-26。mori1 依頼・kaji-kaisetsu 経由）。

盤面 A B / C D / E F（A・F 85〜95、B・D 140〜148、C・E 115〜121、再生）。
芯: 「会心が出なければ範囲に入らずに止まり、会心が出れば理想値で止まる」距離（狙い距離）に、マスを少ない手数で持っていく。
  狙い距離 = その温度・その技の出目で、ふつうの出目の最大では範囲の下限に届かず（v + dmax < lo）、2 倍の最小で下限に届く（v + 2·dmin ≥ lo）
手順:
 1. 火力上げ×2 → 1600 超4連 ABCD → 1550 超4連 CDEF
 2. 1500: A と F のうち進んでいる方の側に 4連（決まった角があれば反対の側）→ 進めた角が狙い距離なら高温のうちにねらい打ち
 3. 1450〜1100: いちばん遅れている B・E に熱風（B を D に、E を C にそろえる）
 4. 縦の 2 マス（B+D、C+E、A+C、D+F）が揃ったら、遠いうちは上下打ちで一緒に寄せ、両方が狙い距離なら上下ねらい
 5. 残りは温度を下げながらねらい。700 以下は弱ねらい・てかげん。C・E は真ん中へ（仕上げは先読み＋厳密解に任せる）
 6. 必殺は超4連で 3〜4 マス（run_amatsuyu.hissatsu_target）

python orb_policy.py -n 200 [--seed0 0] で大成功率
"""
import math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import kaji_sim as KS
from kaji_sim import SKILLS, Sim, dmg_candidates, make_item, turn_multiplier
import run_amatsuyu as R
import finish_lookahead as FL
import kaji_generic as KG

ITEM = "虹色のオーブ"
ROLLOUTS = 16
STEP2_TEMP = 1500          # 4連を打つ温度
HEAT_MIN_TEMP = 1100       # 熱風で B・E をそろえる温度の下限
PAIR_MIN_TEMP = 800        # 上下打ちで寄せる温度の下限
MATCH_TOL = 12             # 相棒との差がこれ以下なら「揃った」
AIM_MIN_TEMP = 700         # ねらい打ち（x1.0）を使う温度の下限。以下は弱ねらい/てかげん（先読みに任せる）
STEP2_FALLBACK = True      # 【既定 ON 2026-10-26: 5 手目の一致 51→72%、大成功 1000 局 74.1→75.0（±0）】 1500℃ の 4連が打てない時（開幕の会心で C/D が範囲に入った・角が進みすぎた）: 反対の側の 4連 → 当たっていないマスのペア技（ななめ DE・上下 CE・左右 EF…）。
                           # アメリーさんの 5 手目（80 局）: C が会心なら ななめ DE（13）、D が会心なら 上下 CE/左右 EF（8）、片方の角が進んでいても遅れた側に 4連（2026-10-26）
AMELIE_OPENING = False     # アメリーさんの固定手順（80 局の 6〜8 手目の多数派、2026-10-26 mori1「固定化されている所は固定して」）:
                           # 1450℃ 熱風 B（B が遅れていなければ E）→ 1300℃ 上下打ち BD（打てなければ たたく B）→ 1250℃ 熱風 E → 以後は通常の方針。角のねらいは 1100℃ 以下まで待つ
STEP2_MARGIN = 0           # 【既定 0。6 だと 4連を避けすぎて一致 62%】 1500℃ の 4連: 各マスが打った後も下限までこれだけ残る側だけ（アメリーさんは C が 84 以上だと 4連を避けて ななめ DE。2026-10-26）
STEP2_TIE = "F8"           # 【既定 "F8" 2026-10-26】 1500 の 4連で A と F の進みが同じ時にどちらの側へ打つか（"A"／"F"／"far"=角の残りが大きい側）
CORNER_HEAT = False        # 遅れている角（A・F）も熱風で狙い距離の手前まで運ぶ（アメリーさん: 1050 熱風 F → 900 ねらい F。AI は F へのねらいが A より 16% 少ない。2026-10-26 mori1「F へのねらいが低い」）
CORNER_HEAT_MIN_TEMP = 900
CORNER_FIRST = False       # 幅の広い角（A・F、幅 11）は単体のねらいで取りに行く: 狙い距離に入った角があれば上下ねらい・他のマスより先にねらい、本会心にならなくても CORNER_TRIES 回まで続ける（mori1 2026-10-26「明確に幅の広い所をねらいで」）
CORNER_TRIES = 2
CORNER_FIRST_MIN_TEMP = 500
CORNER_FIRST_MAX_TEMP = 10 ** 9   # アメリーさんは F を 700〜1050℃ でねらう（1450℃ の直後は A だけ）。上限を 1050 にすると高温の角ねらいを待つ


def corner_first_move(sim: Sim):
    if not (CORNER_FIRST_MIN_TEMP <= sim.temp <= CORNER_FIRST_MAX_TEMP) or not sim.can_use("ねらい打ち"):
        return None
    cells = sim.cells
    tries = {n: sum(1 for m in sim.log if m[0] == "ねらい打ち" and m[1] == (n,)) for n in ("A", "F")}
    best = None
    for n in ("A", "F"):
        c = cells[n]
        if c.pinned or tries[n] >= CORNER_TRIES or not in_aim(sim, n, "x1.0"):
            continue
        key = (gap(c), -tries[n])          # 遠い方（会心で届く範囲なら）を先に
        if best is None or key > best[0]:
            best = (key, n)
    return ("ねらい打ち", (best[1],)) if best else None
CORNER_AIM_MIN_TEMP = 1200 # 角（A・F）が狙い距離なら高温のうちにねらう温度の下限（99999 で無効＝アメリーさんは 1400〜1200 は熱風で運ぶ。2026-10-26 50 局の不一致 56 件）


def _dm(sim: Sim, n: str, mk: str):
    z = turn_multiplier(sim.trait, sim.temp) * (2.0 if sim.lit_cell == n else 1.0)
    cd = dmg_candidates(sim.temp, mk)
    return math.ceil(cd[0] * z), math.ceil(cd[-1] * z)


def gap(c) -> int:
    """範囲の下限までの距離（負なら範囲内かはみ出し）。"""
    return c.lo - c.v


AIM_RESCUE_ORB = False   # 狙い距離の判定でも、再生で救えるはみ出し（上限 +1〜FL.AIM_RESCUE_MAX_OVER）を許す（2026-10-26 実験）


def in_aim(sim: Sim, n: str, mk: str) -> bool:
    """狙い距離: 非会心の最大でも上限を超えず（範囲に入るのは可）、会心（2 倍）の最小で下限に届く。
    （アメリーさん 1 局目 1100℃ の上下ねらい BD は B 残 19・D 残 18 で出目 17〜24＝非会心が範囲に入る目も許容している）"""
    c = sim.cells[n]
    if c.pinned or c.in_zone or c.v > c.hi:
        return False
    dmin, dmax = _dm(sim, n, mk)
    g = gap(c)
    over_ok = FL.AIM_RESCUE_MAX_OVER if (AIM_RESCUE_ORB and sim.temp >= FL.AIM_RESCUE_MIN_TEMP) else 0
    return 0 < g <= 2 * dmin and c.v + dmax <= c.hi + over_ok


def is_far(sim: Sim, n: str, mk: str) -> bool:
    c = sim.cells[n]
    if c.pinned or c.in_zone or c.v > c.hi:
        return False
    dmin, _ = _dm(sim, n, mk)
    return gap(c) > 2 * dmin


def safe(sim: Sim, n: str, mk: str) -> bool:
    """非会心の最大でも上限を超えない。"""
    c = sim.cells[n]
    _, dmax = _dm(sim, n, mk)
    return c.v + dmax <= c.hi


CE_CENTER = False   # 【既定OFF 2026-10-26: 72.0/73.0 と −2.75。C・E を範囲の外で待たせると再生に戻され続ける（失敗局は C 1.12・E 1.0 回/局、成功局 0.84/0.49）】
CE_CELLS = ("C", "E")
CE_PRECISE = 1.0     # 期待誤差がこれ以下で真ん中に置ける手があれば打つ
CE_REGEN_AWARE = False      # CE_CENTER の運ぶ手（②）は、打った後の温度が 200 の倍数になる時は打たない（範囲の外で止まった直後に再生で戻されるのを避ける）
CE_APPROACH_MAX_TEMP = 10 ** 9   # CE_CENTER の運ぶ手（②）を使う温度の上限（高温で運ぶと範囲の外で長く待つ）


def ce_center_move(sim: Sim):
    """C・E を真ん中へ置く 1 手。
    ① 出目の全部が上限以下で、期待誤差（|v+d−中央| の平均）が CE_PRECISE 以下の手があれば打つ（低温の てかげん 4〜5・たたく 7〜10 など）
    ② それが無く、中央までの距離が大きければ、出目の最大でも「中央 − てかげんの最小」より手前に止まる最大の技で運ぶ（最後の 1 手を精密に残す）
    ③ 距離が小さすぎて精密な手が無い（例: 残り 2）なら何もしない（範囲内の誤差として受け入れる）"""
    cells = sim.cells
    best = None
    for n in CE_CELLS:
        c = cells[n]
        if c.pinned or c.v >= c.center or c.v > c.hi:
            continue
        gap_c = c.center - c.v
        for skill, mk in (("てかげん打ち", "x0.5"), ("たたく", "x1.0"), ("2倍打ち", "x2.0"), ("熱風おろし", "x2.5"), ("3倍打ち", "x3.0")):
            if not sim.can_use(skill):
                continue
            dmin, dmax = _dm(sim, n, mk)
            cd = [math.ceil(d * turn_multiplier(sim.trait, sim.temp)) for d in dmg_candidates(sim.temp, mk)]
            if c.v + dmax > c.hi:
                continue
            err = sum(abs(c.v + d - c.center) for d in cd) / len(cd)
            cost = sim.cost_of(skill) if hasattr(sim, "cost_of") else KS.cost_of(skill, sim.trait, sim.temp)
            if err <= CE_PRECISE:
                key = (0, err, cost)
            else:
                tmin, _ = _dm(sim, n, "x0.5")
                if CE_REGEN_AWARE and (sim.temp - 50) % 200 == 0:
                    continue
                if sim.temp > CE_APPROACH_MAX_TEMP:
                    continue
                if c.v + dmax <= c.center - tmin and gap_c >= 2 * tmin:     # 精密な最後の 1 手を残して運ぶ
                    key = (1, -(dmin + dmax), cost)
                else:
                    continue
            if best is None or key < best[0]:
                best = (key, skill, (n,))
    if best and best[0][0] == 0:
        return (best[1], best[2])
    if best and best[0][0] == 1 and sim.temp >= 500:
        return (best[1], best[2])
    return None


CE_EARLY = False        # C・E を早く範囲に入れて再生の対象から外す（mori1 2026-10-26 の次の案）。2 手先までの着地の見込みで「はみ出さず範囲に入る確率」が CE_EARLY_P 以上の手を打つ
CE_EARLY_TEMP = 1300    # この温度以下で発動（全マスに手が入ってから）
CE_EARLY_P = 0.8        # 範囲に入る確率のしきい値
CE_EARLY_POS = "late"   # "late"=ねらいの前（4b の後）／"early"=熱風でそろえる前（3 の前）
CE_EARLY_P2_PEN = 0.0   # 2 手かかる計画の見込みから引く分（手数・温度・集中の節約。0 なら 1 手と同列に比べる）


def _cd_at(sim: Sim, n: str, mk: str, temp: int):
    z = turn_multiplier(sim.trait, temp) * (2.0 if (sim.lit_cell == n and temp == sim.temp) else 1.0)
    return [math.ceil(d * z) for d in dmg_candidates(temp, mk)]


_CE_SINGLES = (("てかげん打ち", "x0.5"), ("たたく", "x1.0"), ("2倍打ち", "x2.0"))


def _p_in_after(sim: Sim, n: str, g: int, temp: int) -> float:
    """残り g（下限まで）のマスに、温度 temp で 1 手打って範囲に入る最大の確率（はみ出す目のある技は使わない）。g<=0 は既に範囲内"""
    if g <= 0:
        return 1.0
    best = 0.0
    for sk, mk in _CE_SINGLES:
        cd = _cd_at(sim, n, mk, temp)
        if max(cd) > g + (sim.cells[n].hi - sim.cells[n].lo):
            continue
        best = max(best, sum(1 for d in cd if d >= g) / len(cd))
    return best


def ce_early_plan(sim: Sim):
    """C・E を「はみ出さずに範囲へ」入れる計画の最初の 1 手（1 手で入る／2 手で入る の見込みが CE_EARLY_P 以上の時）。
    候補: てかげん/たたく/2倍（単体）、上下打ち CE（両方まとめて）。同じ見込みなら消費の安い方"""
    cells = sim.cells
    cand = []
    targets = [n for n in CE_CELLS if not cells[n].pinned and not cells[n].in_zone and cells[n].v <= cells[n].hi]
    if not targets:
        return None
    t2 = sim.temp - 50
    for n in targets:
        c = cells[n]; g = gap(c); w = c.hi - c.lo
        for sk, mk in _CE_SINGLES:
            if not sim.can_use(sk):
                continue
            cd = _cd_at(sim, n, mk, sim.temp)
            if max(cd) > g + w:
                continue
            p1 = sum(1 for d in cd if d >= g) / len(cd)
            p2 = sum(_p_in_after(sim, n, g - d, t2) for d in cd) / len(cd)
            cost = KS.cost_of(sk, sim.trait, sim.temp)
            cand.append((max(p1, p2 - CE_EARLY_P2_PEN), -cost, sk, (n,)))
    if len(targets) == 2 and sim.can_use("上下打ち"):
        ok = True; ps = []
        for n in targets:
            c = cells[n]; g = gap(c); w = c.hi - c.lo
            cd = _cd_at(sim, n, "x1.2", sim.temp)
            if max(cd) > g + w:
                ok = False; break
            p1 = sum(1 for d in cd if d >= g) / len(cd)
            p2 = sum(_p_in_after(sim, n, g - d, t2) for d in cd) / len(cd)
            ps.append(max(p1, p2 - CE_EARLY_P2_PEN))
        if ok:
            cand.append((min(ps), -KS.cost_of("上下打ち", sim.trait, sim.temp), "上下打ち", tuple(targets)))
    if not cand:
        return None
    cand.sort(key=lambda x: (-x[0], -x[1]))
    p, _, sk, tg = cand[0]
    if p >= CE_EARLY_P:
        return (sk, tg)
    return None


AIM_RESET = False          # アメリーさんの「範囲内（か下限の少し下）のマスに、再生の直前にねらい」: 会心なら理想で止まり、外れてもはみ出した分を次の 200℃ の倍数の再生（−12〜16）が戻す。
                           # 60 局で 41 回（600〜1000℃、下限までの距離 −4〜+4、23 回は打った直後に再生、14 回は 1 手後）。外れた後は 弱ねらい/ねらいを打ち直す（2026-10-26）
AIM_RESET_MIN_TEMP = 400
AIM_RESET_MAX_TEMP = 1050
AIM_RESET_MAX_OVER = 11    # 非会心の最大でも上限をこれ以上は超えない（再生の最小 12 で上限以下に戻る）
AIM_RESET_RESERVE = 20     # ねらい 16 の後にこれだけ集中を残す
AIM_RESET_WAIT = 0         # 0=打った直後に 200 の倍数になる時だけ／1=1 手前も可


def aim_reset_move(sim: Sim):
    t = sim.temp
    if not (AIM_RESET_MIN_TEMP <= t <= AIM_RESET_MAX_TEMP) or not sim.can_use("ねらい打ち") or sim.focus < 16 + AIM_RESET_RESERVE:
        return None
    steps = ((t - 50) % 200) // 50          # 打った後の温度から次の 200 の倍数までの手数
    if steps > AIM_RESET_WAIT:
        return None
    cells = sim.cells
    best = None
    for n, c in cells.items():
        if c.pinned or not (c.lo - 4 <= c.v <= c.hi):
            continue
        dmin, dmax = _dm(sim, n, "x1.0")
        if c.v + dmin <= c.hi or c.v + dmax - c.hi > AIM_RESET_MAX_OVER:
            continue                          # 全部の目がはみ出し、かつ再生で戻せる量
        ratio = (c.v + dmin) / c.center
        if any(cells[m].v / cells[m].center >= ratio for m in cells if m != n and not (cells[m].lo <= cells[m].v <= cells[m].hi)):
            continue                          # 再生の対象（範囲外で割合最大）にならないなら無駄
        key = abs(c.v - c.center) + (0.5 if c.v < c.lo else 0)   # 中央から遠い（得が大きい）マスを先に
        if best is None or key > best[0]:
            best = (key, n)
    return ("ねらい打ち", (best[1],)) if best else None


def opens(sim: Sim):
    return [n for n, c in sim.cells.items() if not c.pinned and not c.in_zone and c.v <= c.hi]


def next_move(sim: Sim, rollouts: int = ROLLOUTS):
    cells = sim.cells
    t = sim.temp
    # 6. 必殺
    if sim.hissatsu_armed:
        tg = R.hissatsu_target(sim, 0)
        if tg and sim.can_use(tg[0]):
            return tg
    if sim.hissatsu_charged and not sim.hissatsu_used and not (AMELIE_OPENING and t > 1100):
        tg = R.hissatsu_target(sim)
        if tg is not None:
            return ("必殺", ())
    # 1. 開幕
    if all(c.v == 0 for c in cells.values()):
        if t < 1600:
            return ("火力上げ", ()) if sim.can_use("火力上げ") else None
        if sim.can_use("超4連打ち"):
            return ("超4連打ち", ("A", "B", "C", "D"))
    if cells["E"].v == 0 and cells["F"].v == 0 and cells["A"].v > 0:
        if sim.can_use("超4連打ち") and all(safe(sim, n, "x2.0") for n in "CDEF"):
            return ("超4連打ち", ("C", "D", "E", "F"))
        if sim.can_use("4連打ち") and all(safe(sim, n, "x1.2") for n in "CDEF"):      # C か D が会心で進みすぎた時は 4連で
            return ("4連打ち", ("C", "D", "E", "F"))
        if sim.can_use("左右打ち"):
            return ("左右打ち", ("E", "F"))
    untouched = [n for n, c in cells.items() if c.v == 0]
    # 2. 1500: 進んでいる角の側に 4連
    if t >= STEP2_TEMP - 50 and cells["E"].v > 0 and sim.can_use("4連打ち") and not getattr(sim, "_orb_step2", False):
        a, f = cells["A"], cells["F"]
        side = None
        if a.pinned and not f.pinned:
            side = ("C", "D", "E", "F")
        elif f.pinned and not a.pinned:
            side = ("A", "B", "C", "D")
        elif not a.pinned and not f.pinned:
            if STEP2_TIE == "far":
                side = ("A", "B", "C", "D") if gap(a) >= gap(f) else ("C", "D", "E", "F")
            elif STEP2_TIE == "F":
                side = ("A", "B", "C", "D") if a.v > f.v else ("C", "D", "E", "F")
            else:
                side = ("A", "B", "C", "D") if a.v >= f.v else ("C", "D", "E", "F")
        def quad_ok(q):
            return all(safe(sim, n, "x1.2") and not cells[n].in_zone and gap(cells[n]) - _dm(sim, n, "x1.2")[1] >= STEP2_MARGIN for n in q)
        if STEP2_TIE == "F8" and not a.pinned and not f.pinned:
            side = ("C", "D", "E", "F") if f.v - a.v >= 8 else ("A", "B", "C", "D")   # アメリーさん: F が A より 8 以上進んでいる時だけ CDEF
        if side and quad_ok(side):
            sim._orb_step2 = True
            return ("4連打ち", side)
        if STEP2_FALLBACK and side:
            other = ("C", "D", "E", "F") if side[0] == "A" else ("A", "B", "C", "D")
            if quad_ok(other):
                sim._orb_step2 = True
                return ("4連打ち", other)
            # アメリーさんの型: C が高ければ ななめ DE、D が高ければ 上下 CE → 左右 EF
            def pair_ok(skill, pr):
                return sim.can_use(skill) and all(not cells[n].pinned and not cells[n].in_zone and safe(sim, n, "x1.2") for n in pr)
            hi_c = cells["C"].v >= cells["D"].v and (cells["C"].in_zone or gap(cells["C"]) - _dm(sim, "C", "x1.2")[1] < STEP2_MARGIN)
            hi_d = cells["D"].v > cells["C"].v and (cells["D"].in_zone or gap(cells["D"]) - _dm(sim, "D", "x1.2")[1] < STEP2_MARGIN)
            if hi_c and pair_ok("ななめ打ち", ("D", "E")):
                sim._orb_step2 = True
                return ("ななめ打ち", ("D", "E"))
            if hi_d:
                for skill, pr in (("上下打ち", ("C", "E")), ("左右打ち", ("E", "F"))):
                    if pair_ok(skill, pr):
                        sim._orb_step2 = True
                        return (skill, pr)
            # それ以外: 範囲に入っていない・本会心でない 2 マスのうち、残りの合計が大きい組（両方とも非会心で上限を超えない）
            best = None
            for skill, pairs in (("ななめ打ち", getattr(sim, "pairs_diag", [])), ("上下打ち", sim.pairs_v), ("左右打ち", sim.pairs_lr)):
                if not sim.can_use(skill):
                    continue
                for pr in pairs:
                    a, b = cells[pr[0]], cells[pr[1]]
                    if a.pinned or b.pinned or a.in_zone or b.in_zone or not (safe(sim, pr[0], "x1.2") and safe(sim, pr[1], "x1.2")):
                        continue
                    score = gap(a) + gap(b)
                    if best is None or score > best[0]:
                        best = (score, skill, tuple(pr))
            if best:
                sim._orb_step2 = True
                return (best[1], best[2])
    # 2a'. アメリーさんの固定手順（1450→1300→1250）
    if AMELIE_OPENING and not untouched and getattr(sim, "_orb_step2", False) and 1250 <= t <= 1450:
        def heat_ok(n):
            c = cells[n]
            if c.pinned or c.in_zone or c.v > c.hi or not sim.can_use("熱風おろし"):
                return False
            _, dmax = _dm(sim, n, "x2.5")
            return gap(c) - dmax >= 0            # 熱風の最大でも下限の手前
        if t == 1450:
            # 6 手目（80 局の 4連 ABCD の後）: F が狙い距離なら ねらい F（3/3）／A が残り 18〜24 で B が D からあまり遅れていなければ ねらい A（7）／B が D に近く E が遠ければ 熱風 E（8）／それ以外は 熱風 B（31）
            a, b, d, e, f = cells["A"], cells["B"], cells["D"], cells["E"], cells["F"]
            if sim.can_use("ねらい打ち") and not f.pinned and 19 <= gap(f) <= 22 and safe(sim, "F", "x1.0"):
                return ("ねらい打ち", ("F",))
            # ねらい A（7 局）は 熱風 B（同じ A 残り 18〜21 で 7 局）と条件が重なり分けられない＝彼女自身が割れている → 多数派の 熱風 B
            if (gap(b) - gap(d)) <= 20 and gap(e) >= 60 and heat_ok("E"):
                return ("熱風おろし", ("E",))
            for n in ("B", "E"):
                if heat_ok(n):
                    return ("熱風おろし", (n,))
        def pair_far_ok(x, y, mk="x1.2"):
            cx, cy = cells[x], cells[y]
            return (sim.can_use("上下打ち") and not (cx.pinned or cy.pinned or cx.in_zone or cy.in_zone) and safe(sim, x, mk) and safe(sim, y, mk)
                    and all(gap(c) - _dm(sim, n, mk)[1] >= 0 for n, c in ((x, cx), (y, cy))))
        if t == 1300:
            # 7 手目（123 局の 80 例）: 上下ねらい BD（5）／A 残り 16〜21 で B が近ければ ねらい A（7）／B が済み・近くて E ≥70 なら 熱風 E（4）／
            # B 28〜57・D ≥35 なら 上下打ち BD（23）／B ≥57 で E・F が 32〜57 なら 左右打ち EF（5）／D ≥31・E 32〜50 なら ななめ DE（5）／B ≤26・D ≥40・F ≥39 なら 上下打ち DF（8）／B 27〜45 なら たたく B（6）
            a, b, c, d, e, f = (cells[n] for n in "ABCDEF")
            def pair_ok(skill, x, y, mk="x1.2"):
                cx, cy = cells[x], cells[y]
                return (sim.can_use(skill) and not (cx.pinned or cy.pinned or cx.in_zone or cy.in_zone) and safe(sim, x, mk) and safe(sim, y, mk)
                        and all(gap(cc) - _dm(sim, n, mk)[1] >= 0 for n, cc in ((x, cx), (y, cy))))
            if sim.can_use("上下ねらい打ち") and in_aim(sim, "B", "x1.2") and in_aim(sim, "D", "x1.2"):
                return ("上下ねらい打ち", ("B", "D"))
            if sim.can_use("ねらい打ち") and not a.pinned and 16 <= gap(a) <= 21 and gap(b) <= 28 and safe(sim, "A", "x1.0"):
                return ("ねらい打ち", ("A",))
            if (b.pinned or gap(b) <= 24) and gap(e) >= 70 and heat_ok("E"):
                return ("熱風おろし", ("E",))
            if gap(b) >= 58 and not (gap(f) >= 39 and 32 <= gap(e) <= 57) and heat_ok("B"):
                return ("熱風おろし", ("B",))          # 6 手目に B を熱風しなかった局（熱風 E・ねらい A の後）は 1300℃ で B を熱風（8 例）
            if 28 <= gap(b) <= 57 and gap(d) >= 35 and pair_ok("上下打ち", "B", "D"):
                return ("上下打ち", ("B", "D"))
            if gap(b) > 57 and 32 <= gap(e) <= 57 and 32 <= gap(f) <= 57 and pair_ok("左右打ち", "E", "F"):
                return ("左右打ち", ("E", "F"))
            if gap(d) >= 31 and 32 <= gap(e) <= 50 and pair_ok("ななめ打ち", "D", "E"):
                return ("ななめ打ち", ("D", "E"))
            if gap(b) <= 26 and gap(d) >= 40 and gap(f) >= 39 and pair_ok("上下打ち", "D", "F"):
                return ("上下打ち", ("D", "F"))
            if sim.can_use("たたく") and not b.pinned and not b.in_zone and 27 <= gap(b) <= 45 and safe(sim, "B", "x1.0") and gap(b) - _dm(sim, "B", "x1.0")[1] >= 0:
                return ("たたく", ("B",))
            if pair_ok("上下打ち", "B", "D"):
                return ("上下打ち", ("B", "D"))
            if heat_ok("E"):
                return ("熱風おろし", ("E",))
        if t == 1250:
            # 8 手目（123 局）: 熱風 E が基本（32 例中 21、D・F が狙い距離でも先に E）。B が 50 以上遠くて E が近ければ 熱風 B（たたく の後の 3 例）
            order = ("B", "E") if (gap(cells["B"]) >= 50 and gap(cells["E"]) <= 20) else ("E", "B")
            for n in order:
                if heat_ok(n):
                    return ("熱風おろし", (n,))
    # 2b. 角（A・F）が狙い距離なら高温のうちにねらい（全マスに手が入ってから）
    if untouched and t >= 1000:
        for n in untouched:
            if sim.can_use("熱風おろし") and safe(sim, n, "x2.5"):
                return ("熱風おろし", (n,))
    if t >= CORNER_AIM_MIN_TEMP and sim.can_use("ねらい打ち") and not untouched and not (AMELIE_OPENING and t > 1100):
        for n in ("A", "F"):
            if in_aim(sim, n, "x1.0"):
                return ("ねらい打ち", (n,))
    # 2c. 幅の広い角（A・F）を単体のねらいで（CORNER_FIRST）
    if CORNER_FIRST and not untouched:
        mv = corner_first_move(sim)
        if mv is not None:
            return mv
    # 4a. 縦ペアが両方とも狙い距離 → 上下ねらい（全マスに手が入ってから）
    if sim.can_use("上下ねらい打ち") and not untouched:
        for p in (("B", "D"), ("C", "E"), ("A", "C"), ("D", "F")):
            if all(in_aim(sim, n, "x1.2") for n in p):
                return ("上下ねらい打ち", p)
    # 3'. C・E を早く範囲に入れる（early 位置）
    if CE_EARLY and CE_EARLY_POS == "early" and not untouched and t <= CE_EARLY_TEMP:
        mv = ce_early_plan(sim)
        if mv is not None:
            return mv
    # 3. 遅れている B・E を熱風で相棒にそろえる（高温のうち）
    if t >= HEAT_MIN_TEMP and sim.can_use("熱風おろし"):
        best = None
        for n, mate in (("B", "D"), ("E", "C")):
            c, m = cells[n], cells[mate]
            if c.pinned or c.in_zone or not safe(sim, n, "x2.5"):
                continue
            behind = (m.v - m.lo) - (c.v - c.lo)      # 相棒より遅れている量（下限からの進みで比べる）
            _, dmax = _dm(sim, n, "x2.5")
            if gap(c) - dmax >= 0 and (behind >= MATCH_TOL or gap(c) > 2 * _dm(sim, n, "x1.2")[0] + 20):
                score = behind + gap(c)
                if best is None or score > best[0]:
                    best = (score, n)
        if best:
            return ("熱風おろし", (best[1],))
    # 3b. 遅れている角（A・F）を熱風で狙い距離の手前へ（CORNER_HEAT）
    if CORNER_HEAT and t >= CORNER_HEAT_MIN_TEMP and sim.can_use("熱風おろし") and not untouched:
        best = None
        for n in ("A", "F"):
            c = cells[n]
            if c.pinned or c.in_zone or c.v > c.hi or not is_far(sim, n, "x1.0"):
                continue
            _, dmax = _dm(sim, n, "x2.5")
            dmin_next, _ = _dm(sim, n, "x1.0")
            g = gap(c)
            # 熱風の最大でも下限に届かず、打った後に狙い距離（残り ≤ 2·dmin）に入る見込み
            if g - dmax >= 1 and g - dmax <= 2 * dmin_next + 4:
                score = g
                if best is None or score > best[0]:
                    best = (score, n)
        if best:
            return ("熱風おろし", (best[1],))
    # 4a'. 片方が狙い距離で相棒が少し遠い: 相棒をたたく/てかげんで狙い距離へ（アメリーさんの 1300 たたく D → 1100 上下ねらい BD）
    if t >= PAIR_MIN_TEMP:
        for p in (("B", "D"), ("C", "E"), ("A", "C"), ("D", "F")):
            for n, m in ((p[0], p[1]), (p[1], p[0])):
                if in_aim(sim, n, "x1.2") and not in_aim(sim, m, "x1.2") and is_far(sim, m, "x1.2"):
                    for skill, mk in (("たたく", "x1.0"), ("てかげん打ち", "x0.5"), ("2倍打ち", "x2.0")):
                        if not sim.can_use(skill):
                            continue
                        dmin, dmax = _dm(sim, m, mk)
                        g = gap(cells[m]); dmin12, _ = _dm(sim, m, "x1.2")
                        # 打った後に狙い距離（残り ≤ 2·dmin12）に入り、最大でも範囲に入らない
                        if g - dmax >= 1 and g - dmin <= 2 * dmin12 and g - dmax <= 2 * dmin12 + 4:
                            return (skill, (m,))
    # 4b. 揃った縦ペアが遠いうちは上下打ちで一緒に寄せる
    if t >= PAIR_MIN_TEMP and sim.can_use("上下打ち"):
        for p in (("B", "D"), ("C", "E"), ("A", "C"), ("D", "F")):
            if all(is_far(sim, n, "x1.2") and safe(sim, n, "x1.2") for n in p) and abs(gap(cells[p[0]]) - gap(cells[p[1]])) <= MATCH_TOL:
                # 1 打で両方が狙い距離を割り込まない
                if all(gap(cells[n]) - _dm(sim, n, "x1.2")[1] >= 0 for n in p):
                    return ("上下打ち", p)
    # 4d. 範囲内のマスに再生前提のねらい（アメリーさん型）
    if AIM_RESET:
        mv = aim_reset_move(sim)
        if mv is not None:
            return mv
    # 4c. C・E を早く範囲に入れる（late 位置）
    if CE_EARLY and CE_EARLY_POS == "late" and not untouched and t <= CE_EARLY_TEMP:
        mv = ce_early_plan(sim)
        if mv is not None:
            return mv
    # 5a. C・E は会心を取りに行かず真ん中（118）へ（mori1 2026-10-26「会心は CE 以外から取りに行って、最悪 CE は中央に入れる」）
    if CE_CENTER:
        mv = ce_center_move(sim)
        if mv is not None:
            return mv
    # 5. 残り: 狙い距離のマスにねらい（温度を下げながら）。無ければ仕上げの先読みへ
    if t >= AIM_MIN_TEMP and sim.can_use("ねらい打ち"):
        for n in sorted(opens(sim), key=lambda k: -gap(cells[k])):
            if in_aim(sim, n, "x1.0") and n not in ("C", "E"):
                return ("ねらい打ち", (n,))
    far = [n for n in opens(sim) if is_far(sim, n, "x1.0")]
    if far and t >= HEAT_MIN_TEMP:
        # まだ遠いマスがある: 汎用の稼ぎ（熱風/2倍/たたく を効率で）。ただし火力上げの上げ直しはしない（アメリーさん 24 局で開幕以外の火力上げは 0）
        keep = KG.REHEAT_NEED
        KG.REHEAT_NEED = 10 ** 9
        try:
            mv = KG.bulk_move(sim)
        finally:
            KG.REHEAT_NEED = keep
        if mv is not None and mv[0] != "火力上げ":
            return mv
    if sim.focus < 5:
        return None
    mv = FL.choose(sim, rollouts, rollout_policy=KG.rollout_generic)
    if CE_CENTER and mv and mv[0] in ("ねらい打ち", "弱ねらい打ち") and mv[1] and mv[1][0] in CE_CELLS and not cells[mv[1][0]].pinned:
        alt = ce_center_move(sim)
        if alt is not None:
            return alt
    return mv


def _apply_env():
    """実験用: 環境変数 ORB_SET="CE_EARLY=True,CE_EARLY_P=0.6" で定数を上書き（ベンチの子プロセスにも伝わる）"""
    import os
    s = os.environ.get("ORB_SET", "")
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=", 1)
        k = k.strip()
        if "." in k:                      # "FL.AIM_RESCUE=True" のように他のモジュールの定数も上書きできる
            mod, name = k.split(".", 1)
            setattr({"FL": FL, "KS": KS, "R": R, "KG": KG}[mod], name, eval(v))
        else:
            globals()[k] = eval(v)


_apply_env()


def play(seed: int, rollouts: int = ROLLOUTS) -> Sim:
    sim = make_item(ITEM, random.Random(seed))
    for _ in range(100):
        if sim.focus < 5 and not sim.hissatsu_charged:
            break
        mv = next_move(sim, rollouts)
        if mv is None:
            break
        if mv[0] == "必殺":
            sim.act("必殺")
        else:
            sim.act(mv[0], mv[1])
    return sim


def _one(seed):
    s = play(seed)
    return s.total_error() <= KS.great_of(s), s.total_error(), sum(1 for c in s.cells.values() if c.pinned), s.focus


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=200); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("-w", type=int, default=10)
    ap.add_argument("--trace", type=int, default=None)
    a = ap.parse_args()
    if a.trace is not None:
        s = make_item(ITEM, random.Random(a.trace))
        while True:
            if s.focus < 5 and not s.hissatsu_charged:
                break
            mv = next_move(s)
            if mv is None:
                break
            print(s.temp, s.focus, {n: c.v for n, c in s.cells.items()}, "->", mv)
            s.act(mv[0], mv[1]) if mv[0] != "必殺" else s.act("必殺")
        print("誤差", s.total_error(), "本会心", sum(1 for c in s.cells.values() if c.pinned))
        sys.exit()
    with Pool(a.w) as pool:
        res = pool.map(_one, range(a.seed0, a.seed0 + a.n))
    g = sum(r[0] for r in res); pins = sum(r[2] for r in res) / a.n
    import os
    print(f"## 虹色のオーブ（アメリー型 orb_policy{' ' + os.environ.get('ORB_SET') if os.environ.get('ORB_SET') else ''}） seeds{a.seed0}-{a.seed0 + a.n - 1}: 大成功 {g / a.n * 100:.1f}%  本会心 {pins:.2f}/局  平均誤差 {sum(r[1] for r in res) / a.n:.1f}  残集中 {sum(r[3] for r in res) / a.n:.0f}")
