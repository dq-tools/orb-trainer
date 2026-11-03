#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鍛冶の汎用 AI（超素材以外の武器・防具用、2026-10-20）: recipes.json の任意商材を打つ。
3〜8 マス・地金 倍半/光/再生/集中力変化/なし。

構成: 開幕（火力上げで必要量に見合う温度へ）→ 稼ぎ（全マスを「あとで狙える残り」まで安全に大きく削る。
倍半は倍ターンに大技、光は光ったマスへ、再生は 200℃ の倍数でいちばん進んだマスが戻るのをそのまま受ける）→
仕上げ（finish_lookahead.choose: 候補手×ロールアウト＋終盤の厳密解。大成功の閾値はマス数で切替）。

使い方: python kaji_generic.py 機甲兵のよろい上 -n 100 [--detail]
        from kaji_generic import play; play("輝紋章の盾", seed)
"""
import argparse, math, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import kaji_sim as KS
from kaji_sim import SKILLS, Sim, dmg_candidates, make_item, turn_multiplier, cost_of, GRADE_TABLE, is_crit_turn
import run_amatsuyu as R
import finish_lookahead as FL

NEAR_R = 60          # 全マスの残りがこれ以下になったら仕上げ（先読み）へ
BULK_FLOOR = 20      # 稼ぎでは残りをこれより小さくしない（仕上げでねらえる幅を残す）
OPEN_NEED_HI = 800   # 開幕: 必要量がこれ以上なら 2000〜2200℃ まで上げる（650→800 2026-10-25: 虹色のオーブ（必要量 704）は定石どおり 1600℃ 止まりが +11。出目は 1600 と 2200 でほぼ同じで、火力上げ 2 回ぶんの集中が浮く）
OPEN_NEED_MID = 350  #        これ以上なら 1600℃
ROLLOUTS = 32
AIM_IN_BULK = False   # 稼ぎの途中でも、会心確定の窓に入ったマス/縦ペアがあればねらう（配信者Dの棋譜: 1500℃・1300℃の上下ねらいで本会心。2026-10-20）
AIM_RESERVE = 40      # ねらいに使ってよいのは、残りの集中がこれ以上ある時
AIM_BULK_PAIR_ONLY = False   # 稼ぎ中のねらいは上下ねらい（2マス）だけ
REHEAT_NEED = 250    # 温度 1300 以下でまだ必要量がこれより大きければ火力上げで上げ直す（虹色のオーブの定石は 1300 で上げ直さない → 400 を試す 2026-10-25）
TEMP_VAL_HI = 1.7    # 必要量が大きい時の 50℃ の価値（集中換算）
TEMP_VAL_LO = 0.4
BAIHAN_PHASE = False    # 【既定OFF・コード残置 2026-10-22】倍半: 倍ターン（1600/1200/800）の手前で縦ペアを「外れても再挑戦できる窓」へ運び、倍ターンに上下ねらい
                        # （作者 2026-10-22「倍のときに上下ねらいを決めて、外れてももう一度ねらえるように、倍ターンの手前で調整する」）
BAIHAN_TURNS = (1600, 1200, 800)
# v1（haul_into 流用）200 局×2 群: ツメ 53.0/59.0→53.0/53.0、レシーバー 51.0/50.0→52.5/51.5（±0）。
# v3（_haul_pair_step: 上下打ちで両方を進め、相手のマスが残り手数で届くか見て配分。1600 の上下ねらい 10→20/40 局）: ツメ 56.5/62.0（+3.3）、レシーバー 42.5/46.5（−6）→ 商材で割れるので既定 OFF。1600 の上下ねらいが決まるのは 40 局中 11 局で、開幕の後 B（必要量 424）が遠く A が近いため
# 7 手で (B,D) を窓に入れられない。配信者Dの第 6 局の 1600 上下ねらい BD は熱風 B の会心（+126）に支えられていた。倍ターンごとのペア割り当て（1200 に (A,C)、800 に (B,D) 等）が次の案
PHASE_IN_ROLLOUT = True   # 先読みのロールアウトでも倍ターンの調整を使う（実際の方針と評価を揃える）
# 【採用 2026-10-24】200 局×2 群: こて 48.5/52.0→60.5/56.0、ラッシュバンド 48.5/53.5→63.0/55.5（開幕+みだれ+会心ターンのねらい。要素ごとの切り分けは CLAUDE.md）
SHUCHU_OPENING = True   # 集中力変化の開幕台本（作者本人 こて 5 局・ラッシュバンド 5 局、2026-10-24）:
                        # 火力上げ×2 → 1600（半額）で一番大きい技（2x2 があれば超4連、無ければ必要量最大のマスに 3倍）→ 火力上げ×2 → 2150 熱風 → 2000（半額）で同じ大技
MIDARE_BULK = True      # どの地金でも、全マスが遠い間はみだれ打ちを稼ぎの候補に（作者 グリーブ 8 局。2026-10-25。効果はベンチで確認）
MIDARE_BULK_MAX_CELLS = 4   # マス数がこれ以下の商材だけ（4 打が集中するほど効率がよい。6 マス以上は分散して弱い想定）
SHUCHU_MIDARE = True    # 集中力変化（2x2 の無い商材）: 全マスが遠い間はみだれ打ち（12 で 4 打。こて 5 局で中盤の主力）
SHUCHU_CRIT_AIM = True  # 集中力変化: 会心ターン（400n+200）に「会心なら理想に届き、非会心でははみ出さない」マスへ 熱風/ねらい/上下ねらい
SHUCHU_COOL_TO_200 = False  # 【既定OFF 2026-10-24: バンド −4.5/−4.5、こて −2.5/±0。仕上げは先読みに任せる方がよい】集中力変化の仕上げ: 850℃以下で全マスが近ければ冷やし込みで 200（会心ターン）に合わせてからねらう（600 は飛ばす）
BAIHAN_OPENING = True   # 倍半の開幕台本（配信者D 鋼神のツメ SUNXib_q2gE 第6局、2026-10-22）:
                        # 火力上げ×2 → 1600（倍）で必要量最大のマスに熱風 → 火力上げ×2 → 2050 で 4連（→2000）→ 2000（倍）で超4連


def great_limit(sim: Sim) -> int:
    return KS.great_of(sim)


def need_total(sim: Sim) -> int:
    return sum(max(0, c.lo - c.v) for c in sim.cells.values())


def _zc(sim: Sim, name: str) -> float:
    return turn_multiplier(sim.trait, sim.temp) * (2.0 if sim.lit_cell == name else 1.0)


def _dm(sim: Sim, name: str, mult_key: str):
    z = _zc(sim, name)
    c = dmg_candidates(sim.temp, mult_key)
    return math.ceil(c[0] * z), math.ceil(c[-1] * z)


def target_temp(sim: Sim) -> int:
    """開幕の目標温度（必要量で決める）。倍半は倍ターン（1600/2000）に大技を置きたいので 1600 を経由する。"""
    need = need_total(sim)
    if need >= OPEN_NEED_HI:
        return 2200 if sim.trait != "倍半" else 2050   # 倍半は 2050 に置いて安い一撃で 2000（倍）に乗せる
    if need >= OPEN_NEED_MID:
        return 1600
    return 1300


def bulk_move(sim: Sim):
    """稼ぎの1手。None = 稼ぎ終了（仕上げへ）。"""
    cells = sim.cells
    opens = [n for n, c in cells.items() if not c.in_zone and c.r > 0]
    if not opens:
        return None
    far = [n for n in opens if cells[n].r > NEAR_R]
    if not far:
        return None
    temp = sim.temp
    z = turn_multiplier(sim.trait, temp)
    if BAIHAN_OPENING:
        mv = baihan_opening(sim)
        if mv is not None:
            return mv
    if SHUCHU_OPENING:
        mv = shuchu_opening(sim)
        if mv is not None:
            return mv
    # 開幕: 必要量に見合う温度まで火力上げ（全マス未着手〜序盤のみ）
    if all(c.v == 0 for c in cells.values()) and temp < target_temp(sim) and sim.can_use("火力上げ"):
        return ("火力上げ", ())
    # 倍半: 倍ターンの 50℃ 上（2050/1650）なら安い一撃で倍ターンに乗せる
    if sim.trait == "倍半" and temp % 400 == 50 and temp >= 1250:
        for skill in ("たたく", "てかげん打ち"):
            if not sim.can_use(skill):
                continue
            for n in sorted(far, key=lambda n: -cells[n].r):
                dmin, dmax = _dm(sim, n, SKILLS[skill]["mult"])
                if cells[n].r - dmax >= BULK_FLOOR:
                    return (skill, (n,))
    # 温度切れ: まだ必要量が大きいのに温度が低い → 火力上げ（高温のうちに上げ直す方が1打あたりの威力が大きい）
    need = need_total(sim)
    if sim.can_use("火力上げ") and sim.focus >= 30 and (
            (temp <= 1300 and need > REHEAT_NEED) or (temp <= 600 and need > 120)) and not (sim.trait == "倍半" and temp % 400 == 0):
        return ("火力上げ", ())
    halved = sim.trait == "倍半" and z <= 0.5
    # 会心確定の窓に入ったマスを、高温のうちにねらう（配信者Dは遠いマスが残っていても窓に入ったペアから上下ねらいで本会心を取る）
    if AIM_IN_BULK and not halved and sim.focus >= AIM_RESERVE:
        mv = aim_in_bulk(sim)
        if mv is not None:
            return mv
    # 温度も資源: 火力上げ 10 で 300℃ → 50℃ ≈ 1.7 集中。必要量が大きいほど温度を惜しむ（熱風の −150 は 3 打ぶん）
    temp_val = TEMP_VAL_HI if need > 300 else TEMP_VAL_LO
    # 候補: 面技（2x2）→ ペア → 単体。全対象マスが floor を割らない範囲で「ダメージ合計/消費」最大
    best = None

    def consider(skill, tg):
        if not sim.can_use(skill):
            return
        mk = SKILLS[skill]["mult"]
        tot = 0
        for n in tg:
            c = cells[n]
            if c.in_zone or c.r <= 0:
                return
            dmin, dmax = _dm(sim, n, mk)
            if c.r - dmax < BULK_FLOOR:
                return
            tot += (dmin + dmax) / 2
        cost = cost_of(skill, sim.trait, temp) + temp_val * (-SKILLS[skill]["dtemp"]) / 50
        score = tot / cost
        if halved and cost > 6:
            score *= 0.6   # 半減ターンに高い技は損
        nonlocal best
        if best is None or score > best[0]:
            best = (score, skill, tuple(tg))

    use_midare = (SHUCHU_MIDARE and sim.trait == "集中力変化" and not sim.quads) or (MIDARE_BULK and len(cells) <= MIDARE_BULK_MAX_CELLS)
    if use_midare and sim.can_use("みだれ打ち") and len(far) >= 2:
        # みだれ打ち: 4 打が全マスからランダム（同じマスに重なる）。全マスが 2 打ぶん以上遠い時だけ
        # （作者 グリーブ 8 局: 2200℃ からみだれ×6〜8 で削る。7 で 4 打＝超4連 BCD の 18 より集中効率がよい。2026-10-25）
        dmaxs = {n: _dm(sim, n, SKILLS["みだれ打ち"]["mult"])[1] for n in opens}
        if all(cells[n].r - 2 * dmaxs[n] >= BULK_FLOOR for n in opens) and len(opens) == len(cells):
            tot = 4 * sum((_dm(sim, n, SKILLS["みだれ打ち"]["mult"])[0] + dmaxs[n]) / 2 for n in opens) / len(opens)
            cost = cost_of("みだれ打ち", sim.trait, temp) + temp_val * (-SKILLS["みだれ打ち"]["dtemp"]) / 50
            best = (tot / cost, "みだれ打ち", ())
    for q in sim.quads:
        consider("超4連打ち", q); consider("4連打ち", q)
    for p in sim.pairs_v:
        consider("上下打ち", p)
    for p in sim.pairs_lr:
        consider("左右打ち", p)
    for p in sim.pairs_diag:
        consider("ななめ打ち", p)
    for n in far:
        for skill in ("熱風おろし", "3倍打ち", "2倍打ち", "たたく", "てかげん打ち"):
            consider(skill, (n,))
    if best is None:
        return None
    return (best[1], best[2])


def shuchu_opening(sim: Sim):
    """集中力変化の開幕台本。該当しない局面は None。進み具合は盤面（必要量に対する進み）で判定（ロールアウトのクローンは log が空のため）。"""
    if sim.trait != "集中力変化":
        return None
    cells = sim.cells
    need0 = sum((c.lo + c.hi) // 2 for c in cells.values())
    if need0 < OPEN_NEED_HI:
        return None
    temp = sim.temp
    prog = sum(c.v for c in cells.values()) / need0

    def big():
        if sim.quads and sim.can_use("超4連打ち"):
            q = _safe_quad(sim, "超4連打ち")
            if q:
                return ("超4連打ち", q)
        if sim.can_use("3倍打ち"):
            for n in sorted(cells, key=lambda k: -cells[k].r):
                if not cells[n].in_zone and cells[n].r - _dm(sim, n, SKILLS["3倍打ち"]["mult"])[1] >= BULK_FLOOR:
                    return ("3倍打ち", (n,))
        return None
    if prog == 0:
        if temp < 1600:
            return ("火力上げ", ()) if sim.can_use("火力上げ") else None
        if temp == 1600:
            return big()
        return None
    if prog < 0.2 and temp in (1550, 1850) and sim.can_use("火力上げ"):
        return ("火力上げ", ())
    if prog < 0.2 and temp == 2150 and sim.can_use("熱風おろし"):
        for n in sorted(cells, key=lambda k: -cells[k].r):
            if not cells[n].in_zone and cells[n].r - _dm(sim, n, SKILLS["熱風おろし"]["mult"])[1] >= BULK_FLOOR:
                return ("熱風おろし", (n,))
    if prog < 0.35 and temp == 2000:
        return big()
    return None


def shuchu_crit_aim(sim: Sim):
    """集中力変化の会心ターン: 会心なら理想に届き非会心でもはみ出さないマスへ、上下ねらい > ねらい > 熱風（安い順ではなく届く順）。無ければ None。"""
    if sim.trait != "集中力変化" or not is_crit_turn(sim.trait, sim.temp):
        return None
    cells = sim.cells
    z = turn_multiplier(sim.trait, sim.temp)

    def ok(n, mk):
        c = cells[n]
        if c.in_zone or c.r <= 0 or c.pinned:
            return False
        cd = dmg_candidates(sim.temp, mk)
        dmin, dmax = math.ceil(cd[0] * z), math.ceil(cd[-1] * z)
        return dmax <= c.hi - c.v and 2 * dmin >= c.r
    if sim.can_use("上下ねらい打ち"):
        for p in sim.pairs_v:
            if all(ok(n, "x1.2") for n in p):
                return ("上下ねらい打ち", p)
    order = (("熱風おろし", "x2.5"), ("ねらい打ち", "x1.0"), ("弱ねらい打ち", "x0.5"))
    if sim.temp <= 600:
        order = (("ねらい打ち", "x1.0"), ("弱ねらい打ち", "x0.5"), ("熱風おろし", "x2.5"))   # 低温では温度を落とす熱風を後回し（200 で熱風→50℃ になる）
    for skill, mk in order:
        if not sim.can_use(skill):
            continue
        for n in sorted(cells, key=lambda k: -cells[k].r):
            if ok(n, mk):
                return (skill, (n,))
    return None


def shuchu_cool(sim: Sim):
    """集中力変化の仕上げ: 800/500℃で全マスが近い（残り ≤ NEAR_R）・未完のマスがある → 冷やし込みで 200（会心ターン）へ（600 は飛ばす）。"""
    if sim.trait != "集中力変化" or sim.temp not in (800, 500) or not sim.can_use("冷やし込み"):
        return None
    opens = [n for n, c in sim.cells.items() if not c.in_zone and c.r > 0 and not c.pinned]
    if not opens or any(sim.cells[n].r > NEAR_R for n in opens):
        return None
    return ("冷やし込み", ())


def _safe_quad(sim: Sim, skill: str):
    """全マスが floor を割らない 2x2 のうち必要量の合計が最大のもの。無ければ None。"""
    cells = sim.cells
    mk = SKILLS[skill]["mult"]
    best = None
    for q in sim.quads:
        if any(cells[n].in_zone or cells[n].r <= 0 or cells[n].r - _dm(sim, n, mk)[1] < BULK_FLOOR for n in q):
            continue
        tot = sum(cells[n].r for n in q)
        if best is None or tot > best[0]:
            best = (tot, q)
    return best[1] if best else None


def baihan_opening(sim: Sim):
    """倍半の開幕台本（配信者Dの型）。該当しない局面は None（通常の稼ぎへ）。
    1000 火力上げ×2 → 1600（倍ターン）で必要量最大のマスに熱風（2倍=86 前後）→ 火力上げ×2 → 2050 で 4連打ち（全マス安全なら。−50 で 2000 に乗る）
    → 2000（倍ターン）で超4連打ち（全マス安全なら）。必要量 650 未満の商材には使わない。"""
    if sim.trait != "倍半" or need_total(sim) < OPEN_NEED_HI or not sim.quads:
        return None
    cells = sim.cells
    temp = sim.temp
    touched = [n for n, c in cells.items() if c.v > 0]
    if not touched:
        if temp < 1600:
            return ("火力上げ", ()) if sim.can_use("火力上げ") else None
        if temp == 1600 and sim.can_use("熱風おろし"):
            n = max(cells, key=lambda k: cells[k].r)
            if cells[n].r - _dm(sim, n, SKILLS["熱風おろし"]["mult"])[1] >= BULK_FLOOR:
                return ("熱風おろし", (n,))
        return None
    if len(touched) == 1 and 1400 <= temp < 2050 and temp % 400 != 0:
        return ("火力上げ", ()) if sim.can_use("火力上げ") else None
    if len(touched) == 1 and temp == 2050 and sim.can_use("4連打ち"):
        q = _safe_quad(sim, "4連打ち")
        if q:
            return ("4連打ち", q)
    if temp == 2000 and len(touched) <= len(cells) and sim.can_use("超4連打ち") and sum(1 for c in cells.values() if c.v > 0) <= len(cells):
        # 2000 の倍ターン: 超4連（ここは開幕以外でも、全マス安全なら最も効率がよい）
        q = _safe_quad(sim, "超4連打ち")
        if q and all(cells[n].v <= cells[n].hi * 0.4 for n in q):
            return ("超4連打ち", q)
    return None


def _strict_win(sim: Sim, n: str, temp=None):
    """上下ねらいの窓（残り r の範囲）。下限は「非会心でもゾーンに入らず、次の倍ターンにもう一度ねらえる」位置（aim_window_cell）。"""
    return KS.aim_window_cell(sim, n, "x1.2", temp)


def _baihan_pair(sim: Sim, temp: int):
    """倍ターン temp で上下ねらいを撃つ縦ペア: 両マスが開いていて窓の下限を割っていないもののうち、窓の中央に近いもの。"""
    reserved = R.hissatsu_reserved(sim)
    best = None
    for p in sim.pairs_v:
        if reserved & set(p):
            continue
        ok = True; score = 0
        for n in p:
            c = sim.cells[n]
            lo, hi = _strict_win(sim, n, temp)
            if c.in_zone or c.r <= 0 or c.pinned or c.r < lo:
                ok = False; break
            score += c.r
        if ok and (best is None or score > best[0]):
            best = (score, p)
    return best[1] if best else None


def _haul_pair_step(sim: Sim, pair, T: int, slots: int):
    """ペアの 2 マスを T までの slots 手で窓 [lo,hi] に運ぶ 1 手（配信者Dの型: 上下打ちで両方を進め、熱風は相手のマスの残り手数を見てから）。
    候補: 上下打ち（1 手で両方）・熱風（3 手ぶん）・3倍・2倍・たたく・てかげん。
    採用条件: 打った後に両マスが「残り手数で窓に届く」（不足 ≤ 残り手数 × 2倍打ちの最大）こと、floor を割らないこと。その中でダメージ/消費が最大。"""
    cells = sim.cells
    z = turn_multiplier(sim.trait, sim.temp)
    win = {n: _strict_win(sim, n, T) for n in pair}
    deficit = {n: cells[n].r - win[n][1] for n in pair}      # 窓の上限までの不足（0 以下なら窓かその下）
    if all(d <= 0 for d in deficit.values()):
        return None
    per_slot = max(1, math.ceil(dmg_candidates(T, SKILLS["たたく"]["mult"])[-1]))   # 最後の 1 手で確実に運べる目安（T のたたくの最大）
    # 「最後の 1 手はたたく、前は 2倍打ちの平均」にすると ツメ 49.0/58.5・レシーバー 49.0/48.5 と悪化（2026-10-22）→ たたく基準のまま

    def capacity(rest):
        return 0 if rest <= 0 else rest * per_slot
    best = None; fallback = None

    def consider(skill, tg):
        if not sim.can_use(skill):
            return
        dt = -SKILLS[skill]["dtemp"] // 50
        if dt > slots:
            return
        mk = SKILLS[skill]["mult"]
        tot = 0; new_def = dict(deficit)
        for n in tg:
            dmin, dmax = _dm(sim, n, mk)
            if cells[n].r - dmax < win[n][0]:      # 窓の下限（外れても再挑戦できる位置）を割る
                return
            if deficit[n] <= 0:
                return                            # もう窓にいるマスは触らない
            tot += (dmin + dmax) / 2
            new_def[n] = deficit[n] - (dmin + dmax) / 2
        rest = slots - dt
        cost = cost_of(skill, sim.trait, sim.temp)
        nonlocal best, fallback
        if all(d <= capacity(rest) for d in new_def.values()):
            score = tot / cost
            if best is None or score > best[0]:
                best = (score, skill, tuple(tg))
        else:
            # 残り手数で届かない時の次善: 不足の合計が最小になる手（同じなら安い手）
            short = sum(max(0.0, d) for d in new_def.values())
            key = (short, cost)
            if fallback is None or key < fallback[0]:
                fallback = (key, skill, tuple(tg))

    if pair in sim.pairs_v:
        consider("上下打ち", pair)
    for n in pair:
        for skill in ("熱風おろし", "3倍打ち", "2倍打ち", "たたく", "てかげん打ち"):
            consider(skill, (n,))
    if best:
        return (best[1], best[2])
    return (fallback[1], fallback[2]) if fallback else None


def baihan_fire(sim: Sim):
    """倍ターン当日: 両マスが窓にある縦ペアへ上下ねらい（厳しい窓→通常の窓の順）。無ければ None。"""
    if not sim.can_use("上下ねらい打ち"):
        return None
    reserved = R.hissatsu_reserved(sim)
    for strict in (True, False):
        best = None
        for p in sim.pairs_v:
            if reserved & set(p):
                continue
            ok = True; score = 0
            for n in p:
                c = sim.cells[n]
                lo, hi = _strict_win(sim, n) if strict else KS.aim_window(sim, "x1.2")
                if c.in_zone or c.r <= 0 or c.pinned or not (lo <= c.r <= hi):
                    ok = False; break
                score += c.r
            if ok and (best is None or score > best[0]):
                best = (score, p)
        if best:
            return ("上下ねらい打ち", best[1])
    return None


def baihan_phase_move(sim: Sim):
    """倍半の中盤（開幕台本の後）: 次の倍ターン T に向けて、ペアを窓へ運び（他の遠いマスは次の倍ターンの窓を割らない範囲で運ぶ）、
    T 当日は上下ねらい。打つ手が無ければ None（通常の稼ぎ・仕上げへ）。probe で 1 手だけ取り出す。"""
    if sim.trait != "倍半" or sim.temp < BAIHAN_TURNS[-1] or sim.hissatsu_armed:
        return None
    cells = sim.cells
    if all(c.v == 0 for c in cells.values()):
        return None   # 開幕中
    opens = [n for n, c in cells.items() if not c.in_zone and c.r > 0 and not c.pinned]
    if not any(cells[n].r > NEAR_R for n in opens):
        return None   # 全部近い → 仕上げ（先読み）へ
    T = next((t for t in BAIHAN_TURNS if sim.temp >= t), None)
    if T is None:
        return None
    if sim.temp == T:
        mv = baihan_fire(sim)
        if mv:
            return mv
        probe = sim.clone(random.Random(0), hide_ideal=True)
        probe.plan = {"p12": None, "p8": None, "p8_fixed": True}
        n0 = len(probe.log)
        R.fire_turn(probe)   # 1200/800 はねらい系、それ以外は倍ダメージで大きく安全に削る
        return (probe.log[n0][0], tuple(probe.log[n0][1])) if len(probe.log) > n0 else None
    probe = sim.clone(random.Random(0), hide_ideal=True)
    n0 = len(probe.log)
    slots = (probe.temp - T) // 50
    z = turn_multiplier(probe.trait, probe.temp)
    doubled, halved = z >= 2.0, z <= 0.5
    pair = _baihan_pair(probe, T)
    pair_cells = set(pair) if pair else set()
    T2 = T - 400
    # 1) ペアのマスを T の窓へ運ぶ（両マスが残り手数で届くように配分。下限は厳しい窓＝外れても再挑戦できる位置）
    if pair:
        mv = _haul_pair_step(probe, pair, T, slots)
        if mv is not None:
            return mv
    # 2) ペア外の遠いマスを運ぶ（次の倍ターン T2 の窓の下限を割らない）
    for n in sorted(opens, key=lambda k: -probe.cells[k].r):
        if n in pair_cells:
            continue
        floor = max(12, _strict_win(probe, n, T2)[0]) if T2 >= 400 else 12
        if probe.cells[n].r > max(24, floor + 8) and R.haul_into(probe, n, floor, slots):
            return (probe.log[n0][0], tuple(probe.log[n0][1]))
    # 3) 温度だけ消化: 熱風（安全な最大）→ 冷やし込み → ねらい（通常ターン・窓のマス）→ てかげん
    if probe.temp - 150 >= T and probe.can_use("熱風おろし"):
        for n in sorted(opens, key=lambda k: -probe.cells[k].r):
            if n in pair_cells:
                continue
            if KS.safe_max(probe, n, "x2.5"):
                return ("熱風おろし", (n,))
    if probe.temp - 300 >= T and probe.can_use("冷やし込み"):
        return ("冷やし込み", ())
    if not halved and probe.can_use("ねらい打ち"):
        lo10, hi10 = KS.aim_window(probe, "x1.0")
        for n in opens:
            if n not in pair_cells and lo10 <= probe.cells[n].r <= hi10:
                return ("ねらい打ち", (n,))
    if probe.can_use("てかげん打ち"):
        for n in sorted(opens, key=lambda k: -probe.cells[k].r):
            if n not in pair_cells and KS.safe_max(probe, n, "x0.5"):
                return ("てかげん打ち", (n,))
    return None


def aim_in_bulk(sim: Sim):
    """窓に入った縦ペア → 上下ねらい、単体 → ねらい打ち（非会心ではみ出さず、会心なら最小値でも理想に届く）。無ければ None。"""
    z = turn_multiplier(sim.trait, sim.temp)
    cells = sim.cells

    def in_win(n, mk):
        c = cells[n]
        if c.pinned or c.in_zone or c.r <= 0:
            return False
        zc = z * (2.0 if sim.lit_cell == n else 1.0)
        cd = dmg_candidates(sim.temp, mk)
        dmin, dmax = math.ceil(cd[0] * zc), math.ceil(cd[-1] * zc)
        return dmax <= c.hi - c.v and 2 * dmin >= c.r + 2
    if sim.can_use("上下ねらい打ち"):
        best = None
        for a, b in sim.pairs_v:
            if in_win(a, "x1.2") and in_win(b, "x1.2"):
                sc = cells[a].r + cells[b].r
                if best is None or sc > best[0]:
                    best = (sc, (a, b))
        if best:
            return ("上下ねらい打ち", best[1])
    if sim.can_use("ねらい打ち") and not AIM_BULK_PAIR_ONLY:
        for n in sorted(cells, key=lambda n: -cells[n].r):
            if in_win(n, "x1.0"):
                return ("ねらい打ち", (n,))
    return None


def rollout_generic(s: Sim):
    """ロールアウト方針: 稼ぎ → run_amatsuyu.finish（汎用の仕上げヒューリスティック）。"""
    for _ in range(40):
        if s.focus < 5:
            return
        mv = baihan_phase_move(s) if (BAIHAN_PHASE and PHASE_IN_ROLLOUT) else None
        if mv is None or not s.can_use(mv[0]):
            mv = bulk_move(s)
        if mv is None:
            break
        s.act(mv[0], mv[1])
    R.finish(s)


def next_move(sim: Sim, rollouts: int = ROLLOUTS):
    if sim.hissatsu_armed:
        tg = R.hissatsu_target(sim, 0)
        if tg and sim.can_use(tg[0]):
            return tg
    if sim.hissatsu_charged and not sim.hissatsu_used:
        tg = R.hissatsu_target(sim)
        if tg is not None:
            return ("必殺", ())
    if BAIHAN_OPENING:
        mv = baihan_opening(sim)
        if mv is not None:
            return mv
    if SHUCHU_OPENING:
        mv = shuchu_opening(sim)
        if mv is not None:
            return mv
    if SHUCHU_CRIT_AIM and sim.focus >= 5:
        mv = shuchu_crit_aim(sim)
        if mv is not None:
            return mv
    if SHUCHU_COOL_TO_200:
        mv = shuchu_cool(sim)
        if mv is not None:
            return mv
    if BAIHAN_PHASE and sim.focus >= 5:
        mv = baihan_phase_move(sim)
        if mv is not None and sim.can_use(mv[0]):
            return mv
    mv = bulk_move(sim)
    if mv is not None:
        return mv
    if sim.focus < 5:
        return None
    return FL.choose(sim, rollouts, rollout_policy=rollout_generic)


def play(item: str, seed: int, rollouts: int = ROLLOUTS) -> Sim:
    sim = make_item(item, random.Random(seed))
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


def is_great(sim: Sim) -> bool:
    return sim.total_error() <= great_limit(sim)


def apply_sets(sets):
    """--set=mod.NAME=val を各ワーカーで反映（kaji_bench と同じ書式）。"""
    import importlib
    for x in sets or ():
        mod, rest = x.split(".", 1); name, val = rest.split("=", 1)
        m = importlib.import_module(mod)
        if "[" in name or "." in name:
            exec(f"{mod}.{name} = {val}", {mod: m})
        else:
            setattr(m, name, eval(val))
            if mod == "kaji_generic":
                globals()[name] = eval(val)   # spawn のワーカーでは本体が __mp_main__ として動くので、こちらの名前空間にも入れる


def _one(a):
    item, seed, rollouts = a[:3]
    apply_sets(a[3] if len(a) > 3 else ())
    s = play(item, seed, rollouts)
    return (is_great(s), s.total_error(), s.focus, sum(1 for c in s.cells.values() if c.pinned),
            sum(1 for c in s.cells.values() if c.v > c.hi), sum(1 for c in s.cells.values() if c.v < c.lo), s.temp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+"); ap.add_argument("-n", type=int, default=100); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS); ap.add_argument("-w", type=int, default=10); ap.add_argument("--detail", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="mod.NAME=val（例 kaji_generic.AIM_IN_BULK=True）")
    a = ap.parse_args()
    apply_sets(a.set)
    from multiprocessing import Pool
    for item in a.items:
        t = time.time()
        with Pool(a.w) as pool:
            res = pool.map(_one, [(item, sd, a.rollouts, a.set) for sd in range(a.seed0, a.seed0 + a.n)])
        g = sum(r[0] for r in res); n = len(res)
        s0 = make_item(item, random.Random(0))
        print(f"## {item}（{len(s0.cells)}マス・{s0.trait or 'なし'}・大成功≤{great_limit(s0)}） seeds{a.seed0}-{a.seed0+n-1} {' '.join(a.set) or '基準'}: 大成功 {g/n*100:.1f}%  "
              f"平均誤差 {sum(r[1] for r in res)/n:.1f}  残集中 {sum(r[2] for r in res)/n:.0f}  本会心 {sum(r[3] for r in res)/n:.2f}/局  "
              f"はみ出しあり {sum(1 for r in res if r[4])}  たりずあり {sum(1 for r in res if r[5])}  ({time.time()-t:.0f}s)", flush=True)
        if a.detail:
            s = play(item, a.seed0, a.rollouts)
            for sk, tg, temp, hits in s.log:
                print(f"   {sk}{''.join(tg)} →{temp}℃ {[(h[0], h[1], h[-1]) for h in hits]}")
            for n_, c in s.cells.items():
                print(f"   {n_}: {c.v} [{c.lo}-{c.hi}] 理想{c.ideal_true} 誤差{c.error}{' ★' if c.pinned else ''}")
            print(f"   誤差{s.total_error()} 残集中{s.focus} → {KS.grade(s, s.total_error())}")
