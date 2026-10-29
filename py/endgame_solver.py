#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""終盤の厳密解（商材非依存）: 残集中が少ない局面で、終局までの期待値計算（expectimax）を
メモ化して行い、大成功確率が最大の手を返す。

- 状態 = (温度, 集中, 各マスの(値, 信念マスク), 必殺状態)
- 信念マスク = 隠し理想値(中央±2の5通り)のうち残っている候補。本会心で判明(1通り)、
  会心が出たのに止まらなければ「届く範囲に無かった」分を除外。非会心では変わらない
- 1手の結果分岐 = ダメージ7候補 × 会心/非会心（ペア技は各マス独立）。ロールアウト無しの厳密計算
- 終局値 = P(誤差合計 <= 大成功閾値)。各マスの誤差分布(信念上)を畳み込んで求める
"""
import math

from kaji_sim import (ASSUMPTIONS, SKILLS, Sim, cost_of, dmg_candidates, is_crit_turn,
                      turn_multiplier)
import kaji_sim as KS

ENDGAME_FOCUS = 50      # この集中以下で厳密解に切り替える
MAX_ACTIVE = 4          # 未確定マスがこれ以下のとき
NEAR_R = 60             # 未確定マスの残りがこれ以下の時だけ適用（遠いマスがあると状態爆発）
BULK_R = 30             # 残りがこれより大きいマスへは運搬技(熱風/3倍/2倍)のみ許可（枝刈り）
COARSE_BELIEF = False   # True: 偽会心で理想値候補を絞らない（状態数が大幅減。光特性で有効）
MAX_NODES = 60000       # 状態数の上限（超えたら先読みにフォールバック）。6万で1手最大数秒、上限超えは終盤局面の2〜3割


class TooBig(Exception):
    pass
REGEN = True            # 再生の地金: 行動後の温度が 200℃ の倍数なら「範囲外で割合が最大のマスが 12〜16 戻る」を分岐に入れ、
                        # 戻せる見込みのある「はみ出す手」も候補にする（mori1 2026-10-29「厳密解を頑張りましょう」。
                        # きっかけ: 950℃・集中 38 で B 142/140-148 にねらい→冷やし込み→600℃ の再生で 44%→54% の局面を見落としていた）
REGEN_OVER_MAX = 16     # はみ出す手を候補にする上限（非会心の最大の目で上限をこれ以上超える手は入れない。戻りの最大 16 で範囲に戻れる分だけ）
K_OFFSETS = (-2, -1, 0, 1, 2)
FULL_MASK = 0b11111
SINGLE = ("たたく", "てかげん打ち", "2倍打ち", "3倍打ち", "熱風おろし", "ねらい打ち", "弱ねらい打ち")
PAIR = ("上下ねらい打ち", "上下打ち", "左右打ち", "ななめ打ち")


class Solver:
    def __init__(self, sim: Sim):
        self.trait = sim.trait
        self.great = KS.GRADE_TABLE.get(len(sim.cells), (ASSUMPTIONS["great_success_max_err"],))[0]  # マス数で切替（2026-10-20）
        self.names = list(sim.cells.keys())
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.lo = [sim.cells[n].lo for n in self.names]
        self.hi = [sim.cells[n].hi for n in self.names]
        # 本会心済みのマスは理想値＝現在値で確定。中央±2の外（例: 超あまつゆA 180-190 の下端180）で
        # 止まることがあるので、中央からのずれではなく現在値そのものを中心にする（2026-10-17 実配信で判明）
        self.center = [sim.cells[n].v if sim.cells[n].pinned else sim.cells[n].ideal for n in self.names]
        # マスごとの理想値候補（中心からのずれ）。本会心済みは (0,) のみ。ASSUMPTIONS["ideal_spread"] に従う
        self.koffs = [(0,) if sim.cells[n].pinned else KS.ideal_offsets(sim.cells[n].lo, sim.cells[n].hi)
                      for n in self.names]
        self.pairs = {"上下ねらい打ち": sim.pairs_v, "上下打ち": sim.pairs_v,
                      "左右打ち": sim.pairs_lr, "ななめ打ち": sim.pairs_diag}
        self.quads = list(sim.quads)
        self.regen = REGEN and sim.trait in KS.MODORI_TRAITS
        self.zc = [(sim.cells[n].lo + sim.cells[n].hi) // 2 for n in self.names]   # 再生の対象の決め方（kaji_sim: v / center）に使う中央
        self.memo = {}
        self.nodes = 0

    def _regen_possible(self, t2, f2):
        """この行動の後（温度 t2・集中 f2）に再生を起こせるか: t2 が 200 の倍数、または 冷やし込み／火力上げ 1 回で 200 の倍数に乗せられる"""
        if not self.regen:
            return False
        if t2 > 0 and t2 % 200 == 0:
            return True
        if t2 >= 350 and t2 - 300 > 0 and (t2 - 300) % 200 == 0 and f2 >= cost_of("冷やし込み", self.trait, t2):
            return True
        if t2 <= 400 and (t2 + 300) % 200 == 0 and f2 >= cost_of("火力上げ", self.trait, t2):
            return True
        return False

    def _ideal(self, i, k):
        return max(self.lo[i], min(self.hi[i], self.center[i] + k))

    # ---- 信念・誤差
    def _err_dist(self, i, v, mask):
        """マスiの誤差分布 {誤差: 確率}（信念マスク上で一様）。"""
        ks = [k for b, k in enumerate(self.koffs[i]) if mask >> b & 1]
        d = {}
        for k in ks:
            e = abs(v - self._ideal(i, k))
            if self.lo[i] <= v <= self.hi[i]:
                e = min(e, 4)
            else:
                e += 3
            d[e] = d.get(e, 0.0) + 1.0 / len(ks)
        return d

    def terminal(self, cells):
        """終了時の値 = P(大成功) − 期待誤差/1000。
        第2基準の期待誤差で、大成功確率が同点の手（例: てかげん vs 弱ねらい）は本会心で誤差0を狙える方を選ぶ。"""
        dist = {0: 1.0}
        exp_err = 0.0
        for i, (v, mask) in enumerate(cells):
            nd = {}
            ed = self._err_dist(i, v, mask)
            exp_err += sum(e * p for e, p in ed.items())
            for e1, p1 in dist.items():
                for e2, p2 in ed.items():
                    e = e1 + e2
                    if e > self.great:
                        e = self.great + 1  # 超過はまとめる
                    nd[e] = nd.get(e, 0.0) + p1 * p2
            dist = nd
        return sum(p for e, p in dist.items() if e <= self.great) - exp_err / 1000.0

    # ---- 1マスへの1打の結果分岐
    def hit_outcomes(self, i, v, mask, temp, mult_key, aim, forced_crit, lit=False):
        """(新v, 新mask, 確率) のリスト。lit=光っているマス（ダメージ2倍・会心倍率+4）"""
        z = turn_multiplier(self.trait, temp) * (2.0 if lit else 1.0)
        dl = [math.ceil(d * z) for d in dmg_candidates(temp, mult_key)]
        m = 1.0 + (ASSUMPTIONS["aim_bonus"] if aim else 0.0)
        if lit or is_crit_turn(self.trait, temp):
            m += ASSUMPTIONS["event_bonus_mult"]
        p_crit = 1.0 if forced_crit else min(0.95, ASSUMPTIONS["crit_base"] * m)
        out = {}
        ks = [(b, k) for b, k in enumerate(self.koffs[i]) if mask >> b & 1]
        nk = len(ks)
        pd = 1.0 / len(dl)
        for d in dl:
            key = (v + d, mask)  # 非会心
            out[key] = out.get(key, 0.0) + pd * (1 - p_crit)
            if p_crit > 0:
                fake_mask = 0
                for b, k in ks:
                    r_true = self._ideal(i, k) - v
                    if r_true <= 0:
                        key = (v, 1 << b)  # 到達済み/通り過ぎたマスへの会心は Miss（値そのまま、理想値は判明）。必殺・通常とも（2026-10-18）
                        out[key] = out.get(key, 0.0) + pd * p_crit / nk
                    elif 0 < r_true <= 2 * d:
                        key = (self._ideal(i, k), 1 << b)  # 本会心: 理想値判明
                        out[key] = out.get(key, 0.0) + pd * p_crit / nk
                    else:
                        fake_mask |= 1 << b
                if fake_mask:
                    if COARSE_BELIEF:
                        fake_mask = mask  # 粗い信念: 偽会心でも候補を絞らない（状態数を抑える）
                    key = (v + 2 * d, fake_mask)  # 偽会心: 届かなかった候補だけ残る
                    out[key] = out.get(key, 0.0) + pd * p_crit * bin(fake_mask).count("1") / nk
        return [(nv, nm, p) for (nv, nm), p in out.items()]

    def _pinned(self, i, v, mask):
        if bin(mask).count("1") != 1:
            return False
        b = mask.bit_length() - 1
        return v == self._ideal(i, self.koffs[i][b])

    def _hissatsu_ok(self, temp, focus, cells, z):
        """必殺の使用条件（run_amatsuyu.hissatsu_target と同じ思想）: 超4連で4マス本会心、
        面が作れないなら ペア2マス本会心 or 単体本会心。"""
        def r_of(i):
            v, mask = cells[i]
            return self.center[i] - v
        def pinned(i):
            v, mask = cells[i]
            return self._pinned(i, v, mask)
        def openc(i):
            v, mask = cells[i]
            return (not pinned(i)) and v <= self.hi[i] and r_of(i) >= 4
        reach20 = 2 * math.ceil(dmg_candidates(temp, "x2.0")[0] * z)
        possible = False
        quads = [tuple(self.idx[n] for n in q) for q in getattr(self, "quads", [])]
        for q in quads:
            if all(pinned(i) or openc(i) for i in q):
                live = [i for i in q if not pinned(i)]
                if not live:
                    continue
                possible = True
                if all(r_of(i) <= reach20 - 4 for i in live) and focus + ASSUMPTIONS["hissatsu_focus"] >= 18:
                    return True
        if possible:
            return False
        reach12 = 2 * math.ceil(dmg_candidates(temp, "x1.2")[0] * z)
        for a, b in self.pairs["上下ねらい打ち"]:
            ia, ib = self.idx[a], self.idx[b]
            if openc(ia) and openc(ib) and r_of(ia) <= reach12 - 4 and r_of(ib) <= reach12 - 4:
                return True
        reach10 = 2 * math.ceil(dmg_candidates(temp, "x1.0")[0] * z)
        return any(openc(i) and r_of(i) <= reach10 - 4 for i in range(len(cells)))

    def actions(self, temp, focus, cells, charged, armed, lit=-1):
        acts = []
        z = turn_multiplier(self.trait, temp)
        if charged and not armed and self._hissatsu_ok(temp, focus, cells, z):
            acts.append(("必殺", ()))

        def safe(i, mk, regen_ok=False):
            v, mask = cells[i]
            if self._pinned(i, v, mask):
                return armed  # 本会心済み: 通常は触らない。必殺装填中は Miss なので面技の巻き込みOK
            dmax = math.ceil(dmg_candidates(temp, mk)[-1] * z * (2.0 if i == lit else 1.0))
            if regen_ok and v <= self.hi[i]:
                # 再生で戻せる見込みのある手: 非会心の最大の目で上限を REGEN_OVER_MAX まで超えてよい（範囲内のマスでも可。
                # 会心なら理想値で止まり、外れて飛び出しても 200 の倍数の再生で戻る＝アメリーさんの「再生の直前のねらい」）
                return v + dmax <= self.hi[i] + REGEN_OVER_MAX
            return v + dmax <= self.hi[i] and v < self.center[i] + 2

        for sk in SINGLE:
            cost = cost_of(sk, self.trait, temp)
            if focus < cost:
                continue
            mk = SKILLS[sk]["mult"]
            bulk = sk in ("熱風おろし", "3倍打ち", "2倍打ち")
            rg = self._regen_possible(max(0, temp + SKILLS[sk]["dtemp"]), focus - cost)
            for i in range(len(cells)):
                if not bulk and self.center[i] - cells[i][0] > BULK_R:
                    continue  # 遠いマスは運搬技のみ
                if safe(i, mk, rg):
                    acts.append((sk, (i,)))
        for sk in PAIR:
            cost = cost_of(sk, self.trait, temp)
            if focus < cost:
                continue
            mk = SKILLS[sk]["mult"]
            rg = self._regen_possible(max(0, temp + SKILLS[sk]["dtemp"]), focus - cost)
            for g in self.pairs[sk]:
                ii = tuple(self.idx[n] for n in g)
                if all(safe(i, mk, rg) for i in ii):
                    acts.append((sk, ii))
        if temp >= 350 and focus >= cost_of("冷やし込み", self.trait, temp):
            acts.append(("冷やし込み", ()))
        if temp <= 400 and focus >= cost_of("火力上げ", self.trait, temp) \
                and any(not (self.lo[i] <= v <= self.hi[i]) for i, (v, _) in enumerate(cells)):
            acts.append(("火力上げ", ()))
        return acts

    def value(self, temp, focus, cells, charged, armed, lit=-1):
        key = (temp, focus, cells, charged, armed, lit)
        r = self.memo.get(key)
        if r is not None:
            return r[0]
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise TooBig()
        best_v = self.terminal(cells)
        best_a = None
        for a in self.actions(temp, focus, cells, charged, armed, lit):
            q = self.q(a, temp, focus, cells, charged, armed, lit)
            if q > best_v + 1e-12:
                best_v, best_a = q, a
        self.memo[key] = (best_v, best_a)
        return best_v

    def _after(self, t2, f2, cur, charged, armed):
        """行動後の分岐: 再生（温度が 200 の倍数なら範囲外で割合が最大のマスが 12〜16 戻る）→ 発光（光特性）。"""
        if self.regen and t2 > 0 and t2 % 200 == 0:
            cand = [i for i, (v, _m) in enumerate(cur) if not (self.lo[i] <= v <= self.hi[i])]
            if cand:
                tgt = max(cand, key=lambda i: cur[i][0] / max(1, self.zc[i]))
                pr = ASSUMPTIONS["modori_rate"]
                a, b = ASSUMPTIONS["modori_amount"]
                tot = (1.0 - pr) * self._after_light(t2, f2, cur, charged, armed) if pr < 1.0 else 0.0
                v, mask = cur[tgt]
                for back in range(a, b + 1):
                    c2 = list(cur)
                    c2[tgt] = (max(0, v - back), mask)   # 理想値の信念は変わらない（本会心済みは範囲内なので対象にならない）
                    tot += pr / (b - a + 1) * self._after_light(t2, f2, tuple(c2), charged, armed)
                return tot
        return self._after_light(t2, f2, cur, charged, armed)

    def _after_light(self, t2, f2, cur, charged, armed):
        """行動後の発光分岐（光特性: 温度が200の倍数なら light_rate でゾーン外のマスが光る）。"""
        if self.trait != "光" or t2 <= 0 or t2 % 200 != 0:
            return self.value(t2, f2, cur, charged, armed, -1)
        elig = [i for i, (v, mask) in enumerate(cur)
                if not (self.lo[i] <= v <= self.hi[i]) and v < self.center[i]
                and not self._pinned(i, v, mask)]
        if not elig:
            return self.value(t2, f2, cur, charged, armed, -1)
        pl = ASSUMPTIONS["light_rate"]
        tot = (1 - pl) * self.value(t2, f2, cur, charged, armed, -1)
        for i in elig:
            tot += pl / len(elig) * self.value(t2, f2, cur, charged, armed, i)
        return tot

    def q(self, a, temp, focus, cells, charged, armed, lit=-1):
        sk, tg = a
        if sk == "必殺":
            f2 = min(ASSUMPTIONS["focus_max"], focus + ASSUMPTIONS["hissatsu_focus"])
            t2 = max(0, temp + SKILLS["必殺"]["dtemp"])  # 使用ターンは温度-50
            return self.value(t2, f2, cells, False, True, lit)  # 必殺は発光を消さない
        s = SKILLS[sk]
        f2 = focus - cost_of(sk, self.trait, temp)
        t2 = max(0, temp + s["dtemp"])
        if s["targets"] == "none":
            return self._after(t2, f2, cells, charged, armed)
        aim = s["crit"] > 1
        branches = [(cells, 1.0)]  # 各対象の結果分岐（独立）を掛け合わせる
        for i in tg:
            nb = []
            for cur, p in branches:
                v, mask = cur[i]
                for nv, nm, po in self.hit_outcomes(i, v, mask, temp, s["mult"], aim, armed,
                                                    lit=(i == lit)):
                    c2 = list(cur)
                    c2[i] = (nv, nm)
                    nb.append((tuple(c2), p * po))
            branches = nb
        tot = 0.0
        for cur, p in branches:
            if p < 1e-9:
                continue
            tot += p * self._after(t2, f2, cur, charged, False)
        return tot


ALLOW_LIGHT = False     # 光地金でも厳密解を使う（発光分岐込み。hidane の台本では効果ゼロ・時間 4 倍だったが、学習型の終盤は V 頼みなので試す。2026-10-27）


def applicable(sim: Sim) -> bool:
    if sim.focus > ENDGAME_FOCUS or (sim.trait == "光" and not ALLOW_LIGHT):
        return False  # 光: 発光分岐込みで実装済みだが、ひだね200局で効果ゼロ・時間4倍のため適用外（2026-09-04）
    active = [c for c in sim.cells.values() if not c.pinned]
    if any(c.r > NEAR_R for c in active):
        return False
    return len(active) <= MAX_ACTIVE


def solve_capped(sim: Sim, max_nodes: int):
    """状態数上限を一時的に変えて solve。"""
    global MAX_NODES
    old = MAX_NODES
    MAX_NODES = max_nodes
    try:
        return solve(sim)
    finally:
        MAX_NODES = old


def solve(sim: Sim):
    """最良手 (skill, targets) or None(しあげる)。返り値: (手, P(大成功), 状態数)"""
    sv = Solver(sim)
    cells = []
    for n in sv.names:
        c = sim.cells[n]
        if c.pinned:
            cells.append((c.v, 1))  # 中心＝現在値・候補は (0,) のみ（__init__）
        else:
            full = (1 << len(sv.koffs[len(cells)])) - 1
            legacy = KS.USE_FAKE_INFO and ASSUMPTIONS["ideal_spread"] == 2  # kmask は中央±2 の5ビット
            cells.append((c.v, c.kmask if legacy else full))
    cells = tuple(cells)
    lit = sv.idx[sim.lit_cell] if sim.lit_cell in sv.idx else -1
    key = (sim.temp, sim.focus, cells, sim.hissatsu_charged, sim.hissatsu_armed, lit)
    try:
        v = sv.value(*key)
    except TooBig:
        return "toobig", None, sv.nodes
    _, a = sv.memo[key]
    if a is None:
        return None, v, sv.nodes
    sk, tg = a
    return (sk, tuple(sv.names[i] for i in tg)), v, sv.nodes


if __name__ == "__main__":
    import random
    import time
    import run_amatsuyu as R
    import finish_lookahead as FL
    for seed in (1, 2, 3):
        sim = R.make_amatsuyu(random.Random(seed))
        FL.policy_ref(sim, 32)
        s = R.make_amatsuyu(random.Random(seed))
        for sk, tg, _, _ in sim.log:
            if s.focus <= ENDGAME_FOCUS:
                break
            s.act(sk) if sk == "必殺" else s.act(sk, tg)
        t = time.time()
        mv, v, nodes = solve(s)
        act = sum(1 for c in s.cells.values() if not c.pinned)
        print(f"seed{seed}: 温度{s.temp} 集中{s.focus} 未確定{act}  最良手={mv}  P(大成功)={v:.3f}  "
              f"状態数={nodes}  {time.time()-t:.2f}s  (1手先読みの手={FL.next_move(s, 32)})")
