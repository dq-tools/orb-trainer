#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""虹色のオーブ トレーナーの本体（ブラウザの Pyodide 上で動く）。

dq10-auto の web/orb_web.py（Flask 版）から、サーバー・ファイル保存・HTML を外して
「局の進行」と「AI の推奨」だけにしたもの。JS（worker.js）とは JSON 文字列でやり取りする。

- 局: new_game / restore / state / move / undo / finish
- AI: ai（今の局の推奨）/ advise（実機の盤面を受け取って推奨だけ返す）
- AI の中身: 学習した評価関数（value_policy: 1 マス・2 マスの技は出目×会心を全列挙して先の局面の価値の期待値）
  ＋ 終盤（残り集中 50 以下・未確定 4 マス以下）は全分岐を数え上げる厳密解（endgame_solver）
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import kaji_sim as KS          # noqa: E402
import kaji_feat as KF         # noqa: E402
import endgame_solver as ES    # noqa: E402
import value_policy as VP      # noqa: E402

ITEM = "虹色のオーブ"
TOPK = 2            # 提案は上位 2 手
NET = None
G = None            # 進行中の局

SKILLS = ["たたく", "てかげん打ち", "2倍打ち", "3倍打ち", "熱風おろし", "上下打ち", "左右打ち", "ななめ打ち", "4連打ち", "超4連打ち",
          "みだれ打ち", "ねらい打ち", "弱ねらい打ち", "上下ねらい打ち", "火力上げ", "冷やし込み", "必殺"]
SKILL_GROUP = {"single": ["たたく", "てかげん打ち", "2倍打ち", "3倍打ち", "熱風おろし"], "pair": ["上下打ち", "左右打ち", "ななめ打ち"],
               "quad": ["4連打ち", "超4連打ち", "みだれ打ち"], "aim": ["ねらい打ち", "弱ねらい打ち", "上下ねらい打ち"],
               "temp": ["火力上げ", "冷やし込み", "必殺"]}
SKILL_DESC = {"たたく": "x1.0", "てかげん打ち": "x0.5", "2倍打ち": "x2.0", "3倍打ち": "x3.0", "熱風おろし": "x2.5 −150℃", "上下打ち": "縦2 x1.2",
              "左右打ち": "横2 x1.2", "ななめ打ち": "斜2 x1.2", "4連打ち": "2×2 x1.2", "超4連打ち": "2×2 x2.0", "みだれ打ち": "ランダム4打 x0.8",
              "ねらい打ち": "x1.0 会心7倍", "弱ねらい打ち": "x0.5 会心7倍", "上下ねらい打ち": "縦2 x1.2 会心7倍", "火力上げ": "+300℃", "冷やし込み": "−300℃",
              "必殺": "次の1打が会心確定"}


def _j(o):
    return json.dumps(o, ensure_ascii=False)


# ---------------------------------------------------------------- 起動
def init(model_path, max_nodes=0):
    """評価関数の重み（npz）を読む。max_nodes を渡すと厳密解の状態数の上限を変える（ブラウザが遅い時用）"""
    global NET
    NET = VP.load_net(str(model_path))
    if max_nodes:
        ES.MAX_NODES = int(max_nodes)
    return _j({"ok": True, "model": Path(str(model_path)).name, "item": ITEM})


def meta():
    """UI の組み立てに要る固定情報（マスの配置・範囲、特技のグループ）"""
    _, pos = KS.item_layout(ITEM)
    s = KS.make_item(ITEM, random.Random(0))
    cells = [{"name": n, "lo": c.lo, "hi": c.hi, "row": pos[n][0], "col": pos[n][1]} for n, c in s.cells.items()]
    return _j({"cells": cells, "skill_group": list(SKILL_GROUP.items()), "focus_max": KS.ASSUMPTIONS["focus_max"], "item": ITEM})


# ---------------------------------------------------------------- 局の管理
def _snapshot(sim):
    snap = sim.clone(random.Random(0))
    snap.rng = random.Random(); snap.rng.setstate(sim.rng.getstate())
    return snap


def _new_sim(seed):
    return KS.make_item(ITEM, random.Random(seed))


def _turn():
    return len(G["moves"])


def new_game(seed=0):
    global G
    seed = int(seed) or random.randrange(1, 10 ** 6)
    s = _new_sim(seed)
    G = {"sim": s, "history": [], "seed": seed, "moves": [], "log": [f"局開始（seed {seed}）。集中 {s.focus}・温度 {s.temp}℃"],
         "finished": False, "result": None, "ai_used": 0}
    return state()


def restore(payload):
    """保存した局（seed と手の列）を再生して戻す。乱数も seed から復元されるので同じ局面になる"""
    global G
    d = json.loads(payload) if isinstance(payload, str) else dict(payload)
    s = _new_sim(int(d["seed"]))
    hist = []
    moves = []
    for m in d.get("moves") or []:
        t, skill, tg = int(m[0]), m[1], tuple(m[2])
        hist.append(_snapshot(s))
        s.act(skill, tg)
        moves.append((t, skill, "".join(tg)))
    finished = bool(d.get("finished"))
    if finished:
        hist.append(_snapshot(s))
    G = {"sim": s, "history": hist, "seed": int(d["seed"]), "moves": moves, "log": list(d.get("log") or [f"局再開（seed {d['seed']}）"]),
         "finished": finished, "result": d.get("result"), "ai_used": int(d.get("ai_used") or 0)}
    return state()


# ---------------------------------------------------------------- AI
def _move_key(mv):
    mv = VP.first_step(mv)
    return (mv[0], tuple(mv[1]) if len(mv) > 1 else ())


def _move_label(mv):
    if mv is None:
        return "しあげる（終了）"
    skill, tg = mv[0], (mv[1] if len(mv) > 1 else ())
    if "×" in skill:
        base, k = skill.split("×")
        return f"{base}（{k} 回続ける計画）"
    return skill + ("".join(tg) if tg else "")


def recommend(sim, k=TOPK):
    """学習型（評価関数）の上位 k 手＋終盤は厳密解。
    戻り値 {"now": 今終えた時の大成功確率, "value": この局面の見込み, "moves": [{skill, targets, label, p, exact}], "stop": bool}"""
    p_now = KF.p_great_now(sim)
    cand = {}
    moves = KF.legal_moves(sim)
    if moves:
        vals = VP.value_of_moves(sim, NET, moves, VP.SAMPLES)
        for mv, v in zip(moves, vals):
            key = _move_key(mv)
            if key not in cand or v > cand[key]["p"]:
                cand[key] = {"skill": key[0], "targets": list(key[1]), "label": _move_label(mv), "p": float(v), "exact": False}
    exact_first = None
    if VP.USE_SOLVER and ES.applicable(sim):
        mv, p, _nodes = ES.solve(sim)
        if mv != "toobig":
            if mv is None:
                exact_first = {"skill": None, "targets": [], "label": "しあげる（終了）", "p": float(max(0.0, min(1.0, p))), "exact": True}
            else:
                key = (mv[0], tuple(mv[1]) if len(mv) > 1 else ())
                exact_first = {"skill": key[0], "targets": list(key[1]), "label": _move_label(mv), "p": float(max(0.0, min(1.0, p))), "exact": True}
                cand.pop(key, None)
    ranked = sorted(cand.values(), key=lambda x: -x["p"])
    out = ([exact_first] if exact_first else []) + ranked
    out = out[:k]
    best_p = out[0]["p"] if out else 0.0
    stop = (not out) or (exact_first is None and best_p <= p_now) or (exact_first is not None and exact_first["skill"] is None)
    if stop and not (exact_first and exact_first["skill"] is None):
        out.insert(0, {"skill": None, "targets": [], "label": "しあげる（終了）", "p": float(p_now), "exact": True})
        out = out[:k]
    value = max(float(p_now), max((m["p"] for m in out), default=0.0))
    return {"now": float(p_now), "value": float(value), "moves": out, "stop": stop}


def ai():
    if G is None:
        return _j({"error": "局がありません"})
    if G["finished"]:
        return _j({"now": 0.0, "value": 0.0, "moves": [], "stop": True})
    return _j(recommend(G["sim"]))


# ---------------------------------------------------------------- 状態
def _cell_state(s, name, finished):
    c = s.cells[name]
    st = "pinned" if c.pinned else ("over" if c.v > c.hi else ("zone" if c.in_zone else "before"))
    return {"name": name, "v": c.v, "lo": c.lo, "hi": c.hi, "st": st, "pinned": c.pinned, "rem_lo": c.lo - c.v, "rem_hi": c.hi - c.v,
            "err": (c.error if finished else None)}


def _hissatsu_text(s):
    if s.hissatsu_charged and not s.hissatsu_used:
        return "チャージ済"
    if s.hissatsu_armed:
        return "装填中（次の1打が会心確定）"
    if s.hissatsu_used:
        return "使用済"
    return "なし"


def _state_dict():
    s = G["sim"]
    _, pos = KS.item_layout(ITEM)
    cells = [dict(_cell_state(s, n, G["finished"]), row=pos[n][0], col=pos[n][1]) for n in s.cells]
    skills = []
    for n in SKILLS:
        cost = KS.cost_of(n, s.trait, s.temp)
        ok = (s.hissatsu_charged and not s.hissatsu_used) if n == "必殺" else s.can_use(n)
        skills.append({"name": n, "cost": cost, "ok": bool(ok and not G["finished"]), "tg": KS.SKILLS[n]["targets"], "desc": SKILL_DESC.get(n, "")})
    groups = {"pair_v": ["".join(p) for p in s.pairs_v], "pair_lr": ["".join(p) for p in s.pairs_lr],
              "pair_diag": ["".join(p) for p in s.pairs_diag], "quad": ["".join(q) for q in s.quads]}
    regen = KF.regen_target(s) if not G["finished"] else None
    save = {"seed": G["seed"], "moves": [list(m) for m in G["moves"]], "log": G["log"][-200:], "finished": G["finished"],
            "result": G["result"], "ai_used": G["ai_used"]}
    return {"seed": G["seed"], "temp": s.temp, "focus": s.focus, "focus_max": KS.ASSUMPTIONS["focus_max"], "turn": _turn(),
            "hissatsu": _hissatsu_text(s), "hissatsu_charged": bool(s.hissatsu_charged and not s.hissatsu_used), "regen": regen,
            "cells": cells, "skills": skills, "groups": groups, "log": G["log"][-60:], "finished": G["finished"], "result": G["result"],
            "can_undo": bool(G["history"]), "ai_used": G["ai_used"], "save": save}


def state():
    if G is None:
        return _j({"error": "局がありません"})
    return _j(_state_dict())


def _hits_text(entry):
    """Sim.log の 1 件 → 「A+34 B★+12」の形（★本会心 ☆偽会心 ×Miss）"""
    try:
        _skill, tg, _z, res = entry[0], entry[1], entry[2], entry[3]
        parts = []
        for r in res:
            n, kind, d = r[0], r[1], r[-1]
            if kind == "modori":
                continue
            mark = {"crit_pin": "★", "crit_fake": "☆", "miss": "×"}.get(kind, "")
            parts.append(f"{n}{mark}{d:+d}")
        return " ".join(parts)
    except Exception:
        return ""


def move(skill, targets="", by_ai=False):
    """1 手打つ。targets はマス名を並べた文字列（"BD" など）。戻り値は state と同じ（＋ kifu: 棋譜 1 行）"""
    if G is None:
        return _j({"error": "局がありません"})
    if G["finished"]:
        return _j({"error": "しあげ済みです。1手戻すか新しい局を始めてください"})
    s = G["sim"]
    tg = tuple(targets or ())
    if skill not in KS.SKILLS:
        return _j({"error": "unknown skill"})
    if skill == "必殺":
        if not (s.hissatsu_charged and not s.hissatsu_used):
            return _j({"error": "必殺はチャージされていません"})
    elif not s.can_use(skill):
        return _j({"error": f"{skill} は使えません（集中 {s.focus} / 必要 {KS.cost_of(skill, s.trait, s.temp)}）"})
    tgt = KS.SKILLS[skill]["targets"]
    pools = {"pair_v": s.pairs_v, "pair_lr": s.pairs_lr, "pair_diag": s.pairs_diag, "quad": s.quads}
    if tgt == "cell" and (len(tg) != 1 or tg[0] not in s.cells):
        return _j({"error": "マスを選んでください"})
    if tgt in pools and tuple(tg) not in [tuple(x) for x in pools[tgt]]:
        return _j({"error": "対象の組が不正です"})
    if tgt in ("none", "rand4"):
        tg = ()
    G["history"].append(_snapshot(s))
    t, before, f0 = _turn(), s.temp, s.focus
    s.act(skill, tg)
    hits = _hits_text(s.log[-1]) if s.log else ""
    regen = ""
    prev = G["history"][-1]
    for n, c in s.cells.items():
        dv = c.v - prev.cells[n].v
        if dv < 0:
            regen = f"  ⟲再生 {n}{dv:+d}"
    G["log"].append(f"t{t:2d} {before}℃ {'🤖' if by_ai else ''}{skill}{''.join(tg)}  集中{f0}→{s.focus}  {hits}{regen}")
    if by_ai:
        G["ai_used"] += 1
    G["moves"].append((t, skill, "".join(tg)))
    d = _state_dict()
    d["kifu"] = "\t".join([ITEM, "-", str(G["seed"]), str(t), skill, "".join(tg), "AI" if by_ai else ""])
    return _j(d)


def undo():
    if G is None:
        return _j({"error": "局がありません"})
    if not G["history"]:
        return state()
    G["sim"] = G["history"].pop()
    if G["finished"]:
        G["finished"] = False; G["result"] = None
        G["log"].append("（しあげを取り消した）")
    elif G["moves"]:
        G["moves"].pop()
        G["log"].append("（1手戻した）")
    return state()


def finish():
    if G is None:
        return _j({"error": "局がありません"})
    if G["finished"]:
        return state()
    s = G["sim"]
    G["history"].append(_snapshot(s))
    G["finished"] = True
    err = s.total_error(); gr = KS.grade(len(s.cells), err)
    G["result"] = f"{gr}"
    errs = " ".join(f"{k}{'±' if c.error == 0 else ''}{c.error}" for k, c in s.cells.items())
    G["log"].append(f"== しあげる → {gr}  誤差合計 {err}  残集中 {s.focus}  手数 {_turn()}  マス別誤差: {errs}")
    d = _state_dict()
    d["summary"] = {"grade": gr, "err": err, "focus": s.focus, "turns": _turn(), "ai_used": G["ai_used"]}
    d["kifu"] = "\t".join([ITEM, "-", str(G["seed"]), str(_turn()), "しあげる", "", f"{gr} 誤差{err} 残集中{s.focus}"])
    return _j(d)


# ---------------------------------------------------------------- 実機モード
def advise(payload):
    """実機の盤面 {temp, focus, cells:{A:{v,pinned}}, hissatsu:none|charged|armed|used, turn} → 推奨だけ返す"""
    d = json.loads(payload) if isinstance(payload, str) else dict(payload)
    try:
        s = KS.make_item(ITEM, random.Random(0))
        s.temp = int(d.get("temp", 1000)); s.focus = int(d.get("focus", KS.ASSUMPTIONS["focus_max"]))
        if s.temp % 50 or not (0 <= s.temp <= 3000) or not (0 <= s.focus <= 300):
            return _j({"error": "温度は 50 刻み（0〜3000）、集中は 0〜300"})
        for n, c in s.cells.items():
            cd = (d.get("cells") or {}).get(n) or {}
            c.v = max(0, int(cd.get("v", 0)))
            c.pinned = bool(cd.get("pinned"))
            c.ideal_true = c.v if c.pinned else c.center
        hs = d.get("hissatsu", "none")
        s.hissatsu_charged = hs == "charged"; s.hissatsu_armed = hs == "armed"; s.hissatsu_used = hs in ("armed", "used")
        s.started = int(d.get("turn", 0)) > 0 or any(c.v > 0 for c in s.cells.values())
        s.log = ["dummy"] if s.started else []
    except (TypeError, ValueError):
        return _j({"error": "入力が不正です"})
    out = recommend(s)
    _, pos = KS.item_layout(ITEM)
    out["cells"] = [dict(_cell_state(s, n, False), row=pos[n][0], col=pos[n][1]) for n in s.cells]
    out["regen"] = KF.regen_target(s)
    return _j(out)


if __name__ == "__main__":
    # 手元での確認: python orb_core.py → AI だけで 1 局打つ
    import time
    print(init(HERE.parent / "model.npz"))
    st = json.loads(new_game(1))
    while not st["finished"]:
        t0 = time.time(); a = json.loads(ai()); dt = time.time() - t0
        m = a["moves"][0]
        print(f"t{st['turn']:2d} {st['temp']}℃ 集中{st['focus']:3d} → {m['label']:12s} {m['p']:.3f} {'厳密' if m['exact'] else ''} {dt:.2f}s")
        st = json.loads(finish() if m["skill"] is None else move(m["skill"], "".join(m["targets"]), True))
    print(st["log"][-1])
