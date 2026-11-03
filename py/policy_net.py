#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方策（どの手を選ぶか）の網（2026-11-01、さいほうから。鍛冶も同じ形で使える）。

評価関数（value_net）と同じ「マスの集合 → 束ねる」作りで、出力は 2 つ:
  - 技の確率  P(skill | 局面)            … softmax（技の番号表は saihou_value.SKILL_LIST）
  - 対象マスの確率 P(マス i が対象 | 局面) … マスごとの sigmoid（単体技は 1 つ、多マス技は複数、対象なしの技は学習に使わない）
正解は自己対局で選ばれた手（npz の A = [技の番号, 対象マスのビット]）。評価関数の重みとは別の npz（キー p_*）。

学習:  python policy_net.py train @一覧.txt -o value_models/saihou_pol_v1.npz [--epochs 6] [--hidden 64]
評価:  python policy_net.py eval value_models/saihou_pol_v1.npz @一覧.txt     （技の正答率・対象の正答率）
使い方（推論）: net = NumpyPolicy(path); logp_skill, logp_cell = net.forward(X, M, G)
                score_moves(net, X, M, G, moves, names, skill_list) → 手ごとの log 確率（技 ＋ 対象マスの平均）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import value_net as VN


def load_policy_data(paths):
    """A（選んだ手）のある npz だけ読む。戻り値 X, M, G, A, w（一覧に 2 回あるファイルは重み 2）"""
    from collections import Counter
    cnt = Counter(VN.expand_paths(paths))
    Xs, Ms, Gs, As, ws = [], [], [], [], []
    skipped = 0
    for p, k in cnt.items():
        with np.load(p) as d:
            if "A" not in d.files:
                skipped += 1; continue
            A = d["A"]
            keep = A[:, 0] >= 0
            if not keep.any():
                continue
            Xs.append(d["X"][keep]); Ms.append(d["M"][keep]); Gs.append(d["G"][keep]); As.append(A[keep]); ws.append(np.full(int(keep.sum()), k, dtype=np.float32))
    if not Xs:
        raise RuntimeError("A（選んだ手）のある npz が無い")
    gw = max(g.shape[1] for g in Gs)
    Gs = [g if g.shape[1] == gw else np.pad(g, ((0, 0), (0, gw - g.shape[1]))) for g in Gs]
    X = np.concatenate(Xs).astype(np.float32); M = np.concatenate(Ms).astype(np.float32); G = np.concatenate(Gs).astype(np.float32)
    A = np.concatenate(As).astype(np.int64); w = np.concatenate(ws)
    if skipped:
        print(f"A の無い npz {skipped} 本は飛ばした")
    return X, M, G, A, w


def build_policy_torch(fc: int, fg: int, n_skill: int, hidden: int = 64):
    import torch
    import torch.nn as nn

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(fc, hidden); self.c2 = nn.Linear(hidden, hidden)
            self.g1 = nn.Linear(2 * hidden + fg, 2 * hidden); self.g2 = nn.Linear(2 * hidden, hidden)
            self.s = nn.Linear(hidden, n_skill)                 # 技
            self.t1 = nn.Linear(2 * hidden, hidden); self.t2 = nn.Linear(hidden, 1)   # 対象マス（マスの h と全体の z を結合）

        def forward(self, X, M, G):
            h = torch.relu(self.c1(X)); h = torch.relu(self.c2(h))
            m = M.unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = torch.where(m > 0, h, torch.full_like(h, -1e9)).max(1).values
            mx = torch.where(M.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
            z = torch.cat([mean, mx, G], dim=1)
            z = torch.relu(self.g1(z)); z = torch.relu(self.g2(z))
            skill = self.s(z)
            zz = z.unsqueeze(1).expand(-1, h.shape[1], -1)
            cell = self.t2(torch.relu(self.t1(torch.cat([h, zz], dim=2))))[:, :, 0]
            return skill, cell

    return PolicyNet()


KEYS = ("c1", "c2", "g1", "g2", "s", "t1", "t2")


def export_policy(net, path: str, n_skill: int):
    sd = net.state_dict(); w = {"p_n_skill": np.array([n_skill])}
    for k in KEYS:
        w[f"p_{k}_w"] = sd[f"{k}.weight"].cpu().numpy().astype(np.float64)
        w[f"p_{k}_b"] = sd[f"{k}.bias"].cpu().numpy().astype(np.float64)
    np.savez(path, **w)


def load_policy_into(net, path: str):
    import torch
    w = np.load(path); sd = net.state_dict()
    for k in KEYS:
        if tuple(sd[f"{k}.weight"].shape) != tuple(w[f"p_{k}_w"].shape):
            raise ValueError(f"{path}: {k} の形が違う")
        sd[f"{k}.weight"] = torch.tensor(w[f"p_{k}_w"], dtype=torch.float32); sd[f"{k}.bias"] = torch.tensor(w[f"p_{k}_b"], dtype=torch.float32)
    net.load_state_dict(sd)


def train_policy(paths, out: str, epochs: int = 6, hidden: int = 64, lr: float = 1e-3, batch: int = 1024, seed: int = 0, val_frac: float = 0.05, device=None,
                 init: str | None = None, n_skill_fixed: int | None = None):
    """init=前の方策の重みを初期値に（本人の棋譜での微調整用。epochs を 2〜3・lr を 3e-4 程度に）。n_skill_fixed=技の数を固定（微調整のデータに無い技があっても形を合わせる）"""
    import torch
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X, M, G, A, w = load_policy_data(paths)
    n_skill = n_skill_fixed or int(A[:, 0].max()) + 1
    cells = X.shape[1]
    bits = ((A[:, 1:2] >> np.arange(cells)[None, :]) & 1).astype(np.float32)   # 対象マスのビット → [n, cells]
    has_t = (bits.sum(1) > 0).astype(np.float32)                                # 対象のある手だけマスの損失
    n = len(A); rng = np.random.default_rng(seed); perm = rng.permutation(n); nv = int(n * val_frac); va, tr = perm[:nv], perm[nv:]
    Xt, Mt, Gt, wt = (torch.from_numpy(np.ascontiguousarray(a)) for a in (X, M, G, w))
    St = torch.from_numpy(A[:, 0]); Bt = torch.from_numpy(bits); Ht = torch.from_numpy(has_t)
    net = build_policy_torch(X.shape[2], G.shape[1], n_skill, hidden)
    if init:
        load_policy_into(net, init)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    ce = torch.nn.CrossEntropyLoss(reduction="none"); bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(f"方策の学習 {len(tr)} 局面（重み合計 {float(w[tr].sum()):.0f}）・検証 {nv}・技 {n_skill} 種・マス {cells}・hidden {hidden}・{device}")
    for ep in range(epochs):
        net.train(); rng.shuffle(tr); tot = 0.0; cnt = 0; t0 = time.time()
        for i in range(0, len(tr), batch):
            idx = torch.tensor(tr[i:i + batch])
            opt.zero_grad()
            sl, cl = net(Xt[idx].to(device), Mt[idx].to(device), Gt[idx].to(device))
            wb = wt[idx].to(device); mb = Mt[idx].to(device); hb = Ht[idx].to(device)
            l_s = ce(sl, St[idx].to(device))
            l_c = (bce(cl, Bt[idx].to(device)) * mb).sum(1) / mb.sum(1).clamp(min=1.0) * hb
            loss = ((l_s + 2.0 * l_c) * wb).sum() / wb.sum()
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx); cnt += len(idx)
        net.eval()
        with torch.no_grad():
            vi = torch.tensor(va)
            sl = []; cl = []
            for j in range(0, len(vi), 8192):
                a, b = net(Xt[vi[j:j + 8192]].to(device), Mt[vi[j:j + 8192]].to(device), Gt[vi[j:j + 8192]].to(device)); sl.append(a); cl.append(b)
            sl = torch.cat(sl); cl = torch.cat(cl)
            acc_s = float((sl.argmax(1).cpu() == St[vi]).float().mean())
            # 対象の正答: 対象のある手で、正解ビットの中でいちばん高いマス == 予測の最大マス
            cm = torch.where(Mt[vi].to(device) > 0, cl, torch.full_like(cl, -1e9)).argmax(1).cpu()
            hit = Bt[vi][torch.arange(len(vi)), cm] > 0
            ht = Ht[vi] > 0
            acc_c = float(hit[ht].float().mean()) if ht.any() else float("nan")
        print(f"epoch {ep + 1}: 学習 loss {tot / max(cnt, 1):.4f}  検証 技の正答 {acc_s:.3f}  対象の正答 {acc_c:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    export_policy(net, out, n_skill)
    print("保存", out)


class NumpyPolicy:
    """npz（p_*）で前向きだけ。forward → (技の log 確率 [B, S], マスの log P [B, cells], マスの log(1-P) [B, cells])"""

    def __init__(self, path: str):
        w = np.load(path); self.w = {k: w[k] for k in w.files}
        self.hidden = self.w["p_c1_w"].shape[0]; self.n_skill = int(self.w["p_n_skill"][0])

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def forward(self, X, M, G):
        w = self.w
        gw = w["p_g1_w"].shape[1] - 2 * self.hidden
        if G.shape[1] > gw:
            G = G[:, :gw]
        elif G.shape[1] < gw:
            G = np.pad(G, ((0, 0), (0, gw - G.shape[1])))
        h = self._relu(X @ w["p_c1_w"].T + w["p_c1_b"]); h = self._relu(h @ w["p_c2_w"].T + w["p_c2_b"])
        m = M[:, :, None]
        mean = (h * m).sum(1) / np.maximum(m.sum(1), 1.0)
        mx = np.where(m > 0, h, -1e9).max(1); mx = np.where(M.sum(1, keepdims=True) > 0, mx, 0.0)
        z = np.concatenate([mean, mx, G], axis=1)
        z = self._relu(z @ w["p_g1_w"].T + w["p_g1_b"]); z = self._relu(z @ w["p_g2_w"].T + w["p_g2_b"])
        sl = z @ w["p_s_w"].T + w["p_s_b"]
        sl = sl - sl.max(1, keepdims=True); logp_s = sl - np.log(np.exp(sl).sum(1, keepdims=True))
        zz = np.repeat(z[:, None, :], h.shape[1], axis=1)
        c = self._relu(np.concatenate([h, zz], axis=2) @ w["p_t1_w"].T + w["p_t1_b"]) @ w["p_t2_w"].T + w["p_t2_b"]
        c = c[:, :, 0]
        logp_c = -np.logaddexp(0.0, -c); log1m_c = -np.logaddexp(0.0, c)
        return logp_s, logp_c, log1m_c


def score_moves(net: NumpyPolicy, X, M, G, moves, names: list, skill_list: list) -> np.ndarray:
    """1 局面の特徴（X [cells, FC] を pad 済み [MAX, FC]、M [MAX]、G [FG]）と候補手 → 手ごとの log 確率（技 ＋ 対象マスの log P の平均。対象なしは技だけ）"""
    ls, lc, _ = net.forward(X[None], M[None], G[None])
    ls, lc = ls[0], lc[0]
    out = np.empty(len(moves))
    for i, mv in enumerate(moves):
        sk = mv[0]; tg = mv[1] if len(mv) > 1 else ()
        s = ls[skill_list.index(sk)] if sk in skill_list else -20.0
        tg = list(tg) if isinstance(tg, (tuple, list, frozenset, set)) else ([tg] if tg else [])
        idx = [names.index(t) for t in tg if t in names]
        if idx:
            s += float(np.mean([lc[j] for j in idx]))
        out[i] = s
    return out


def evaluate(model: str, paths):
    net = NumpyPolicy(model)
    X, M, G, A, w = load_policy_data(paths)
    ls, lc, _ = net.forward(X, M, G)
    acc_s = float((ls.argmax(1) == A[:, 0]).mean())
    cells = X.shape[1]; bits = ((A[:, 1:2] >> np.arange(cells)[None, :]) & 1)
    cm = np.where(M > 0, lc, -1e9).argmax(1); has = bits.sum(1) > 0
    acc_c = float(bits[np.arange(len(A)), cm][has].mean()) if has.any() else float("nan")
    print(f"{model}: {len(A)} 局面  技の正答 {acc_s:.3f}  対象の正答 {acc_c:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train"); t.add_argument("data", nargs="+"); t.add_argument("-o", required=True); t.add_argument("--epochs", type=int, default=6); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--init", default=None); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--n-skill", type=int, default=None)
    e = sub.add_parser("eval"); e.add_argument("model"); e.add_argument("data", nargs="+")
    a = ap.parse_args()
    if a.cmd == "train":
        train_policy(a.data, a.o, epochs=a.epochs, hidden=a.hidden, init=a.init, lr=a.lr, n_skill_fixed=a.n_skill)
    elif a.cmd == "eval":
        evaluate(a.model, a.data)
