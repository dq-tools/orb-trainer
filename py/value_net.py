#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局面の価値（大成功確率）を学習する評価関数（2026-10-27）。ギルド・商材に依存しない骨組み。

入力 = マスの集合（各マス FC 次元、最大 MAX_CELLS、mask 付き）＋ 盤面全体 FG 次元。
各マスを同じ小さな MLP で読み、mask つきの平均と最大で束ねて全体の特徴と結合 → MLP → sigmoid。
学習は torch（CPU）、推論は numpy（重みは npz）。

python value_net.py train <data.npz ...> -o <model.npz> [--epochs 8] [--hidden 64]
python value_net.py eval <model.npz> <data.npz>
"""
import os
import sys
import numpy as np


class NumpyNet:
    """npz の重みで前向き計算だけ行う（並列ワーカーで torch を読まずに済む）"""

    def __init__(self, path: str):
        w = np.load(path)
        self.w = {k: w[k] for k in w.files}
        self.hidden = self.w["c1_w"].shape[0]

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def forward(self, X: np.ndarray, M: np.ndarray, G: np.ndarray) -> np.ndarray:
        """X [B, MAX_CELLS, FC], M [B, MAX_CELLS], G [B, FG] → P(大成功) [B]"""
        w = self.w
        h = self._relu(X @ w["c1_w"].T + w["c1_b"])
        h = self._relu(h @ w["c2_w"].T + w["c2_b"])
        m = M[:, :, None]
        mean = (h * m).sum(1) / np.maximum(m.sum(1), 1.0)
        mx = np.where(m > 0, h, -1e9).max(1)
        mx = np.where(M.sum(1, keepdims=True) > 0, mx, 0.0)
        z = np.concatenate([mean, mx, G], axis=1)
        z = self._relu(z @ w["g1_w"].T + w["g1_b"])
        z = self._relu(z @ w["g2_w"].T + w["g2_b"])
        o = z @ w["o_w"].T + w["o_b"]
        return 1.0 / (1.0 + np.exp(-o[:, 0]))


class StageNet:
    """段階ごとの重み（2048 の multi-stage weight promotion に倣う、2026-10-29）。局面の段階は残り集中力で決める:
    stage_bounds=[170,100] なら 集中 >170 → 段階 0、100<集中≤170 → 段階 1、≤100 → 段階 2。npz のキーは s{k}_c1_w … と stage_bounds"""

    def __init__(self, path: str):
        w = np.load(path)
        self.bounds = [float(b) for b in w["stage_bounds"]]
        self.nets = []
        for k in range(len(self.bounds) + 1):
            sub = NumpyNet.__new__(NumpyNet)
            sub.w = {key[len(f"s{k}_"):]: w[key] for key in w.files if key.startswith(f"s{k}_")}
            sub.hidden = sub.w["c1_w"].shape[0]
            self.nets.append(sub)
        self.hidden = self.nets[0].hidden

    def stage_of(self, G: np.ndarray) -> np.ndarray:
        focus = G[:, 8] * 253.0
        st = np.zeros(len(G), dtype=np.int64)
        for b in self.bounds:
            st += (focus <= b + 1e-6)
        return st

    def forward(self, X, M, G):
        st = self.stage_of(G)
        out = np.zeros(len(G))
        for k, net in enumerate(self.nets):
            sel = st == k
            if sel.any():
                out[sel] = net.forward(X[sel], M[sel], G[sel])
        return out


def stage_mask(G: np.ndarray, bounds, k: int) -> np.ndarray:
    """段階 k の局面だけ True（StageNet.stage_of と同じ決め方）"""
    focus = G[:, 8] * 253.0
    st = np.zeros(len(G), dtype=np.int64)
    for b in bounds:
        st += (focus <= b + 1e-6)
    return st == k


def load_model(path: str):
    """npz に stage_bounds があれば StageNet、無ければ NumpyNet"""
    with np.load(path) as w:
        staged = "stage_bounds" in w.files
    return StageNet(path) if staged else NumpyNet(path)


def load_data(paths):
    Xs, Ms, Gs, ys = [], [], [], []
    for p in paths:
        d = np.load(p)
        Xs.append(d["X"]); Ms.append(d["M"]); Gs.append(d["G"]); ys.append(d["y"])
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(Ms).astype(np.float32),
            np.concatenate(Gs).astype(np.float32), np.concatenate(ys).astype(np.float32))


def build_torch(fc: int, fg: int, hidden: int = 64):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(fc, hidden); self.c2 = nn.Linear(hidden, hidden)
            self.g1 = nn.Linear(2 * hidden + fg, 2 * hidden); self.g2 = nn.Linear(2 * hidden, hidden)
            self.o = nn.Linear(hidden, 1)

        def forward(self, X, M, G):
            h = torch.relu(self.c1(X)); h = torch.relu(self.c2(h))
            m = M.unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = torch.where(m > 0, h, torch.full_like(h, -1e9)).max(1).values
            mx = torch.where(M.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
            z = torch.cat([mean, mx, G], dim=1)
            z = torch.relu(self.g1(z)); z = torch.relu(self.g2(z))
            return self.o(z)[:, 0]

    return Net()


def load_into(net, path: str):
    """npz の重みを torch モデルに入れる（前の重みを初期値にした微調整用）"""
    import torch
    w = np.load(path)
    sd = net.state_dict()
    for k in ("c1", "c2", "g1", "g2", "o"):
        if sd[f"{k}.weight"].shape != tuple(w[f"{k}_w"].shape):
            raise ValueError(f"{path}: {k} の形が違う {tuple(w[f'{k}_w'].shape)} vs {tuple(sd[f'{k}.weight'].shape)}")
        sd[f"{k}.weight"] = torch.tensor(w[f"{k}_w"], dtype=torch.float32)
        sd[f"{k}.bias"] = torch.tensor(w[f"{k}_b"], dtype=torch.float32)
    net.load_state_dict(sd)


def export(net, path: str):
    w = {}
    sd = net.state_dict()
    for k in ("c1", "c2", "g1", "g2", "o"):
        w[f"{k}_w"] = sd[f"{k}.weight"].cpu().numpy().astype(np.float64)
        w[f"{k}_b"] = sd[f"{k}.bias"].cpu().numpy().astype(np.float64)
    np.savez(path, **w)


def train(paths, out: str, epochs: int = 8, hidden: int = 64, lr: float = 1e-3, batch: int = 1024, seed: int = 0,
          val_frac: float = 0.05, weight_decay: float = 1e-5, init: str | None = None, device: str | None = None,
          data=None, select=None):
    """init=前の重み（npz）を初期値にして微調整（epochs を 2〜3 に減らせる）。device は cuda があれば自動で使う（jobq の GPU の仕事には hold の合図を出す）。
    data=(X,M,G,y) を渡せば paths を読まない。select=G→bool の関数で学習に使う行を絞る（段階ごとの学習用、2026-10-29）"""
    import torch
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            sys.path.insert(0, "c:/work-space/tool/jobq")
            import gpu as _gpu
            _gpu.hold_while_running()
        except Exception as e:
            print("gpu hold を出せない:", e)
    X, M, G, y = data if data is not None else load_data(paths)
    if select is not None:
        keep = select(G)
        X, M, G, y = X[keep], M[keep], G[keep], y[keep]
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = int(n * val_frac)
    va, tr = perm[:nv], perm[nv:]
    Xt, Mt, Gt, yt = (torch.tensor(a) for a in (X, M, G, y))   # データは CPU に置き、バッチだけ GPU へ（300 万局面で 5GB 超なので VRAM に載せない）
    net = build_torch(X.shape[2], G.shape[1], hidden)
    if init:
        load_into(net, init)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"学習 {len(tr)} 局面・検証 {nv}・正例率 {y.mean():.3f}・hidden {hidden}・{device}{'・初期値 ' + init if init else ''}")
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr)
        tot = 0.0; cnt = 0
        for i in range(0, len(tr), batch):
            idx = torch.tensor(tr[i:i + batch])
            opt.zero_grad()
            logit = net(Xt[idx].to(device), Mt[idx].to(device), Gt[idx].to(device))
            loss = lossf(logit, yt[idx].to(device))
            loss.backward(); opt.step()
            tot += float(loss) * len(idx); cnt += len(idx)
        net.eval()
        with torch.no_grad():
            vi = torch.tensor(va)
            vlog = torch.cat([net(Xt[vi[j:j + 8192]].to(device), Mt[vi[j:j + 8192]].to(device), Gt[vi[j:j + 8192]].to(device)) for j in range(0, len(vi), 8192)])
            vo = torch.sigmoid(vlog).cpu().numpy()
            vy = y[va]
            vl = float(lossf(vlog, yt[vi].to(device)))
            acc = float(((vo > 0.5) == (vy > 0.5)).mean())
            brier = float(((vo - vy) ** 2).mean())
        print(f"epoch {ep + 1}: 学習 loss {tot / cnt:.4f}  検証 loss {vl:.4f}  正答 {acc:.3f}  Brier {brier:.4f}", flush=True)
        export(net, out)   # 1 周ごとに保存（途中で止めても直前の周まで残る。2026-10-27 の 3 時間分の損失の再発防止）
    export(net, out)
    print("保存", out)


def train_stages(paths, out: str, bounds, init: str | None, epochs: int = 3, hidden: int = 64, seed: int = 0, lr: float = 1e-3):
    """段階ごとの重みを順に学習して 1 つの npz にまとめる（weight promotion: 段階 0 は init から、段階 k は段階 k−1 の結果から始める）。
    bounds=[170,100] のように残り集中の境を高い順に。データは 1 回だけ読む"""
    import tempfile
    data = load_data(paths)
    print(f"全体 {len(data[3])} 局面、段階の境（残り集中） {bounds}")
    prev = init
    merged = {"stage_bounds": np.array(bounds, dtype=np.float64)}
    tmpdir = tempfile.mkdtemp(prefix="stages_")
    for k in range(len(bounds) + 1):
        tmp = os.path.join(tmpdir, f"s{k}.npz")
        print(f"--- 段階 {k}（初期値 {prev}）")
        train(None, tmp, epochs=epochs, hidden=hidden, lr=lr, seed=seed, init=prev, data=data, select=lambda G, k=k: stage_mask(G, bounds, k))
        w = np.load(tmp)
        for key in w.files:
            merged[f"s{k}_{key}"] = w[key]
        prev = tmp
    np.savez(out, **merged)
    print("保存", out)


def evaluate(model: str, paths):
    X, M, G, y = load_data(paths)
    net = load_model(model)
    p = np.concatenate([net.forward(X[i:i + 4096], M[i:i + 4096], G[i:i + 4096]) for i in range(0, len(y), 4096)])
    print(f"{len(y)} 局面: Brier {((p - y) ** 2).mean():.4f}  正答 {((p > 0.5) == (y > 0.5)).mean():.3f}  平均予測 {p.mean():.3f} 実際 {y.mean():.3f}")
    # 予測の帯ごとの実際（校正）
    for lo in np.arange(0, 1.0, 0.1):
        sel = (p >= lo) & (p < lo + 0.1)
        if sel.sum():
            print(f"  予測 {lo:.1f}〜{lo + 0.1:.1f}: {sel.sum():6d} 局面  実際 {y[sel].mean():.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train"); t.add_argument("data", nargs="+"); t.add_argument("-o", required=True)
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--hidden", type=int, default=64); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--init", default=None, help="前の重み（npz）を初期値にして微調整"); t.add_argument("--device", default=None)
    e = sub.add_parser("eval"); e.add_argument("model"); e.add_argument("data", nargs="+")
    st = sub.add_parser("train-stages", help="段階ごとの重み（残り集中の境で分ける）"); st.add_argument("data", nargs="+"); st.add_argument("-o", required=True)
    st.add_argument("--bounds", default="170,100"); st.add_argument("--init", default=None); st.add_argument("--epochs", type=int, default=3); st.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data, a.o, epochs=a.epochs, hidden=a.hidden, lr=a.lr, init=a.init, device=a.device)
    elif a.cmd == "train-stages":
        train_stages(a.data, a.o, [float(x) for x in a.bounds.split(",")], a.init, epochs=a.epochs, hidden=a.hidden)
    elif a.cmd == "eval":
        evaluate(a.model, a.data)
    else:
        ap.print_help()
