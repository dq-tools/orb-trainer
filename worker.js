/* 虹色のオーブ トレーナー: Python（Pyodide）を Web Worker で動かす。
   UI（app.js）とは {id, fn, args} → {id, result | error} でやり取りする。
   起動の進み具合は {type:"progress"}、起動完了は {type:"ready"}、起動失敗は {type:"fatal"} で知らせる */
// 0.29 系の最新を使う。314.x 系（2026 時点の最新）は importScripts で読めない（NetworkError、2026-10-29 に Chromium で確認）
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const VERSION = "2026-10-29c";   // py/ や model.npz を変えたらここも変える（ブラウザのキャッシュ避け）
const PY_FILES = ["kaji_data.py", "kaji_sim.py", "recipes.py", "recipes.json", "kaji_feat.py", "value_net.py", "run_amatsuyu.py",
                  "endgame_solver.py", "finish_lookahead.py", "value_policy.py", "orb_core.py"];

importScripts(PYODIDE_URL + "pyodide.js");

let core = null;
const post = (m) => self.postMessage(m);

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} が読めない (${r.status})`);
  return r;
}

async function boot() {
  try {
    post({ type: "progress", step: 1, msg: "Python 実行環境を読み込み中…（初回は 10〜20 秒）" });
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    post({ type: "progress", step: 2, msg: "数値ライブラリ（numpy）を読み込み中…" });
    await pyodide.loadPackage("numpy");
    post({ type: "progress", step: 3, msg: "シミュレータと AI を読み込み中…" });
    pyodide.FS.mkdir("/orb");
    for (const f of PY_FILES) {
      const r = await fetchOk(`py/${f}?v=${VERSION}`);
      pyodide.FS.writeFile(`/orb/${f}`, await r.text());
    }
    const mr = await fetchOk(`model.npz?v=${VERSION}`);
    pyodide.FS.writeFile("/orb/model.npz", new Uint8Array(await mr.arrayBuffer()));
    pyodide.runPython("import sys; sys.path.insert(0, '/orb')");
    core = pyodide.pyimport("orb_core");
    const info = JSON.parse(core.init("/orb/model.npz"));
    post({ type: "ready", info });
  } catch (e) {
    post({ type: "fatal", error: String((e && e.message) || e) });
  }
}
const booted = boot();

self.onmessage = async (ev) => {
  const { id, fn, args } = ev.data;
  await booted;
  try {
    if (!core) throw new Error("AI が起動していません");
    const f = core[fn];
    if (!f) throw new Error("unknown fn " + fn);
    const out = f(...(args || []));
    post({ id, result: JSON.parse(out) });
  } catch (e) {
    post({ id, error: String((e && e.message) || e) });
  }
};
