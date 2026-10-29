"""職人レシピの基準値データローダー。

recipes.json に5職人（武器鍛冶・防具鍛冶・道具鍛冶・木工・さいほう）計526件の
各マス基準値・素材特性が入っている。
出典: https://meluce.jp/flyby/dqx/ （2026-09-03取得）

grid の書式:
  行を "/" で区切り、行内のマスを "," で区切る。 "-" は空マス。
  鍛冶系:   "500-508" のような基準値範囲（成功ゾーン）
  木工:     "620v" / "270h" のように数値+方向（v=縦, h=横）
  さいほう: 基準値のみ（tol が許容誤差）
"""
import json
import os

_JSON_PATH = os.path.join(os.path.dirname(__file__), 'recipes.json')
_cache = None


def _load():
    global _cache
    if _cache is None:
        with open(_JSON_PATH, encoding='utf-8') as f:
            _cache = json.load(f)
    return _cache


def guilds():
    """職人名の一覧を返す。"""
    return list(_load()['guilds'].keys())


def items(guild=None):
    """レシピ一覧を返す。guild を指定するとその職人のみ。"""
    data = _load()['guilds']
    if guild:
        return data[guild]
    return [it for lst in data.values() for it in lst]


def find(name):
    """装備名でレシピを探す。(職人名, レシピdict) を返す。見つからなければ (None, None)。"""
    for guild, lst in _load()['guilds'].items():
        for it in lst:
            if it['name'] == name:
                return guild, it
    return None, None


def parse_grid(grid):
    """grid 文字列をマスの2次元リストに変換する（鍛冶系向け）。

    各マスは None（空）か dict:
      {'lo': 500, 'hi': 508}          鍛冶系の基準値範囲
      {'value': 620, 'dir': 'v'}      木工（v=縦, h=横）
      {'value': 420}                  さいほう
    """
    rows = []
    for row_s in grid.split('/'):
        row = []
        for cell in row_s.split(','):
            cell = cell.strip()
            if cell in ('', '-'):
                row.append(None)
            elif '-' in cell:
                lo, hi = cell.split('-')
                row.append({'lo': int(lo), 'hi': int(hi)})
            elif cell[-1] in ('v', 'h'):
                row.append({'value': int(cell[:-1]), 'dir': cell[-1]})
            else:
                row.append({'value': int(cell)})
        rows.append(row)
    return rows


def target_cells(name):
    """装備名から叩くべきマスのリストを返す。

    [(行, 列, lo, hi), ...] 形式（鍛冶系向け）。
    """
    _, it = find(name)
    if it is None:
        raise KeyError(f'レシピが見つかりません: {name}')
    result = []
    for r, row in enumerate(parse_grid(it['grid'])):
        for c, cell in enumerate(row):
            if cell and 'lo' in cell:
                result.append((r, c, cell['lo'], cell['hi']))
    return result


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    for g in guilds():
        print(f'{g}: {len(items(g))}件')
    print()
    g, it = find('幽紋刀')
    print(f"幽紋刀 ({g}) 特性:{it['trait']} 基準値合計:{it['baseLo']}-{it['baseHi']}")
    for r, c, lo, hi in target_cells('幽紋刀'):
        print(f'  マス({r},{c}): {lo}〜{hi}')
