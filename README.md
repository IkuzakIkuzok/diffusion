
# 拡散実験

物理化学・高分子物性実験「拡散」Part 3の粒子観察を自動で行うためのスクリプト

## Disclaimer

本リポジトリで公開されているスクリプトを使用ないし改変して使用した場合に生じた一切の損害について本リポジトリの管理者は責任を負うものではありません。

## 必要環境

- Python 3.8以上 (セイウチ演算子を使用しているため); Python 3.8.3で動作確認済み
- `click`, `cv2`, `numpy`, `pandas` (必要に応じて`pip`などでインストール)

## 使い方

1. スクリプトをダウンロードする。
2. 読み込むgifファイルのパスなどを指定して`analysis.py`を実行する。
3. 平均二乗変位の一覧のCSVが得られる!

### パラメータ

- `source`: 読み込むgifファイルのパス。必ず指定しなければならない。
- `--destination` (`-o`): 出力ファイル名。省略した場合はgifの拡張子をcsvに変更した値が使用される。
- `--mode` (`-o`): 出力の書き込みモード。`a`を指定すると追記モードになる(バッチ処理で複数のデータを処理する際に便利)。規定値は`w`。
- `--threshold` (`-t`): 画像を二値化するのに用いる閾値。規定値は150。
- `--width` (`-w`): 画像の横の実際の長さ。規定値は150。
- `--height` (`-h`): 画像の縦の実際の長さ。規定値は150。
- `--distance` (`-d`): 粒子の経時変化による移動を検出する最長の距離。実際の長さではなく画像上の距離で指定する。規定値は15。
- `--lower` (`-l`): 解析に用いる経時変化の最小の長さ。規定値は2。

#### パラメータ調整のポイント

`--distance`を小さくしすぎると経時変化を追跡できなくなるが，大きくしすぎると関係ない粒子を同一のものとして認識してしまう。
300 nmの粒子(A)では規定値を，直径未知粒子では12くらいがよいでしょう。

### 使用例

Windowsバッチファイルを利用した例を示す。

```
@echo off

cd %~dp0

set SRC=./data/A/
set DST=data_a.csv

if exist %DST% del %DST%

for %%f in (%SRC%*.gif) do (
  python analysis.py %SRC%%%f -o "%DST%" -m a -l 11
)
```
