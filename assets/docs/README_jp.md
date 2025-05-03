# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/README.md) | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_en.md) | 日本語 \]

## 概要
「長門桜-イメージチャーム」は「長門桜プロジェクト」の派生品であり、「長門桜」の画像コアをベースにした画像強化・評価のためのグラフィカルツールです。AI超解像、画像・動画の強化、品質評価などの機能をサポートしています。

## お知らせ
計算リソースの不足と開発者の多忙により、バージョン1.1.1以降は長門桜-イメージチャームの開発を一時停止し、長門桜-イメージチャームトレーナー関連のリポジトリとDiscordバージョンのcogの完成を優先します。

## 最近の変更
### 1.1.1 (2025年5月04日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.1.jpg)
### 重要な変更
- 【重要】AI画像評価コアのモデル構造を完全に書き直し、巨大なTransformerモデルを廃止して従来のCNNモデルに戻し、モデル構造の軽量化を行いました。
### 新機能
- 【新規】評価ページにエッジ比較機能を追加し、OpenCV Cannyを使用して画像のエッジを描画し、2つの画像を比較できるようになりました。
- 【更新】評価ページのUIレイアウトを更新し、下部のプレビューボックスをスクロールボックスから操作がより自由なviews.py形式に変更しました。
- 【更新】Ritsuka-HQのモデル説明を最適化しました。
- 【修正】AI画像評価モデルを全面的に書き直し、現在約1MBのスペースで良好なAI評価効果を実現できます。
### 既知の問題
- 【バグ】超解像画像のプレビューサイズが元の画像と異なるため、直感的な比較が難しくなっています。
- 【バグ】拡張プラグイン管理機能が未完成のため、バージョン1.1.1ではNagato-Sakura-Image-Quality-Assessment画像評価機能（約1MB）が強制的にプリインストールされます。

### 1.1.0 (2025年4月28日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.0.jpg)
### 重要な変更
- 【重要】計算デバイス選択機能を追加し、手動でCPUを強制的に有効にして画像・動画処理を行うことができるようになりました。
### 新機能
- 【新規】Ritsuka-HQ 第七世代アニメモザイク復元モデルを追加しました。
- 【更新】ベンチマークテストのバランスを最適化し、メモリ割り当て戦略を調整、VRAMの割り当て戦略を改善、進捗バーの精度を最適化しました。
- 【更新】計算デバイス選択機能で、CPU及びNVIDIA GPUのモデルを直接確認でき、複数デバイスの切り替えが容易になりました（NS_DeviceInfo.pyを新規追加）。
- 【修正】モデルがない状態で外部モデルをインポートして登録した後、モデルが使用できない問題を修正しました。
- 【修正】ベンチマークテストを切り替えるとVRAMが正常に解放されない問題を修正しました。
### 既知の問題
- 【バグ】画像評価コアが未完成のため、評価エラーが発生していました。（1.1.0で修正済み）
- 【バグ】超解像画像のプレビューサイズが元の画像と異なるため、直感的な比較が難しくなっています。

### 1.0.2 (2025年4月25日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.0.2.jpg)
### 重要な変更
- 【重要】ベンチマークテストを全面的に書き直しました。
### 新機能
- 【更新】ベンチマークテストのページを更新し、デバイスの性能差をより正確に判断でき、より視覚的に魅力的になりました。
- 【修正】CPUでベンチマークテストを実行する際、異なるデバイス上のテンソルによるエラーを修正しました。
- 【修正】ベンチマークテストでCPUモデルを正確に表示できるようになりました。
- 【修正】異なるタイルサイズでモデルを実行した後、モデルが正常にアンロードされない問題を修正しました。
### 既知の問題
- 【バグ】画像評価コアが未完成のため、評価エラーが発生していました。（1.1.0で修正済み）
- 【バグ】超解像画像のプレビューサイズが元の画像と異なるため、直感的な比較が難しくなっています。
- 【バグ】モデルがない状態で外部モデルをインポートして登録した後、モデルが使用できない問題がありました。（1.1.0で修正済み）
- 【修正】ベンチマークテストを切り替えるとVRAMが正常に解放されない問題がありました。（1.1.0で修正済み）

[すべてのリリースバージョン](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/Changelog.md)

## クイックスタート
> [!NOTE]
> モデルトレーニング機能を使用しない場合や、NVIDIAグラフィックカードを使用していない場合は、最初の3項目のみのインストールで十分です。
### 環境設定
- **Python 3**
  - ダウンロード: https://www.python.org/downloads/windows/
- **FFMPEG**
  - ダウンロード: https://ffmpeg.org/download.html
- **PyTorch**
  - ダウンロード: https://pytorch.org/
- NVIDIA GPUドライバー
  - ダウンロード: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - ダウンロード: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - ダウンロード: https://developer.nvidia.com/cudnn
> [!TIP]
> 現在のPyTorchがサポートするCUDAバージョンに合わせてインストールしてください。

### リポジトリのインストール
> [!IMPORTANT]
> これは必須のステップです。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm.git
cd Nagato-Sakura-Image-Charm
pip install -r requirements.txt
```
## GUIインターフェース
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/GUI_1.png)
### GUIの起動
```shell
python main.py
```

## モデル紹介
### 鏡花（Kyouka） - アニメJPEG圧縮復元モデル
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kyouka_comparison.webp">
</p>

### 律花（Ritsuka） - アニメモザイク復元モデル
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Ritsuka_comparison.webp">
</p>


## TODO
- [ ] **高優先度：**
  - [x] クイックインストールガイド。
  - [ ] ユーザーガイド（wiki）。
  - [x] AI画像評価モデルの書き直し。

- [ ] **機能:**
  - プログラム
    - [x] モデル強度調整機能。
    - [x] モデル超解像機能。
    - [x] 計算デバイス選択機能。
    - [x] 実際のシナリオに合ったベンチマークテスト。
    - [x] 隠されたイースターエッグ。
    - [ ] GUIトレーナーの更新（第5世代 → 第7世代）。
    - [ ] 拡張プラグイン管理機能。
    
  - モデル
    - [ ] 次世代モデルアーキテクチャの開発（既存のモデルアーキテクチャの限界に達しています）。
    - [x] 第6世代《鏡花・碎象還映》Kyouka 汎用アニメ画像JPEG圧縮復元モデル。
    - [x] 第6世代《鏡花・幽映深層》Kyouka-LQ 低品質特化アニメ画像JPEG圧縮復元モデル。
    - [x] 第7世代《鏡花・霞緲輪影》Kyouka-MQ 通常品質特化アニメ画像JPEG圧縮復元モデル。
    - [x] 第7世代《断律・映格輪廻》Ritsuka-HQ 2〜4ピクセルモザイク復元モデル。
    - [x] 第7世代《界律・重編因像》Kairitsu リアル画像修復モデル。（効果が理想的ではないため、releasesには追加されていますがモデルリストには含まれていません）

  - その他
    - [ ] Discordボットバージョンのcog。
    - [ ] モデルトレーナーGithubリポジトリの設定。

## 謝辞
以下のプロジェクトと貢献者に特に感謝します：

### プロジェクト
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Quality-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)

### 貢献者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>