# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/README.md) | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_en.md) | 日本語 \]

## 概要
「長門桜-イメージチャーム」は「長門桜プロジェクト」の派生品であり、「長門桜」の画像コアをベースにした画像強化・評価のためのグラフィカルツールです。AI超解像、画像・動画の強化、品質評価などの機能をサポートしています。

## お知らせ
Apple Metal Performance Shaders (MPS) のサポートを追加しました。これにより、Appleシステムのハードウェアアクセラレーションが利用できるようになりました。
NS-IC-Kairitsu-v7pro-370 および NS-IC-Ritsuka-LQ-v7-228(Beta) は内蔵モデルリストに含まれていません。Github model space からモデルをダウンロードしてください。

## 最近の変更
### 1.2.1 (2025年7月15日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.1.jpg)
### 重要な変更
- 【重要】Apple Metal Performance Shaders (MPS) のサポートを追加しました。これにより、Appleシステムのハードウェアアクセラレーションが利用できるようになりました。
- 【重要】ExtractUtility.py を削除し、解凍機能を廃止しました。
### 新機能
- 【新規】動画処理機能にNS-Cモデル推薦機能を追加しました。
- 【更新】NS-Cモデルの呼び出し方式を最適化し、使用後にリソースが自動的に解放されるようになりました。
- 【更新】ベンチマークテストの起動効率を最適化し、UIのブロックを回避するためにバックグラウンドスレッドでシステム情報を収集するようになりました。
- 【修正】モデル推薦機能が一部のシステムやライトモードで使用できない問題を修正しました。
- 【修正】動画処理機能で、MacOSで動画処理完了後にフォルダを開くダイアログでクラッシュする問題を修正しました。
- 【修正】モデルが存在しない状態でモデルをダウンロードした後、新しいモデルのダイアログで「いいえ」を選択すると、モデルリストでそのモデルを選択しても使用できない問題を修正しました。
### 既知の問題
- 【バグ】超解像画像のプレビューサイズが元の画像と異なるため、直感的な比較が難しくなっています。
- 【バグ】拡張プラグイン管理機能が未完成のため、バージョン1.2.0以降ではNagato-Sakura-Image-Quality-Assessment画像評価機能（約1MB）が強制的にプリインストールされます。
- 【バグ】拡張プラグイン管理機能が未完成のため、バージョン1.2.0以降ではNagato-Sakura-Image-Classification (NS-C)モデル推薦機能（約16MB）が強制的にプリインストールされます。
- 【バグ】MacOSで2つの画像が同時に存在する状態で完全画像評価機能を実行すると直接クラッシュする問題があります。
- 【バグ】Linuxで2つの画像が同時に存在する状態で完全画像評価機能を実行すると、フォントの不足により直接クラッシュする問題があります。
- 【バグ】MacOS、Linuxでダウンロードページの文字がプレビューボックスに重なって表示される問題があります。

### 1.2.0 (2025年5月20日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.0.jpg)
### 重要な変更
- 【重要】Nagato-Sakura-Image-Classification (NS-C)画像分類モデルを正式にモデル推薦プラグインとしてシステムに導入しました。
- 【重要】Nagato-Sakura-Image-Classification (NS-C)のモデル構造を変更し、元のResNet-19からより軽量なEfficientNet-B0に変更しました。
### 新機能
- 【新規】画像処理ページおよび画像評価ページにNS-C機能を追加し、入力画像に基づいて最適なNS-IQEモデルを推薦できるようになりました。
- 【新規】Kairitsu第九世代汎用リアルJPEG圧縮復元モデル。
### 既知の問題
- 【バグ】超解像画像のプレビューサイズが元の画像と異なるため、直感的な比較が難しくなっています。
- 【バグ】拡張プラグイン管理機能が未完成のため、バージョン1.2.0ではNagato-Sakura-Image-Quality-Assessment画像評価機能（約1MB）が強制的にプリインストールされます。
- 【バグ】拡張プラグイン管理機能が未完成のため、バージョン1.2.0ではNagato-Sakura-Image-Classification (NS-C)モデル推薦機能（約16MB）が強制的にプリインストールされます。

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
### 鏡花（Kyouka） -《鏡花》- 砕け散った姿を、鏡の中で元の形に再構築する。鏡の中の花は幻ではなく、砕け散った姿は皆映し出される。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kyouka_comparison.webp">
</p>

- ### モデルバージョン: v6 (Kyouka-MQ は v7)
- ### 訓練パラメータ: 10K
- ### 適用シーン: アニメJPEG圧縮復元

| Model            |   Type    |                              Download                                 |  
|:----------------:|:---------:|:---------------------------------------------------------------------:|
| Kyouka-v6-314    |   NULL    | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-LQ-v6-310 |    LQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-MQ-v7-349 |    MQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
<br/>

### 界律（Kairitsu） -《界律》- 幻象の法則が砕けた時の因果を再編し、封じられた記憶が鮮明な輪郭として再現される。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kairitsu_comparison.webp">
</p>

- ### モデルバージョン: v9 
- ### 訓練パラメータ: 25K
- ### 適用シーン: リアルJPEG圧縮復元
<br/>

### 律花（Ritsuka） -《断律》- 格子状の影のブロックは、秩序断絶の残響である。光と影のリズムの中で、真実の形が復活する。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Ritsuka_comparison.webp">
</p>

- ### モデルバージョン: v7
- ### 訓練パラメータ: 10K
- ### 適用シーン: アニメモザイク復元
<br/>


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
    - [ ] GUIトレーナーの更新。
    - [ ] 拡張プラグイン管理機能。
    
  - モデル
    - [x] VGG損失関数の導入。
    - [x] 第6世代《鏡花・砕象還映》Kyouka 汎用アニメ画像JPEG圧縮復元モデル。
    - [x] 第6世代《鏡花・幽映深層》Kyouka-LQ 低品質特化アニメ画像JPEG圧縮復元モデル。
    - [x] 第7世代《鏡花・霞緲輪影》Kyouka-MQ 通常品質特化アニメ画像JPEG圧縮復元モデル。
    - [x] 第7世代《断律・映格輪廻》Ritsuka-HQ 2〜4ピクセルモザイク復元モデル。
    - [x] 第7世代《界律・重編因像》Kairitsu リアル画像修復モデル。（効果が理想的ではないため、releasesには追加されていますがモデルリストには含まれていません）

  - その他
    - [ ] Discordボットバージョンのcog。
    - [x] モデルトレーナーGithubリポジトリの設置。

## 謝辞
以下のプロジェクトと貢献者に特に感謝します：

### プロジェクト
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Charm-Trainer](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)
- [Nagato-Sakura-Image-Quality-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)
- [Nagato-Sakura-Image-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Classification)

### 貢献者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>