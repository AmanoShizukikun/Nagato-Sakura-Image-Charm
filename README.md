# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_jp.md) \]

## 簡介
「長門櫻-影像魅影」是「長門櫻計畫」的衍生品，是以「長門櫻」的影像核心為基底衍生出的圖形化的影像增強與評估工具，支援 AI 超分辨率、圖片及影片增強、品質評估等功能。

## 公告
由於算力短缺加上開發者最近相較忙碌，將在1.1.1版本後會暫緩長門櫻-影像魅影的開發，優先完成長門櫻-影像魅影訓練器相關的倉庫以及 Discord 版的 cog。

## 近期變動
### 1.2.0 (2025 年 5 月 20 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.0.jpg)
### 重要變更
- 【重大】正式將 Nagato-Sakura-Image-Classification (NS-C)圖像分類模型作為模型推薦外掛引入系統。
- 【重大】更改了 Nagato-Sakura-Image-Classification (NS-C)的模型架構，從原本的 ResNet-19 改為更加輕量化的EfficientNet-B0。
### 新增功能
- 【新增】圖片處理頁面以及圖像評估頁面現在添加 NS-C 功能，可根據輸入圖像推薦最適合的的 NS-IQE 模型。
- 【新增】Kairitsu 第九代泛用型寫實 JPEG 壓縮還原模型。
### 已知問題
- 【錯誤】超分辨率圖像預覽大小和原始圖像不同導致無法直觀比較。
- 【錯誤】擴充插件管理功能未完成，目前 1.2.0 版本將強制預裝 Nagato-Sakura-Image-Quality-Assessment 影像評分功能 (約1MB)。
- 【錯誤】擴充插件管理功能未完成，目前 1.2.0 版本將強制預裝 Nagato-Sakura-Image-Classification (NS-C)模型推薦功能 (約16MB)。

### 1.1.1 (2025 年 5 月 04 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.1.jpg)
### 重要變更
- 【重大】完全重寫了 AI 影像評分核心的模型架構，棄用了龐大的 Transformer 模型改回傳統的 CNN 模型，並對模型架構進行了輕量化處理。
### 新增功能
- 【新增】評估頁面添加邊緣比較功能，能透過 OpenCV Canny 畫出圖像邊緣進行兩張圖的比較。
- 【更新】評估頁面更新 UI 版面，原本下方預覽框由滾動框改為操作更加自由的 views.py 型式。
- 【更新】優化了 Ritsuka-HQ 的模型描述。
- 【修復】AI 影像評分，全面改寫了模型，現在只需要 1MB 左右的空間就能做到不錯的AI評分效果。
### 已知問題
- 【錯誤】超分辨率圖像預覽大小和原始圖像不同導致無法直觀比較。
- 【錯誤】擴充插件管理功能未完成，目前 1.1.1 版本將強制預裝 Nagato-Sakura-Image-Quality-Assessment 影像評分功能 (約1MB)。

### 1.1.0 (2025 年 4 月 28 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.0.jpg)
### 重要變更
- 【重大】計算設備選擇功能，現在可以手動強制開啟CPU來進行圖像、影像處理。
### 新增功能
- 【新增】Ritsuka-HQ 第七代動漫馬賽克還原模型。
- 【更新】優化基準測試的平衡性並調整記憶體的調用策略及改良顯存調用策略及優化進度條準確度。
- 【更新】計算設備選擇能直接看到 CPU 及 NVIDIA GPU 的型號方便多設備切換。 (新增 NS_DeviceInfo.py)
- 【修復】在沒有模型的時候匯入外部模型後註冊卻無法使用模型的問題。
- 【修復】切換基準測試會導致顯存未成功釋放的問題。
### 已知問題
- 【錯誤】影像評估核心尚未完成，導致評估錯誤。 (1.1.0 已修正)
- 【錯誤】超分辨率圖像預覽大小和原始圖像不同導致無法直觀比較。


[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/Changelog.md)

## 快速開始
> [!NOTE]
> 如果沒有使用到模型訓練功能或著非 NVIDIA 顯卡用戶可只安裝前三項即可。
### 環境設置
- **Python 3**
  - 下載: https://www.python.org/downloads/windows/
- **FFMPEG**
  - 下載: https://ffmpeg.org/download.html
- **PyTorch**
  - 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
  - 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - 下載: https://developer.nvidia.com/cudnn
> [!TIP]
> 請按照當前 PyTorch 支援安裝對應的 CUDA 版本。

### 安裝倉庫
> [!IMPORTANT]
> 此為必要步驟。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm.git
cd Nagato-Sakura-Image-Charm
pip install -r requirements.txt
```
## GUI 介面
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/GUI_1.png)
### 開啟GUI
```shell
python main.py
```

## 模型介紹
### Kyouka -《鏡花》- 碎裂之象，在鏡中重構原初之姿。鏡中之花非虛幻，碎裂之象皆可還映。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kyouka_comparison.webp">
</p>

- ### 模型版本: v6 (Kyouka-MQ 為 v7)
- ### 訓練參數: 10K
- ### 適用場景: 動漫JEPG壓縮還原

| Model            |   Type    |                              Download                                 |  
|:----------------:|:---------:|:---------------------------------------------------------------------:|
| Kyouka-v6-314    |   NULL    | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-LQ-v6-310 |    LQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-MQ-v7-349 |    MQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
<br/>

###  Kairitsu-《界律》- 幻象之律重編破碎時因，塵封記憶再現為清晰輪廓。。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kairitsu_comparison.webp">
</p>

- ### 模型版本: v9 
- ### 訓練參數: 25K
- ### 適用場景: 寫實JEPG壓縮還原
<br/>

### Ritsuka -《斷律》- 格狀的影塊，是序斷的殘響。在光與影的律動中，復甦真實之形。
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Ritsuka_comparison.webp">
</p>

- ### 模型版本: v7
- ### 訓練參數: 10K
- ### 適用場景: 動漫馬賽克還原
<br/>


## 待辦事項
- [ ] **高優先度：**
  - [x] 快速安裝指南。
  - [ ] 使用指南(wiki)。
  - [x] 改寫AI圖像評分模型。

- [ ] **功能:**
  - 程式
    - [x] 模型強度調整功能。
    - [x] 模型超分辨率功能。
    - [x] 計算設備選擇功能。
    - [x] 符合實際場景的基準測試。
    - [x] 隱藏的小彩蛋。
    - [ ] GUI 訓練器版本更新。
    - [ ] 擴充插件管理功能。
    
  - 模型
    - [ ] 引入 VGG 損失函數。
    - [ ] 次世代模型架構開發(已達到現有模型架構極限)。
    - [x] 第6代《鏡花・碎象還映》Kyouka 泛用型動畫圖像 JPEG 壓縮還原模型。
    - [x] 第6代《鏡花・幽映深層》Kyouka-LQ 低畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [x] 第7代《鏡花・霞緲輪影》Kyouka-MQ 普通畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [x] 第7代《斷律・映格輪廻》Ritsuka-HQ 2~4格像素點馬賽克還原模型。 
    - [x] 第7代《界律・重編因像》Kairitsu 寫實畫面修復模型。 (效果不理想只添加到releases不添加進模型清單)

  - 其他
    - [ ] Discord 機器人版本的 cog。
    - [x] 模型訓練器 Github 倉庫架設。

## 致謝
特別感謝以下項目和貢獻者：

### 項目
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Charm-Trainer](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)
- [Nagato-Sakura-Image-Quality-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)
- [Nagato-Sakura-Image-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Classification)

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>
