# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_jp.md) \]

## 簡介
「長門櫻-影像魅影」是「長門櫻計畫」的衍生品，是以「長門櫻」的影像核心為基底衍生出的圖形化的影像增強與評估工具，支援 AI 超分辨率、圖片及影片增強、品質評估等功能。

## 公告
1.3.0 版本後添加了擴充插件管理器，正式移除倉庫原本的預裝擴充功能，可透過擴充功能頁面安裝插件。
1.2.1 版本後添加了對 Apple Metal Performance Shaders (MPS) 的支援，現在可以使用蘋果系統的圖形硬體加速。

## 近期變動
### 1.3.0 (2025 年 8 月 10 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.3.0.jpg)
### 重要變更
- 【重大】訓練頁面處理功能添加新的 data_processer 及影片擷取圖片的功能。
- 【重大】訓練頁面訓練模型的內核版本從 v4 升級至 v7+ 版本。
- 【重大】移除預裝的 Nagato-Sakura-Image-Quality-Assessment 及 Nagato-Sakura-Image-Classification。
- 【重大】專案報告從 assets/docs 移動至 assets/report。
### 新增功能
- 【新增】圖片處理、影片處理新增銳化功能。
- 【新增】擴充插件管理器並添加了擴充功能頁面，可在應用程式內安裝、更新及卸載擴充功能。
- 【新增】日誌記錄監視器可在應用程式內監看動作紀錄。
- 【新增】檢查更新功能可以檢查應用程式是否有可用新版本。
- 【更新】訓練頁面的資料處理功能。
- 【更新】訓練頁面的訓練器內核版本。
- 【更新】主程式的 CLI 介面添加添加日誌等級顏色方便查看日誌。
- 【更新】彩蛋變更，並添加擴充功能的可安裝擴展包。
- 【更新】下載頁面改善了縮放比例及縮放上限的問題。
- 【修復】MacOS、Linux 下載頁面文字重疊在預覽框上面。 
- 【修復】超分辨率圖像預覽大小和原始圖像不同導致無法直觀比較的問題。
- 【修復】分割畫面向左拉線條抖動問題。
### 已知問題
- 【錯誤】Linux、MacOS 同時存在兩張圖片執行完整圖片評估功能時會直接閃退的問題。

### 1.2.1 (2025 年 7 月 15 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.1.jpg)
### 重要變更
- 【重大】添加了對 Apple Metal Performance Shaders (MPS) 的支援，現在可以使用蘋果系統的硬體加速。
- 【重大】刪除了 ExtractUtility.py ，移除了解壓縮功能。
### 新增功能
- 【新增】影片處理功能添加了 NS-C 模型推薦功能。
- 【更新】優化了 NS-C 模型的調用方式，在使用完畢會自動釋放資源。
- 【更新】優化了基準測試開啟效率，在背景執行緒中收集系統資訊以避免UI阻塞。
- 【修復】模型推薦功能在部分系統及淺色模式下無法使用的問題。
- 【修復】影片處理功能，在 MacOS 完成影片處理後開啟對話框開啟資料夾閃退的問題。
- 【修復】當不存在模型時，下載模型行後跳出使用新模型對話框按否會出現，在模型清單選擇該模型時卻無法使用該模型的問題。
### 已知問題
- 【錯誤】超分辨率圖像預覽大小和原始圖像不同導致無法直觀比較。
- 【錯誤】擴充插件管理功能未完成，目前 1.2.0 版本後將強制預裝 Nagato-Sakura-Image-Quality-Assessment 影像評分功能 (約1MB)。
- 【錯誤】擴充插件管理功能未完成，目前 1.2.0 版本後將強制預裝 Nagato-Sakura-Image-Classification (NS-C)模型推薦功能 (約16MB)。
- 【錯誤】Linux、MacOS 同時存在兩張圖片執行完整圖片評估功能時會直接閃退的問題。
- 【錯誤】MacOS、Linux 下載頁面文字重疊在預覽框上面。 

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
    - [x] GUI 訓練器版本更新。
    - [x] 擴充插件管理功能。
    
  - 模型
    - [x] 第6代《鏡花・碎象還映》Kyouka 泛用型動畫圖像 JPEG 壓縮還原模型。
    - [x] 第6代《鏡花・幽映深層》Kyouka-LQ 低畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [x] 第7代《鏡花・霞緲輪影》Kyouka-MQ 普通畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [x] 第7代《斷律・映格輪廻》Ritsuka-HQ 2~4格像素點馬賽克還原模型。 
    - [x] 第7代《界律・重編因像》Kairitsu 寫實畫面修復模型。

  - 其他
    - [ ] Discord 機器人版本的 cog。
    - [x] 模型訓練器 Github 倉庫架設。

## 致謝
特別感謝以下項目和貢獻者：

### 項目
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Charm-Trainer](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)
- [Nagato-Sakura-Image-Quality-Assessment](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment)
- [Nagato-Sakura-Image-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Classification)
- [Nagato-Sakura-Bounce-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Bounce-py)

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>