# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_jp.md) \]

## 簡介
「長門櫻-影像魅影」是「長門櫻計畫」的衍生品，是以「長門櫻」的影像核心為基底衍生出的圖形化的影像增強與評估工具，支援 AI 超分辨率、圖片及影片增強、品質評估等功能。

## 公告
不再每個小版本更新預覽圖了，以後只有大版本才會有提更預覽更新。

## 近期變動
### 1.0.0 (2025 年 4 月 19 日) 版本代號: Kyouka (鏡花)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.0.0.jpg)
### 重要變更
- 【重大】首個發布版本，其實 1.0.0 版4月8日就完成了但還在決定版本命名及其他細項所以一直沒發佈。
- 【重大】全面改寫模型的存取機制，從開啟時載入改為要使用時再載入，減少資源開銷。 (NS_ModelManager.py)
- 【重大】改寫了影像顯示核心，現在能使用並排顯示、分割顯示、單獨顯示，且能使用滑鼠對影像放大及拖動。 (views.py)
### 新增功能
- 【新增】新增了模型選單功能，可以快速切換模型 (適用於:image_tab.py、video_tab.py)
- 【新增】新增了影片處理頁面，更好的影片處理 GUI ，更細的影片處理功能，包含輸出分辨率、裁切方式、編碼器選像、影片封裝格式、幀預覽選項...等。
- 【新增】新增了訓練模型頁面，添加了第1代資料處理器以及第5代 NS-IC 模型訓練器。
- 【新增】新增了影像評估頁面，添加 PSNR 、 SSIM 、差異圖及色彩直方圖等參數方便用戶快速評斷圖片。
- 【新增】新增了模型下載功能，用戶不再需要從官網下載模型。 
### 已知問題
- 【錯誤】在沒有模型時下載模型切換模型會顯示錯誤，需要手動重新切模型。 (1.0.1 已修正)
- 【錯誤】刪除模型頁面模型大小顯示錯誤。 (1.0.1 已修正)
- 【錯誤】影像評估核心尚未完成，導致評估錯誤。

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
cd Nagato-Sakura-Image-Charm-py
pip install -r requirements.txt
```
## GUI 介面
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/GUI.png)
### 開啟GUI
```shell
python main.py
```

## 待辦事項
- [ ] **高優先度：**
  - [x] 快速安裝指南。
  - [ ] 使用指南(wiki)。
  - [ ] 全部改寫影像評估頁面的模型。 (Nagato-Sakura-Image-Quality-Classification)

- [ ] **功能:**
  - 程式相關
    - [x] 模型強度調整。
    - [x] 圖片超分辨率。
    - [ ] 影片超分辨率。
    
  - 模型
    - [x] 第6代《鏡花・碎象還映》Kyouka 泛用型動畫圖像 JPEG 壓縮還原模型。
    - [x] 第6代《鏡花・幽映深層》Kyouka-LQ 低畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [ ] 第6代《鏡花・霞緲輪影》Kyouka-MQ 普通畫質特化動畫圖像 JPEG 壓縮還原模型。
    - [ ] 第6代《斷律・映格輪廻》Ritsuka 馬賽克還原模型。 
    - [ ] 第6代《界律・重編因像》Kairitsu 寫實畫面修復模型。

## 致謝
特別感謝以下項目和貢獻者：

### 項目

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Discord-Bot-py" />
</a>
