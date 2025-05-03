# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/README.md) | English | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_jp.md) \]

## Introduction
"Nagato Sakura Image Charm" is a derivative of the "Nagato Sakura Project," a graphical image enhancement and evaluation tool based on the image core of "Nagato Sakura." It supports AI super-resolution, image and video enhancement, quality assessment, and other features.

## Announcement
Due to computational resource constraints and the developer's busy schedule, development of Nagato Sakura Image Charm will be paused after version 1.1.1. Priority will be given to completing the Nagato Sakura Image Charm Trainer repository and the Discord version cog.

## Recent Changes
### 1.1.1 (May 04, 2025)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.1.jpg)
### Important Changes
- [Major] Completely rewrote the AI image scoring core model architecture, abandoning the large Transformer model in favor of a traditional CNN model, with architecture optimization for lightweight processing.
### New Features
- [New] Added edge comparison functionality to the evaluation page, allowing image edge comparison using OpenCV Canny.
- [Update] Updated the UI layout of the evaluation page, replacing the bottom preview scrolling box with a more flexible views.py style.
- [Update] Optimized the model description for Ritsuka-HQ.
- [Fix] AI image scoring model completely rewritten, now requiring only about 1MB of space for good AI scoring results.
### Known Issues
- [Bug] Super-resolution image preview size differs from original image, making intuitive comparisons difficult.
- [Bug] Plugin management feature is incomplete. Version 1.1.1 will forcibly pre-install the Nagato-Sakura-Image-Quality-Assessment image scoring functionality (approximately 1MB).

### 1.1.0 (April 28, 2025)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.1.0.jpg)
### Important Changes
- [Major] Computing device selection feature, now allowing manual forcing of CPU for image and video processing.
### New Features
- [New] Ritsuka-HQ seventh-generation anime mosaic restoration model.
- [Update] Optimized benchmark test balance, adjusted memory allocation strategies, improved VRAM allocation strategies, and optimized progress bar accuracy.
- [Update] Computing device selection now displays CPU and NVIDIA GPU models for easier multi-device switching (added NS_DeviceInfo.py).
- [Fix] Fixed issue where imported external models couldn't be used after registration when there were no models.
- [Fix] Fixed issue where switching benchmark tests would cause VRAM to not be properly released.
### Known Issues
- [Bug] Image evaluation core not completed, causing evaluation errors. (Fixed in 1.1.0)
- [Bug] Super-resolution image preview size differs from original image, making intuitive comparisons difficult.

### 1.0.2 (April 25, 2025)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.0.2.jpg)
### Important Changes
- [Major] Completely rewrote the benchmark tests.
### New Features
- [Update] Benchmark test page now better determines device performance differences and is more visually appealing.
- [Fix] Fixed benchmark test errors caused by tensors on different devices when running benchmarks on CPU.
- [Fix] Benchmark tests now correctly display CPU model.
- [Fix] Fixed issue where models weren't successfully unloaded after running with different tile sizes.
### Known Issues
- [Bug] Image evaluation core not completed, causing evaluation errors. (Fixed in 1.1.0)
- [Bug] Super-resolution image preview size differs from original image, making intuitive comparisons difficult.
- [Bug] Issue where imported external models couldn't be used after registration when there were no models. (Fixed in 1.1.0)
- [Fix] Switching benchmark tests causes VRAM to not be properly released. (Fixed in 1.1.0)

[All Release Versions](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/Changelog.md)

## Quick Start
> [!NOTE]
> If you're not using model training functionality or are not an NVIDIA GPU user, only the first three items need to be installed.
### Environment Setup
- **Python 3**
  - Download: https://www.python.org/downloads/windows/
- **FFMPEG**
  - Download: https://ffmpeg.org/download.html
- **PyTorch**
  - Download: https://pytorch.org/
- NVIDIA GPU Driver
  - Download: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - Download: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - Download: https://developer.nvidia.com/cudnn
> [!TIP]
> Please install the appropriate CUDA version according to the current PyTorch support.

### Repository Installation
> [!IMPORTANT]
> This is a required step.
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm.git
cd Nagato-Sakura-Image-Charm
pip install -r requirements.txt
```
## GUI Interface
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/GUI_1.png)
### Opening the GUI
```shell
python main.py
```

## Model Introduction
### Kyouka -《Mirror Flower》(Anime JPEG Compression Restoration Model)
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kyouka_comparison.webp">
</p>

### Ritsuka -《Breaking Law》(Anime Mosaic Restoration Model)
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Ritsuka_comparison.webp">
</p>


## To-Do List
- [ ] **High Priority:**
  - [x] Quick installation guide.
  - [ ] User guide (wiki).
  - [x] Rewrite AI image scoring model.

- [ ] **Features:**
  - Program
    - [x] Model strength adjustment functionality.
    - [x] Model super-resolution functionality.
    - [x] Computing device selection functionality.
    - [x] Realistic benchmark testing.
    - [x] Hidden easter eggs.
    - [ ] GUI trainer update (5th gen → 7th gen).
    - [ ] Plugin management functionality.
    
  - Models
    - [ ] Next-generation model architecture development (reached the limit of existing model architecture).
    - [x] 6th Gen《Mirror Flower・Fragmented Image Reflection》Kyouka general-purpose anime image JPEG compression restoration model.
    - [x] 6th Gen《Mirror Flower・Deep Shadow Projection》Kyouka-LQ low-quality specialized anime image JPEG compression restoration model.
    - [x] 7th Gen《Mirror Flower・Hazy Ring Shadow》Kyouka-MQ normal-quality specialized anime image JPEG compression restoration model.
    - [x] 7th Gen《Breaking Law・Framed Samsara》Ritsuka-HQ 2-4 pixel mosaic restoration model.
    - [x] 7th Gen《Boundary Law・Reconstructed Causality》Kairitsu realistic image restoration model. (Unsatisfactory results, added to releases but not to the model list)

  - Others
    - [ ] Discord bot version cog.
    - [ ] Model trainer GitHub repository setup.

## Acknowledgements
Special thanks to the following projects and contributors:

### Projects
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Quality-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)

### Contributors
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>
