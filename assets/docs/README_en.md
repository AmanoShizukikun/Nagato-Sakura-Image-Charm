# Nagato-Sakura-Image-Charm

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/README.md) | English | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/docs/README_jp.md) \]

## Introduction
"Nagato Sakura Image Charm" is a derivative of the "Nagato Sakura Project," a graphical image enhancement and evaluation tool based on the image core of "Nagato Sakura." It supports AI super-resolution, image and video enhancement, quality assessment, and other features.

## Announcement
Added support for Apple Metal Performance Shaders (MPS), now you can use hardware acceleration on Apple systems.
NS-IC-Kairitsu-v7pro-370 and NS-IC-Ritsuka-LQ-v7-228(Beta) are not included in the built-in model list, please download models from Github model space.

## Recent Changes
### 1.2.1 (July 15, 2025)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.1.jpg)
### Important Changes
- [Major] Added support for Apple Metal Performance Shaders (MPS), now you can use hardware acceleration on Apple systems.
- [Major] Removed ExtractUtility.py, eliminated decompression functionality.
### New Features
- [New] Video processing functionality now includes NS-C model recommendation feature.
- [Update] Optimized NS-C model invocation method, resources are automatically released after use.
- [Update] Optimized benchmark test startup efficiency, collecting system information in background threads to avoid UI blocking.
- [Fix] Fixed issue where model recommendation feature couldn't be used on some systems and in light mode.
- [Fix] Fixed issue where video processing would crash when opening folder dialog after completing video processing on MacOS.
- [Fix] Fixed issue where selecting a model from the model list after downloading would result in being unable to use that model when no models exist and clicking "No" in the new model dialog.
### Known Issues
- [Bug] Super-resolution image preview size differs from original image, making intuitive comparisons difficult.
- [Bug] Plugin management feature is incomplete. Version 1.2.0 and later will forcibly pre-install the Nagato-Sakura-Image-Quality-Assessment image scoring functionality (approximately 1MB).
- [Bug] Plugin management feature is incomplete. Version 1.2.0 and later will forcibly pre-install the Nagato-Sakura-Image-Classification (NS-C) model recommendation functionality (approximately 16MB).
- [Bug] MacOS crashes when running full image evaluation with two images present simultaneously.
- [Bug] Linux crashes due to missing fonts when running full image evaluation with two images present simultaneously.
- [Bug] MacOS/Linux download page text overlaps on preview box.

### 1.2.0 (May 20, 2025)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/preview/1.2.0.jpg)
### Important Changes
- [Major] Officially introduced Nagato-Sakura-Image-Classification (NS-C) image classification model as a model recommendation plugin into the system.
- [Major] Changed the model architecture of Nagato-Sakura-Image-Classification (NS-C), from the original ResNet-19 to a more lightweight EfficientNet-B0.
### New Features
- [New] Image processing page and image evaluation page now include NS-C functionality, which can recommend the most suitable NS-IQE model based on input images.
- [New] Kairitsu ninth-generation general-purpose realistic JPEG compression restoration model.
### Known Issues
- [Bug] Super-resolution image preview size differs from original image, making intuitive comparisons difficult.
- [Bug] Plugin management feature is incomplete. Version 1.2.0 will forcibly pre-install the Nagato-Sakura-Image-Quality-Assessment image scoring functionality (approximately 1MB).
- [Bug] Plugin management feature is incomplete. Version 1.2.0 will forcibly pre-install the Nagato-Sakura-Image-Classification (NS-C) model recommendation functionality (approximately 16MB).

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
### Kyouka -《Mirror Flower》- Shattered reflections, reconstructed in the mirror to their original form. The flower in the mirror is not an illusion, all shattered reflections can be restored.
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kyouka_comparison.webp">
</p>

- ### Model Version: v6 (Kyouka-MQ is v7)
- ### Training Parameters: 10K
- ### Application Scenario: Anime JPEG Compression Restoration

| Model            |   Type    |                              Download                                 |  
|:----------------:|:---------:|:---------------------------------------------------------------------:|
| Kyouka-v6-314    |   NULL    | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-LQ-v6-310 |    LQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
| Kyouka-MQ-v7-349 |    MQ     | [🤗 Huggingface](https://huggingface.co/AmanoShizukikun/NS-IC-Kyouka) |
<br/>

###  Kairitsu -《Boundary Law》- The law of illusion rewrites the broken cause and effect, sealed memories reappear as clear outlines.
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Kairitsu_comparison.webp">
</p>

- ### Model Version: v9 
- ### Training Parameters: 25K
- ### Application Scenario: Realistic JPEG Compression Restoration
<br/>

### Ritsuka -《Breaking Law》- Grid-like shadow blocks are the broken echoes of sequence. In the rhythm of light and shadow, the true form is revived.
<p align="center">
  <img src="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/blob/main/assets/samples/Ritsuka_comparison.webp">
</p>

- ### Model Version: v7
- ### Training Parameters: 10K
- ### Application Scenario: Anime Mosaic Restoration
<br/>


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
    - [ ] GUI trainer update.
    - [ ] Plugin management functionality.
    
  - Models
    - [x] Introduced VGG loss function.
    - [x] 6th Gen《Mirror Flower・Fragmented Image Reflection》Kyouka general-purpose anime image JPEG compression restoration model.
    - [x] 6th Gen《Mirror Flower・Deep Shadow Projection》Kyouka-LQ low-quality specialized anime image JPEG compression restoration model.
    - [x] 7th Gen《Mirror Flower・Hazy Ring Shadow》Kyouka-MQ normal-quality specialized anime image JPEG compression restoration model.
    - [x] 7th Gen《Breaking Law・Framed Samsara》Ritsuka-HQ 2-4 pixel mosaic restoration model.
    - [x] 7th Gen《Boundary Law・Reconstructed Causality》Kairitsu realistic image restoration model. (Unsatisfactory results, added to releases but not to the model list)

  - Others
    - [ ] Discord bot version cog.
    - [x] Model trainer GitHub repository setup.

## Acknowledgements
Special thanks to the following projects and contributors:

### Projects
- [Nagato-Sakura-Discord-Bot-py](https://github.com/AmanoShizukikun/Nagato-Sakura-Discord-Bot-py)
- [Nagato-Sakura-Image-Charm-Trainer](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)
- [Nagato-Sakura-Image-Quality-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)
- [Nagato-Sakura-Image-Classification](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Classification)

### Contributors
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Charm" />
</a>
