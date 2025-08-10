import os
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class ImageClassifier:
    """用於分類圖像並推薦適合的模型處理器"""
    def __init__(self):
        self.model = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.loaded = False
    
    def load_model(self, model_path: Optional[str] = None, labels_path: Optional[str] = None) -> bool:
        """載入分類模型和標籤文件"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_models_dir = os.path.join(base_dir, "extensions", "Nagato-Sakura-Image-Classification", "models")
            if labels_path is None:
                labels_path = os.path.join(default_models_dir, "labels.txt")
            if not os.path.exists(labels_path):
                logger.error(f"找不到標籤文件: {labels_path}")
                return False
            encodings = ['utf-8', 'big5', 'gbk', 'cp950', 'latin-1']
            for encoding in encodings:
                try:
                    with open(labels_path, 'r', encoding=encoding) as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                    logger.info(f"成功載入標籤文件: {labels_path} (使用 {encoding} 編碼)")
                    break
                except UnicodeDecodeError:
                    continue
            if not self.class_names:
                logger.error(f"無法解碼標籤文件: {labels_path}")
                return False
            if model_path is None:
                model_files = [f for f in os.listdir(default_models_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(default_models_dir, model_files[0])
                else:
                    logger.error(f"在 {default_models_dir} 中找不到 .pth 模型文件")
                    return False
            if not os.path.exists(model_path):
                logger.error(f"找不到模型文件: {model_path}")
                return False
            num_classes = len(self.class_names)
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                logger.info("使用 weights_only=True 成功載入模型")
            except TypeError:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("使用兼容模式載入模型")
                
            self.model = self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info(f"成功載入分類模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"載入分類模型時出錯: {str(e)}")
            self.model = None
            self.class_names = None
            self.loaded = False
            return False
    
    def classify_image(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        對圖像進行分類並推薦適合的處理模型
        
        參數:
            image: 圖像路徑或PIL圖像對象
            
        返回:
            包含分類結果和推薦模型的字典
        """
        if not self.loaded or self.model is None or self.class_names is None:
            success = self.load_model()
            if not success:
                return {
                    "success": False,
                    "error": "未能載入分類模型",
                    "top_class": None,
                    "top_probability": 0,
                    "results": []
                }
        
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    return {
                        "success": False,
                        "error": f"找不到圖像文件: {image}",
                        "top_class": None,
                        "top_probability": 0,
                        "results": []
                    }
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs, indices = torch.sort(probabilities, descending=True)
                probs = probs[0].cpu().numpy()
                indices = indices[0].cpu().numpy()
                results = []
                for i in range(min(len(self.class_names), 10)):
                    idx = indices[i]
                    prob = float(probs[i] * 100)
                    class_name = self.class_names[idx]
                    results.append({
                        "class": class_name,
                        "probability": prob
                    })
                return {
                    "success": True,
                    "top_class": self.class_names[indices[0]],
                    "top_probability": float(probs[0] * 100),
                    "results": results
                }
        except Exception as e:
            logger.error(f"圖像分類過程中出錯: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "top_class": None,
                "top_probability": 0,
                "results": []
            }
    
    def is_model_loaded(self) -> bool:
        """檢查模型是否已載入"""
        return self.loaded and self.model is not None and self.class_names is not None
    
    def unload_model(self) -> None:
        """卸載模型以釋放資源"""
        if self.model is not None:
            try:
                if self.device.type == 'cuda':
                    self.model.to('cpu')
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("已成功卸載圖像分類模型")
            except Exception as e:
                logger.error(f"卸載模型時出錯: {str(e)}")
        self.loaded = False