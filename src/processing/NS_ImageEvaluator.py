import numpy as np
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import lru_cache
from typing import Tuple, Dict, Optional, Union, Any

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 


def _ensure_numpy(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise TypeError(f"不支援的圖片型態: {type(img)}")

def _match_dimensions(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img1.shape != img2.shape:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]
    return img1, img2

def _convert_to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif img.ndim == 2:
        return img 
    else:
        raise ValueError(f"不支援的圖片形狀進行灰階轉換: {img.shape}")

def _pil_to_qimage(pil_img: Image.Image) -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"轉換PIL至BytesIO時發生錯誤: {e}")
        return None

class ImageEvaluator:
    CANNY_SIGMA: float = 0.33 
    def __init__(self, max_cache_size: int = 16):
        self._detect_edges_cached = lru_cache(maxsize=max_cache_size)(self._detect_edges_cv2)

    def calculate_psnr(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> float:
        np_img1 = _ensure_numpy(img1)
        np_img2 = _ensure_numpy(img2)
        np_img1, np_img2 = _match_dimensions(np_img1, np_img2)
        return compare_psnr(np_img1, np_img2, data_range=255)

    def calculate_ssim(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> float:
        np_img1 = _ensure_numpy(img1)
        np_img2 = _ensure_numpy(img2)
        np_img1, np_img2 = _match_dimensions(np_img1, np_img2)
        multichannel = np_img1.ndim == 3 and np_img1.shape[2] in [3, 4]
        channel_axis = 2 if multichannel else None
        if multichannel and np_img1.shape[2] == 4:
             np_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGBA2RGB)
             np_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGBA2RGB)
             channel_axis = 2 
        return compare_ssim(np_img1, np_img2, channel_axis=channel_axis, data_range=255)

    def calculate_mse(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> float:
        np_img1 = _ensure_numpy(img1).astype(np.float32) / 255.0
        np_img2 = _ensure_numpy(img2).astype(np.float32) / 255.0
        np_img1, np_img2 = _match_dimensions(np_img1, np_img2)
        return np.mean((np_img1 - np_img2) ** 2)

    def generate_difference_map(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        np_img1 = _ensure_numpy(img1).astype(np.float32) / 255.0
        np_img2 = _ensure_numpy(img2).astype(np.float32) / 255.0
        np_img1, np_img2 = _match_dimensions(np_img1, np_img2)
        diff = np.abs(np_img1 - np_img2)
        if diff.ndim == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        diff_enhanced = np.clip(diff_gray * 5.0, 0, 1.0) 
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        im = ax.imshow(diff_enhanced, cmap='jet')
        cbar = fig.colorbar(im, label='差異程度')
        cbar.ax.tick_params(labelsize=10)
        ax.set_title('影像差異熱力圖', fontsize=14, fontweight='bold')
        ax.axis('off')
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
            plt.close(fig) 
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"產生差異圖時發生錯誤: {e}")
            plt.close(fig) 
            return None

    def get_image_histograms(self, img: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        np_img = _ensure_numpy(img)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
             if np_img.ndim == 2: 
                 np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
             elif np_img.ndim == 3 and np_img.shape[2] == 4: 
                 np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
             else:
                 print("直方圖生成需要RGB圖像。")
                 return None
        colors = ('red', 'green', 'blue')
        channel_names = ('紅色', '綠色', '藍色')
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
        fig.suptitle('RGB 色彩通道直方圖', fontsize=16, fontweight='bold')
        for i, (col, name) in enumerate(zip(colors, channel_names)):
            hist = cv2.calcHist([np_img], [i], None, [256], [0, 256])
            axes[i].plot(hist, color=col, alpha=0.9, linewidth=2)
            axes[i].set_title(f'{name}通道', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('像素數量', fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.5)
            axes[i].set_xlim([0, 256])
            axes[i].tick_params(labelsize=9)

        axes[2].set_xlabel('像素值 (0-255)', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"產生RGB直方圖時發生錯誤: {e}")
            plt.close(fig)
            return None

    def get_hsv_histograms(self, img: Union[Image.Image, np.ndarray]) -> Optional[Image.Image]:
        np_img = _ensure_numpy(img)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            if np_img.ndim == 3 and np_img.shape[2] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
            else:
                print("HSV直方圖生成需要RGB圖像。")
                return None
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False, dpi=300)
        fig.suptitle('HSV 色彩空間直方圖', fontsize=16, fontweight='bold')
        axes[0].plot(h_hist, color='purple', linewidth=2)
        axes[0].set_title('色調 (H)', fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 180])
        axes[0].set_ylabel('像素數量', fontsize=10)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].tick_params(labelsize=9)
        
        axes[1].plot(s_hist, color='cyan', linewidth=2)
        axes[1].set_title('飽和度 (S)', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 256])
        axes[1].set_ylabel('像素數量', fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].tick_params(labelsize=9)
        
        axes[2].plot(v_hist, color='gray', linewidth=2)
        axes[2].set_title('亮度 (V)', fontsize=12, fontweight='bold')
        axes[2].set_xlim([0, 256])
        axes[2].set_xlabel('值', fontsize=10)
        axes[2].set_ylabel('像素數量', fontsize=10)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].tick_params(labelsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"產生HSV直方圖時發生錯誤: {e}")
            plt.close(fig)
            return None

    def _detect_edges_cv2(self, img_tuple: Tuple[bytes, Optional[int], Optional[int]]) -> np.ndarray:
        img_bytes, threshold1, threshold2 = img_tuple
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        gray = _convert_to_gray(img)
        if threshold1 is None or threshold2 is None:
            v = np.median(gray)
            lower = int(max(0, (1.0 - self.CANNY_SIGMA) * v))
            upper = int(min(255, (1.0 + self.CANNY_SIGMA) * v))
            threshold1, threshold2 = lower, upper
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges

    def detect_edges(self, img: Union[Image.Image, np.ndarray], threshold1: Optional[int] = None, threshold2: Optional[int] = None) -> Optional[Image.Image]:
        np_img = _ensure_numpy(img)
        _, img_encoded = cv2.imencode('.png', np_img)
        img_bytes = img_encoded.tobytes()
        try:
            edges = self._detect_edges_cached((img_bytes, threshold1, threshold2))
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            ax.imshow(edges, cmap='gray')
            ax.set_title('邊緣檢測結果', fontsize=14, fontweight='bold')
            ax.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"檢測邊緣時發生錯誤: {e}")
            if 'fig' in locals() and fig: plt.close(fig)
            return None

    def compare_edges(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> Tuple[Optional[Image.Image], Optional[float]]:
        np_img1 = _ensure_numpy(img1)
        np_img2 = _ensure_numpy(img2)
        _, img1_encoded = cv2.imencode('.png', np_img1)
        img1_bytes = img1_encoded.tobytes()
        _, img2_encoded = cv2.imencode('.png', np_img2)
        img2_bytes = img2_encoded.tobytes()
        try:
            edges1 = self._detect_edges_cached((img1_bytes, None, None))
            edges2 = self._detect_edges_cached((img2_bytes, None, None))
            edges1, edges2 = _match_dimensions(edges1, edges2) 
            edge_diff = cv2.bitwise_xor(edges1, edges2)
            edge_similarity = 1.0 - np.count_nonzero(edge_diff) / max(edge_diff.size, 1)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
            fig.suptitle('邊緣比較', fontsize=16, fontweight='bold')
            axes[0].imshow(edges1, cmap='gray')
            axes[0].set_title('圖A 邊緣', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(edges2, cmap='gray')
            axes[1].set_title('圖B 邊緣', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            im = axes[2].imshow(edge_diff, cmap='hot')
            axes[2].set_title(f'邊緣差異 (相似度: {edge_similarity:.3f})', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            edge_comparison_img = Image.open(buf)
            return edge_comparison_img, edge_similarity
        except Exception as e:
            print(f"比較邊緣時發生錯誤: {e}")
            if 'fig' in locals() and fig: plt.close(fig)
            return None, None

    def calculate_image_hash_similarity(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray]) -> Dict[str, float]:
        np_img1 = _ensure_numpy(img1)
        np_img2 = _ensure_numpy(img2)
        gray1 = _convert_to_gray(np_img1)
        gray2 = _convert_to_gray(np_img2)
        small1 = cv2.resize(gray1, (32, 32), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(gray2, (32, 32), interpolation=cv2.INTER_AREA)
        dct1 = cv2.dct(np.float32(small1))
        dct2 = cv2.dct(np.float32(small2))
        dct1_block = dct1[1:9, 1:9]
        dct2_block = dct2[1:9, 1:9]
        avg1 = np.mean(dct1_block)
        avg2 = np.mean(dct2_block)
        hash1 = (dct1_block > avg1).flatten()
        hash2 = (dct2_block > avg2).flatten()
        hamming_distance = np.sum(hash1 != hash2)
        max_bits = hash1.size
        phash_similarity = 1.0 - hamming_distance / max(max_bits, 1)
        try:
            _, img1_encoded = cv2.imencode('.png', np_img1)
            img1_bytes = img1_encoded.tobytes()
            _, img2_encoded = cv2.imencode('.png', np_img2)
            img2_bytes = img2_encoded.tobytes()
            edges1_full = self._detect_edges_cached((img1_bytes, None, None))
            edges2_full = self._detect_edges_cached((img2_bytes, None, None))
            edges1_resized = cv2.resize(edges1_full, (16, 16), interpolation=cv2.INTER_AREA)
            edges2_resized = cv2.resize(edges2_full, (16, 16), interpolation=cv2.INTER_AREA)
            edge_hash1 = (edges1_resized > 127).flatten()
            edge_hash2 = (edges2_resized > 127).flatten()
            edge_distance = np.sum(edge_hash1 != edge_hash2)
            edge_similarity = 1.0 - edge_distance / max(edge_hash1.size, 1)
        except Exception as e:
            print(f"警告：邊緣雜湊計算失敗: {e}")
            edge_similarity = 0.0
        avg_similarity = (phash_similarity + edge_similarity) / 2.0
        return {
            'phash_similarity': phash_similarity,
            'edge_hash_similarity': edge_similarity,
            'avg_hash_similarity': avg_similarity
        }

    def color_analysis(self, img: Union[Image.Image, np.ndarray]) -> Dict[str, float]:
        np_img = _ensure_numpy(img)
        stats = {}
        if np_img.ndim == 3 and np_img.shape[2] >= 3:
            stats['r_mean'] = float(np.mean(np_img[:,:,0]))
            stats['g_mean'] = float(np.mean(np_img[:,:,1]))
            stats['b_mean'] = float(np.mean(np_img[:,:,2]))
            stats['r_std'] = float(np.std(np_img[:,:,0]))
            stats['g_std'] = float(np.std(np_img[:,:,1]))
            stats['b_std'] = float(np.std(np_img[:,:,2]))
            if np_img.shape[2] == 4:
                rgb_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
            else:
                rgb_img = np_img
        elif np_img.ndim == 2:
            stats['brightness_mean'] = float(np.mean(np_img))
            stats['brightness_std'] = float(np.std(np_img))
            rgb_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB) 
        else:
            print("不支援的圖片格式進行完整色彩分析。")
            return stats
        try:
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            stats['h_mean'] = float(np.mean(hsv_img[:,:,0]))
            stats['s_mean'] = float(np.mean(hsv_img[:,:,1]))
            stats['v_mean'] = float(np.mean(hsv_img[:,:,2])) 
            stats['h_std'] = float(np.std(hsv_img[:,:,0]))
            stats['s_std'] = float(np.std(hsv_img[:,:,1]))
            stats['v_std'] = float(np.std(hsv_img[:,:,2]))
            stats['brightness'] = stats['v_mean'] 
        except Exception as e:
            print(f"HSV分析失敗: {e}")
        try:
            lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            stats['l_mean'] = float(np.mean(lab_img[:,:,0]))
            stats['a_mean'] = float(np.mean(lab_img[:,:,1]))
            stats['b_lab_mean'] = float(np.mean(lab_img[:,:,2]))
            stats['colorfulness'] = float(np.std(lab_img[:,:,1]) + np.std(lab_img[:,:,2]) + \
                                     0.3 * np.sqrt(np.mean(lab_img[:,:,1])**2 + np.mean(lab_img[:,:,2])**2))
        except Exception as e:
            print(f"LAB分析失敗: {e}")
        return stats

    def evaluate_region(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray], region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict[str, float]]:
        np_img1 = _ensure_numpy(img1)
        np_img2 = _ensure_numpy(img2)
        np_img1, np_img2 = _match_dimensions(np_img1, np_img2)
        if region:
            x, y, w, h = region
            max_h, max_w = np_img1.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(max_w, x + w), min(max_h, y + h)
            if x2 <= x1 or y2 <= y1:
                print("指定的區域無效。")
                return None
            img1_region = np_img1[y1:y2, x1:x2]
            img2_region = np_img2[y1:y2, x1:x2]
        else:
            img1_region = np_img1
            img2_region = np_img2

        if img1_region.size == 0 or img2_region.size == 0:
             print("裁剪後的區域為空。")
             return None
        try:
            region_psnr = self.calculate_psnr(img1_region, img2_region)
            region_ssim = self.calculate_ssim(img1_region, img2_region)
            region_mse = self.calculate_mse(img1_region, img2_region)
            return {
                'region_psnr': region_psnr,
                'region_ssim': region_ssim,
                'region_mse': region_mse
            }
        except Exception as e:
            print(f"評估區域時發生錯誤: {e}")
            return None

    def clear_cache(self):
        self._detect_edges_cached.cache_clear()
        print("邊緣檢測快取已清除。")

    def evaluate_images(self, img1: Union[Image.Image, np.ndarray], img2: Union[Image.Image, np.ndarray], advanced: bool = False) -> Dict[str, Any]:
        if img1 is None or img2 is None:
            return {"error": "需要兩張有效的圖像進行比較"}
        results: Dict[str, Any] = {}
        try:
            results["psnr"] = self.calculate_psnr(img1, img2)
            results["ssim"] = self.calculate_ssim(img1, img2)
            results["mse"] = self.calculate_mse(img1, img2)
            results["difference_map"] = self.generate_difference_map(img1, img2)
            results["histogram_img1"] = self.get_image_histograms(img1)
            results["histogram_img2"] = self.get_image_histograms(img2)
            if advanced:
                results["hsv_histogram_img1"] = self.get_hsv_histograms(img1)
                results["hsv_histogram_img2"] = self.get_hsv_histograms(img2)
                results["edge_img1"] = self.detect_edges(img1)
                results["edge_img2"] = self.detect_edges(img2)
                edge_comparison_img, edge_similarity = self.compare_edges(img1, img2)
                results["edge_comparison"] = edge_comparison_img
                results["edge_similarity"] = edge_similarity
                hash_similarities = self.calculate_image_hash_similarity(img1, img2)
                results.update(hash_similarities) 
                color_stats1 = self.color_analysis(img1)
                color_stats2 = self.color_analysis(img2)
                results["color_stats_img1"] = {f"{k}_img1": v for k, v in color_stats1.items()}
                results["color_stats_img2"] = {f"{k}_img2": v for k, v in color_stats2.items()}
        except Exception as e:
            print(f"圖像評估過程中發生錯誤: {e}")
            results["error"] = f"評估過程中發生錯誤: {str(e)}"
        results = {k: v for k, v in results.items() if v is not None}
        return results