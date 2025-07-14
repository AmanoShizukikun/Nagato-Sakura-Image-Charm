import os
import platform
import logging
import torch
import concurrent.futures
import time

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger(__name__)

def _safe_call_with_timeout(func, timeout=5.0, default=None):
    """
    安全地調用函數，設置超時時間
    Args:
        func: 要調用的函數
        timeout: 超時時間（秒）
        default: 超時或錯誤時返回的預設值
    Returns:
        函數的返回值，或預設值
    """
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning(f"函數 {func.__name__} 執行超時 ({timeout}s)")
        return default
    except Exception as e:
        logger.warning(f"函數 {func.__name__} 執行出錯: {str(e)}")
        return default

class SystemInfo:
    """用於收集和提供系統與設備資訊的工具類"""
    @staticmethod
    def collect_device_info():
        """
        收集系統和設備資訊，包括CPU、GPU、記憶體等
        Returns:
            dict: 包含系統和設備資訊的字典
        """
        system_info = {}
        try:
            system = platform.system()
            architecture = platform.architecture()[0]
            os_info = f"{system} {platform.release()} {architecture}"
            if system == "Windows":
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                        build_number = int(winreg.QueryValueEx(key, "CurrentBuildNumber")[0])
                        product_name = winreg.QueryValueEx(key, "ProductName")[0]
                        windows_version = "11" if build_number >= 22000 else "10"
                        os_info = f"{product_name} ({windows_version} Build {build_number}) {architecture}"
                except Exception as e:
                    logger.warning(f"無法從註冊表獲取 Windows 版本: {str(e)}")
            system_info['os'] = os_info
        except Exception as e:
            logger.error(f"獲取作業系統信息時發生錯誤: {str(e)}")
            system_info['os'] = "未知作業系統"
        try:
            if cpuinfo:
                cpu_info_dict = _safe_call_with_timeout(cpuinfo.get_cpu_info, timeout=3.0)
                if cpu_info_dict:
                    cpu_brand = cpu_info_dict.get('brand_raw', platform.processor() or "未知 CPU")
                    if " CPU " in cpu_brand:
                        cpu_brand_model = cpu_brand.split(" CPU ")[0]
                    elif " with " in cpu_brand:
                        cpu_brand_model = cpu_brand.split(" with ")[0]
                    elif "@" in cpu_brand:
                        cpu_brand_model = cpu_brand.split("@")[0].strip()
                    else:
                        cpu_brand_model = cpu_brand
                    try:
                        import psutil
                        cpu_count = psutil.cpu_count(logical=True)
                        physical_cpu_count = psutil.cpu_count(logical=False)
                    except ImportError:
                        cpu_count = os.cpu_count() or 1
                        physical_cpu_count = cpu_count
                    system_info['cpu_brand'] = cpu_brand
                    system_info['cpu_brand_model'] = cpu_brand_model
                    system_info['cpu_count'] = cpu_count
                    system_info['physical_cpu_count'] = physical_cpu_count
                    system_info['cpu_info'] = f"{cpu_brand} ({physical_cpu_count} 核, {cpu_count} 緒)"
                else:
                    cpu_brand_model = platform.processor() or "未知 CPU"
                    cpu_count = os.cpu_count() or 1
                    system_info['cpu_brand_model'] = cpu_brand_model
                    system_info['cpu_count'] = cpu_count
                    system_info['cpu_info'] = f"{cpu_brand_model} ({cpu_count} 緒)"
            else:
                cpu_brand_model = platform.processor() or "未知 CPU"
                cpu_count = os.cpu_count() or 1
                system_info['cpu_brand_model'] = cpu_brand_model
                system_info['cpu_count'] = cpu_count
                system_info['cpu_info'] = f"{cpu_brand_model} ({cpu_count} 緒)"
        except Exception as e:
            logger.error(f"獲取CPU信息時發生錯誤: {str(e)}")
            system_info['cpu_brand'] = "未知 CPU"
            system_info['cpu_brand_model'] = "未知 CPU"
            system_info['cpu_count'] = os.cpu_count() or 1
            system_info['cpu_info'] = "未知 CPU"
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            system_info['total_memory'] = total_memory
            system_info['available_memory'] = available_memory
            system_info['memory_info'] = f"{total_memory:.2f} GB"
        except Exception as e:
            logger.error(f"獲取記憶體信息時發生錯誤: {str(e)}")
            system_info['memory_info'] = "未知記憶體"
        system_info['has_cuda'] = torch.cuda.is_available()
        system_info['has_mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if system_info['has_cuda']:
            try:
                cuda_count = torch.cuda.device_count()
                system_info['cuda_count'] = cuda_count
                system_info['gpus'] = []
                for i in range(cuda_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    compute_capability = torch.cuda.get_device_capability(i)
                    total_gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    try:
                        if pynvml:
                            def get_nvml_info():
                                pynvml.nvmlInit()
                                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                total_gpu_mem_nv = mem_info.total / (1024**3)
                                used_gpu_mem = mem_info.used / (1024**3)
                                pynvml.nvmlShutdown()
                                return total_gpu_mem_nv, used_gpu_mem
                            nvml_result = _safe_call_with_timeout(get_nvml_info, timeout=2.0)
                            if nvml_result:
                                total_gpu_mem_nv, used_gpu_mem = nvml_result
                                total_gpu_mem = total_gpu_mem_nv
                            else:
                                used_gpu_mem = allocated
                        else:
                            used_gpu_mem = allocated
                    except Exception as e:
                        logger.debug(f"使用NVML獲取GPU {i} 顯存信息失敗: {str(e)}")
                        used_gpu_mem = allocated
                    gpu_info = {
                        'index': i,
                        'name': gpu_name,
                        'compute_capability': f"{compute_capability[0]}.{compute_capability[1]}",
                        'total_memory': total_gpu_mem,
                        'allocated_memory': allocated,
                        'used_memory': used_gpu_mem,
                        'display_name': f"{gpu_name} (CC {compute_capability[0]}.{compute_capability[1]})",
                        'memory_info': f"{total_gpu_mem:.2f} GB",
                        'device_str': f"cuda:{i}" if cuda_count > 1 else "cuda",
                        'combo_text': f"{gpu_name} (CUDA:{i})" if cuda_count > 1 else f"{gpu_name} (CUDA)"
                    }
                    system_info['gpus'].append(gpu_info)
                if system_info['gpus']:
                    system_info['primary_gpu'] = system_info['gpus'][0]
                    system_info['gpu_info'] = system_info['primary_gpu']['display_name']
                    system_info['gpu_memory_info'] = system_info['primary_gpu']['memory_info']
                    system_info['total_gpu_memory'] = system_info['primary_gpu']['total_memory']
                    system_info['is_low_memory_gpu'] = system_info['total_gpu_memory'] < 3 
                else:
                    system_info['gpu_info'] = "未找到GPU信息"
                    system_info['gpu_memory_info'] = "未知"
                    system_info['is_low_memory_gpu'] = False
            except Exception as e:
                logger.error(f"獲取GPU信息時發生錯誤: {str(e)}")
                system_info['gpu_info'] = "獲取GPU信息出錯"
                system_info['gpu_memory_info'] = "未知"
                system_info['is_low_memory_gpu'] = False
        else:
            system_info['gpu_info'] = "未檢測到支援的顯示卡"
            system_info['gpu_memory_info'] = "不適用"
            system_info['is_low_memory_gpu'] = False
        if system_info['has_mps']:
            try:
                if platform.system() == "Darwin":
                    try:
                        def get_mac_hardware_info():
                            import subprocess
                            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True, timeout=8)
                            if result.returncode == 0:
                                lines = result.stdout.split('\n')
                                chip_info = None
                                memory_info = None
                                for line in lines:
                                    if 'Chip:' in line:
                                        chip_info = line.split(':', 1)[1].strip()
                                    elif 'Memory:' in line:
                                        memory_info = line.split(':', 1)[1].strip()
                                return chip_info, memory_info
                            return None, None
                        mac_info = _safe_call_with_timeout(get_mac_hardware_info, timeout=10.0)
                        if mac_info:
                            chip_info, memory_info = mac_info
                            system_info['mps_device_name'] = chip_info or "Apple Silicon"
                            system_info['mps_memory_info'] = memory_info or "未知"
                        else:
                            system_info['mps_device_name'] = "Apple Silicon"
                            system_info['mps_memory_info'] = "未知"
                    except Exception as e:
                        logger.debug(f"無法獲取詳細的 Mac 硬體資訊: {str(e)}")
                        system_info['mps_device_name'] = "Apple Silicon"
                        system_info['mps_memory_info'] = "未知"
                else:
                    system_info['mps_device_name'] = "Metal GPU"
                    system_info['mps_memory_info'] = "未知"
                system_info['mps_info'] = f"{system_info['mps_device_name']} (MPS)"
                logger.info(f"檢測到 MPS 支援: {system_info['mps_info']}")
            except Exception as e:
                logger.error(f"獲取 MPS 設備資訊時發生錯誤: {str(e)}")
                system_info['mps_device_name'] = "Metal GPU"
                system_info['mps_memory_info'] = "未知"
                system_info['mps_info'] = "Metal GPU (MPS)"
        else:
            system_info['mps_info'] = None
        system_info['pytorch_version'] = torch.__version__
        return system_info
    
    @staticmethod
    def get_device_info_for_combobox(system_info):
        """
        為UI的下拉框準備設備選項
        Args:
            system_info (dict): 包含系統資訊的字典
        Returns:
            list: 包含設備選項的列表，每個選項是一個(display_text, device_value)的元組
        """
        device_options = []
        device_options.append(("自動選擇", "auto"))
        cpu_name = system_info.get('cpu_brand_model', '未知 CPU')
        device_options.append((f"{cpu_name} (CPU)", "cpu"))
        if system_info.get('has_mps', False):
            mps_name = system_info.get('mps_device_name', 'Metal GPU')
            device_options.append((f"{mps_name} (MPS)", "mps"))
        if system_info.get('has_cuda', False):
            for gpu in system_info.get('gpus', []):
                device_options.append((gpu['combo_text'], gpu['device_str']))
        return device_options
    
    @staticmethod
    def get_device_display_name(device, system_info):
        """
        獲取設備的友好顯示名稱
        Args:
            device (torch.device): PyTorch設備對象
            system_info (dict): 包含系統資訊的字典
        Returns:
            str: 設備的友好顯示名稱
        """
        device_type = device.type
        if device_type == "cpu":
            return f"{system_info.get('cpu_brand_model', '未知 CPU')} (CPU)"
        elif device_type == "mps":
            mps_name = system_info.get('mps_device_name', 'Metal GPU')
            return f"{mps_name} (MPS)"
        elif device_type == "cuda":
            device_index = device.index if hasattr(device, 'index') else 0
            gpu_info = None
            for gpu in system_info.get('gpus', []):
                if gpu['index'] == device_index:
                    gpu_info = gpu
                    break
            if gpu_info:
                gpu_name = gpu_info['name']
                gpu_mem = gpu_info['total_memory']
                return f"{gpu_name} {gpu_mem:.1f}GB (CUDA:{device_index})"
            else:
                gpu_name = torch.cuda.get_device_name(device_index)
                return f"{gpu_name} (CUDA:{device_index})"
        return str(device)

def get_system_info():
    """獲取系統和設備資訊"""
    return SystemInfo.collect_device_info()

def get_device_options(system_info=None):
    """獲取UI下拉框的設備選項"""
    if system_info is None:
        system_info = get_system_info()
    return SystemInfo.get_device_info_for_combobox(system_info)

def get_device_name(device, system_info=None):
    """獲取設備的友好顯示名稱"""
    if system_info is None:
        system_info = get_system_info()
    return SystemInfo.get_device_display_name(device, system_info)