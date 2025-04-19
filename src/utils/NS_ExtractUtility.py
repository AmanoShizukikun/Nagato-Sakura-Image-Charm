import os
import logging
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Union, Any


HAS_RARFILE = False
HAS_PY7ZR = False

try:
    import rarfile
    HAS_RARFILE = True
except ImportError:
    pass

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ExtractUtility:
    """檔案解壓縮工具類，支援:
    - ZIP (.zip)
    - TAR (.tar)
    - TAR.GZ/TGZ (.tar.gz, .tgz)
    - RAR (.rar, 需安裝 rarfile 套件)
    - 7Z (.7z, 需安裝 py7zr 套件)
    """
    
    @staticmethod
    def extract_file(file_path: str, 
                    extract_dir: Optional[str] = None, 
                    delete_after: bool = False,
                    progress_callback: Optional[Callable[[int, int, str], None]] = None,
                    password: Optional[str] = None,
                    limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮檔案
        Args:
            file_path: 要解壓的檔案路徑
            extract_dir: 解壓目的地目錄，若為None則使用檔案所在目錄
            delete_after: 解壓後是否刪除原檔案
            progress_callback: 進度回調函數 (current, total, filename)
            password: 解壓密碼，若檔案有加密保護
            limit_extensions: 限制僅解壓這些副檔名的文件，None表示解壓所有文件
            
        Returns:
            tuple: (成功與否, 訊息, 解壓出的文件列表)
        """
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}", []
        
        try:
            if extract_dir is None:
                extract_dir = os.path.dirname(os.path.abspath(file_path))
            extract_dir = os.path.abspath(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            if not ExtractUtility._check_disk_space(file_path, extract_dir):
                return False, "目標磁碟空間不足", []
            file_path_lower = file_path.lower()
            extracted_files = []
            if file_path_lower.endswith('.zip'):
                success, msg, extracted_files = ExtractUtility._extract_zip(
                    file_path, extract_dir, progress_callback, password, limit_extensions)
            elif file_path_lower.endswith('.tar'):
                success, msg, extracted_files = ExtractUtility._extract_tar(
                    file_path, extract_dir, progress_callback, limit_extensions)
            elif file_path_lower.endswith('.tar.gz') or file_path_lower.endswith('.tgz'):
                success, msg, extracted_files = ExtractUtility._extract_tar_gz(
                    file_path, extract_dir, progress_callback, limit_extensions)
            elif file_path_lower.endswith('.rar'):
                success, msg, extracted_files = ExtractUtility._extract_rar(
                    file_path, extract_dir, progress_callback, password, limit_extensions)
            elif file_path_lower.endswith('.7z'):
                success, msg, extracted_files = ExtractUtility._extract_7z(
                    file_path, extract_dir, progress_callback, password, limit_extensions)
            else:
                return False, f"不支援的壓縮格式: {os.path.basename(file_path)}", []
            
            if not success:
                return False, msg, []
                
            if delete_after and success:
                try:
                    os.remove(file_path)
                    logger.info(f"已刪除原始壓縮檔: {file_path}")
                except Exception as e:
                    logger.warning(f"刪除原始壓縮檔失敗: {str(e)}")
            return True, f"解壓縮成功，共{len(extracted_files)}個文件", extracted_files
        
        except Exception as e:
            error_msg = f"解壓縮失敗: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, []
    
    @staticmethod
    def _extract_zip(file_path: str, extract_dir: str, 
                     progress_callback: Optional[Callable] = None, 
                     password: Optional[str] = None,
                     limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮ZIP檔案"""
        try:
            extracted_files = []
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if ExtractUtility._is_unsafe_path(file_info.filename, extract_dir):
                        return False, f"檢測到潛在的路徑遍歷攻擊: {file_info.filename}", []
                members_to_extract = ExtractUtility._filter_files(
                    [f.filename for f in zip_ref.infolist()], limit_extensions)
                total_files = len(members_to_extract)
                for i, filename in enumerate(members_to_extract):
                    if progress_callback:
                        progress_callback(i, total_files, filename)
                    pwd = None if password is None else password.encode()
                    zip_ref.extract(filename, extract_dir, pwd=pwd)
                    extracted_files.append(filename)
            logger.info(f"成功解壓縮ZIP檔案: {file_path}")
            return True, "ZIP檔案解壓成功", extracted_files
        
        except zipfile.BadZipFile:
            return False, "ZIP檔案格式錯誤或損壞", []
        except RuntimeError as e:
            if "password required" in str(e).lower() or "wrong password" in str(e).lower():
                return False, "ZIP檔案需要密碼或密碼錯誤", []
            raise
            
    @staticmethod
    def _extract_tar(file_path: str, extract_dir: str, 
                    progress_callback: Optional[Callable] = None,
                    limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮TAR檔案"""
        try:
            extracted_files = []
            with tarfile.open(file_path, 'r') as tar_ref:
                for member in tar_ref.getmembers():
                    if ExtractUtility._is_unsafe_path(member.name, extract_dir) or not member.name:
                        return False, f"檢測到潛在的路徑遍歷攻擊: {member.name}", []
                members_to_extract = ExtractUtility._filter_files_tar(tar_ref.getmembers(), limit_extensions)
                total_files = len(members_to_extract)
                for i, member in enumerate(members_to_extract):
                    if progress_callback:
                        progress_callback(i, total_files, member.name)
                    tar_ref.extract(member, extract_dir)
                    extracted_files.append(member.name)
            logger.info(f"成功解壓縮TAR檔案: {file_path}")
            return True, "TAR檔案解壓成功", extracted_files
        except tarfile.ReadError:
            return False, "TAR檔案格式錯誤或損壞", []
            
    @staticmethod
    def _extract_tar_gz(file_path: str, extract_dir: str, 
                       progress_callback: Optional[Callable] = None,
                       limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮TAR.GZ或TGZ檔案"""
        try:
            extracted_files = []
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                for member in tar_ref.getmembers():
                    if ExtractUtility._is_unsafe_path(member.name, extract_dir) or not member.name:
                        return False, f"檢測到潛在的路徑遍歷攻擊: {member.name}", []
                members_to_extract = ExtractUtility._filter_files_tar(tar_ref.getmembers(), limit_extensions)
                total_files = len(members_to_extract)
                for i, member in enumerate(members_to_extract):
                    if progress_callback:
                        progress_callback(i, total_files, member.name)
                    tar_ref.extract(member, extract_dir)
                    extracted_files.append(member.name)
            logger.info(f"成功解壓縮TAR.GZ檔案: {file_path}")
            return True, "TAR.GZ檔案解壓成功", extracted_files
        except tarfile.ReadError:
            return False, "TAR.GZ檔案格式錯誤或損壞", []
            
    @staticmethod
    def _extract_rar(file_path: str, extract_dir: str, 
                    progress_callback: Optional[Callable] = None,
                    password: Optional[str] = None,
                    limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮RAR檔案"""
        if not HAS_RARFILE:
            return False, "缺少rarfile模組，無法解壓RAR檔案，請安裝: pip install rarfile", []
            
        try:
            extracted_files = []
            with rarfile.RarFile(file_path) as rf:
                if password:
                    rf.setpassword(password)
                for filename in rf.namelist():
                    if ExtractUtility._is_unsafe_path(filename, extract_dir):
                        return False, f"檢測到潛在的路徑遍歷攻擊: {filename}", []
                members_to_extract = ExtractUtility._filter_files(rf.namelist(), limit_extensions)
                total_files = len(members_to_extract)
                for i, filename in enumerate(members_to_extract):
                    if progress_callback:
                        progress_callback(i, total_files, filename)
                    rf.extract(filename, path=extract_dir)
                    extracted_files.append(filename)
            logger.info(f"成功解壓縮RAR檔案: {file_path}")
            return True, "RAR檔案解壓成功", extracted_files
        except rarfile.BadRarFile:
            return False, "RAR檔案格式錯誤或損壞", []
        except rarfile.PasswordRequired:
            return False, "RAR檔案需要密碼", []
        except rarfile.BadRarPassword:
            return False, "RAR檔案密碼錯誤", []
            
    @staticmethod
    def _extract_7z(file_path: str, extract_dir: str, 
                   progress_callback: Optional[Callable] = None,
                   password: Optional[str] = None,
                   limit_extensions: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
        """解壓縮7Z檔案"""
        if not HAS_PY7ZR:
            return False, "缺少py7zr模組，無法解壓7Z檔案，請安裝: pip install py7zr", []
        try:
            extracted_files = []
            with py7zr.SevenZipFile(file_path, mode='r', password=password) as z:
                file_list = z.getnames()
                for filename in file_list:
                    if ExtractUtility._is_unsafe_path(filename, extract_dir):
                        return False, f"檢測到潛在的路徑遍歷攻擊: {filename}", []
                if limit_extensions:
                    fileinfos = {}
                    for f in file_list:
                        if any(f.lower().endswith(ext.lower()) for ext in limit_extensions):
                            fileinfos[f] = z.archiveinfo.files_info[f]
                    if fileinfos:
                        z.extract(path=extract_dir, targets=fileinfos.keys())
                        extracted_files = list(fileinfos.keys())
                    else:
                        return True, "沒有匹配的文件可以解壓", []
                else:
                    if progress_callback:
                        total = len(file_list)
                        for i, filename in enumerate(file_list):
                            progress_callback(i, total, filename)
                    z.extractall(path=extract_dir)
                    extracted_files = file_list
            logger.info(f"成功解壓縮7Z檔案: {file_path}")
            return True, "7Z檔案解壓成功", extracted_files
        except py7zr.exceptions.Bad7zFile:
            return False, "7Z檔案格式錯誤或損壞", []
        except py7zr.exceptions.PasswordRequired:
            return False, "7Z檔案需要密碼", []
        except py7zr.exceptions.WrongPassword:
            return False, "7Z檔案密碼錯誤", []
            
    @staticmethod
    def is_archive_file(file_path: str) -> bool:
        """檢查檔案是否為支援的壓縮格式
        Args:
            file_path: 檔案路徑
            
        Returns:
            bool: 是否為支援的壓縮檔案
        """
        supported_extensions = ['.zip', '.tar', '.tar.gz', '.tgz']
        if HAS_RARFILE:
            supported_extensions.append('.rar')
        if HAS_PY7ZR:
            supported_extensions.append('.7z')
            
        file_path_lower = file_path.lower()
        for ext in supported_extensions:
            if file_path_lower.endswith(ext):
                return True
                
        return False
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """獲取當前支援的壓縮格式列表
        Returns:
            List[str]: 支援的檔案格式列表
        """
        formats = ['.zip', '.tar', '.tar.gz', '.tgz']
        if HAS_RARFILE:
            formats.append('.rar')
        if HAS_PY7ZR:
            formats.append('.7z')
        return formats
        
    @staticmethod
    def get_extracted_files_of_type(extracted_files: List[str], file_types: List[str]) -> List[str]:
        """從解壓的文件列表中篩選特定類型的文件
        Args:
            extracted_files: 解壓出的文件列表
            file_types: 文件類型列表，例如 ['.pth', '.pt']
            
        Returns:
            list: 符合類型的文件列表
        """
        if not extracted_files or not file_types:
            return []
        result = []
        for file_path in extracted_files:
            file_path_lower = file_path.lower()
            for file_type in file_types:
                if file_path_lower.endswith(file_type.lower()):
                    result.append(file_path)
                    break    
        return result
    
    @staticmethod
    def get_archive_info(file_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """獲取壓縮檔案的資訊
        Args:
            file_path: 壓縮檔路徑
            password: 如果需要的話提供密碼
            
        Returns:
            Dict: 包含檔案資訊的字典，包括檔案列表，總大小等
        """
        if not os.path.exists(file_path):
            return {"error": "檔案不存在"}
        result = {
            "file_path": file_path,
            "format": "",
            "size": os.path.getsize(file_path),
            "file_count": 0,
            "files": [],
            "encrypted": False,
            "error": None
        }
        file_path_lower = file_path.lower()
        try:
            if file_path_lower.endswith(".zip"):
                result["format"] = "ZIP"
                with zipfile.ZipFile(file_path) as z:
                    file_list = z.namelist()
                    result["files"] = file_list
                    result["file_count"] = len(file_list)
                    result["encrypted"] = any(info.flag_bits & 0x1 for info in z.infolist())
            elif file_path_lower.endswith(".tar"):
                result["format"] = "TAR"
                with tarfile.open(file_path, 'r') as t:
                    file_list = [m.name for m in t.getmembers()]
                    result["files"] = file_list
                    result["file_count"] = len(file_list)
            elif file_path_lower.endswith((".tar.gz", ".tgz")):
                result["format"] = "TAR.GZ"
                with tarfile.open(file_path, 'r:gz') as t:
                    file_list = [m.name for m in t.getmembers()]
                    result["files"] = file_list
                    result["file_count"] = len(file_list)
            elif file_path_lower.endswith(".rar") and HAS_RARFILE:
                result["format"] = "RAR"
                with rarfile.RarFile(file_path) as r:
                    if password:
                        r.setpassword(password)
                    file_list = r.namelist()
                    result["files"] = file_list
                    result["file_count"] = len(file_list)
                    result["encrypted"] = r.needs_password()
            elif file_path_lower.endswith(".7z") and HAS_PY7ZR:
                result["format"] = "7Z"
                with py7zr.SevenZipFile(file_path, mode='r', password=password) as z:
                    file_list = z.getnames()
                    result["files"] = file_list
                    result["file_count"] = len(file_list)
                    result["encrypted"] = z.needs_password()
            else:
                result["error"] = "不支援的壓縮格式"
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    @staticmethod
    def _is_unsafe_path(file_path: str, extract_dir: str) -> bool:
        """檢查解壓路徑是否安全（防止路徑遍歷攻擊）"""
        extract_dir = os.path.abspath(extract_dir)
        target_path = os.path.abspath(os.path.join(extract_dir, file_path))
        return not target_path.startswith(extract_dir)
    
    @staticmethod
    def _filter_files(file_list: List[str], limit_extensions: Optional[List[str]]) -> List[str]:
        """根據副檔名過濾文件"""
        if not limit_extensions:
            return file_list
        return [f for f in file_list if any(f.lower().endswith(ext.lower()) for ext in limit_extensions)]
    
    @staticmethod
    def _filter_files_tar(members, limit_extensions: Optional[List[str]]):
        """根據副檔名過濾 tarfile.TarInfo 成員"""
        if not limit_extensions:
            return members
        return [m for m in members if any(m.name.lower().endswith(ext.lower()) for ext in limit_extensions)]
    
    @staticmethod
    def _check_disk_space(file_path: str, extract_dir: str) -> bool:
        """檢查磁碟空間是否足夠，簡單估算為壓縮檔大小的4倍"""
        try:
            archive_size = os.path.getsize(file_path)
            estimated_size = archive_size * 4 
            disk_usage = shutil.disk_usage(extract_dir)
            free_space = disk_usage.free
            return free_space > estimated_size
        except Exception as e:
            logger.warning(f"檢查磁碟空間時出錯: {str(e)}")
            return True 