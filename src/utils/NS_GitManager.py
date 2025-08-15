import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class GitManager:
    """Git 管理工具類別"""
    @staticmethod
    def is_git_available() -> bool:
        """檢查 Git 是否可用"""
        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                ['git', '--version'], 
                capture_output=True, 
                timeout=10,
                startupinfo=startupinfo
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    @staticmethod
    def clone_repository(repo_url: str, local_path: str, depth: int = 1) -> Tuple[bool, str]:
        """
        克隆 Git 儲存庫
        Args:
            repo_url: 儲存庫 URL
            local_path: 本地路徑
            depth: 克隆深度（預設為 1，僅克隆最新提交）
        Returns:
            (成功與否, 訊息)
        """
        try:
            if os.path.exists(local_path):
                return False, f"目標目錄已存在: {local_path}"
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            cmd = ['git', 'clone', '--depth', str(depth), repo_url, local_path]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                return True, "儲存庫克隆成功"
            else:
                error_msg = result.stderr or result.stdout or "未知錯誤"
                return False, f"克隆失敗: {error_msg}"
        except subprocess.TimeoutExpired:
            return False, "克隆超時，請檢查網路連線"
        except Exception as e:
            return False, f"克隆過程發生錯誤: {str(e)}"
    
    @staticmethod
    def pull_repository(local_path: str) -> Tuple[bool, str]:
        """
        更新 Git 儲存庫
        Args:
            local_path: 本地儲存庫路徑
        Returns:
            (成功與否, 訊息)
        """
        try:
            if not GitManager.is_git_repository(local_path):
                return False, "不是有效的 Git 儲存庫"
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                ['git', 'pull'], 
                cwd=local_path,
                capture_output=True, 
                text=True, 
                timeout=300,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                stdout = result.stdout.strip()
                if "Already up to date" in stdout or "Already up-to-date" in stdout:
                    return True, "儲存庫已是最新版本"
                else:
                    return True, "儲存庫更新成功"
            else:
                error_msg = result.stderr or result.stdout or "未知錯誤"
                return False, f"更新失敗: {error_msg}"
        except subprocess.TimeoutExpired:
            return False, "更新超時"
        except Exception as e:
            return False, f"更新過程發生錯誤: {str(e)}"
    
    @staticmethod
    def is_git_repository(path: str) -> bool:
        """
        檢查路徑是否為有效的 Git 儲存庫
        Args:
            path: 要檢查的路徑  
        Returns:
            是否為 Git 儲存庫
        """
        git_dir = os.path.join(path, '.git')
        return os.path.exists(git_dir)
    
    @staticmethod
    def get_repository_info(local_path: str) -> dict:
        """
        獲取 Git 儲存庫資訊
        Args:
            local_path: 本地儲存庫路徑
        Returns:
            儲存庫資訊字典
        """
        info = {
            'remote_url': '',
            'last_commit_date': '',
            'last_commit_hash': '',
            'branch': '',
            'status': 'unknown'
        }
        try:
            if not GitManager.is_git_repository(local_path):
                info['status'] = 'not_git_repo'
                return info
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=local_path,
                capture_output=True,
                text=True,
                timeout=10,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                info['remote_url'] = result.stdout.strip()
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%cd', '--date=short'],
                cwd=local_path,
                capture_output=True,
                text=True,
                timeout=10,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                info['last_commit_date'] = result.stdout.strip()
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=local_path,
                capture_output=True,
                text=True,
                timeout=10,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                info['last_commit_hash'] = result.stdout.strip()[:8]
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=local_path,
                capture_output=True,
                text=True,
                timeout=10,
                startupinfo=startupinfo
            )
            if result.returncode == 0:
                info['branch'] = result.stdout.strip() or 'main'
            info['status'] = 'ok'
        except Exception as e:
            logger.warning(f"獲取 Git 資訊時發生錯誤: {str(e)}")
            info['status'] = 'error'
        return info
    
    @staticmethod
    def extract_repo_name_from_url(repo_url: str) -> str:
        """
        從 Git URL 中提取儲存庫名稱
        Args:
            repo_url: Git 儲存庫 URL
        Returns:
            儲存庫名稱
        """
        try:
            if repo_url.endswith('.git'):
                repo_url = repo_url[:-4]
            return repo_url.split('/')[-1]
        except:
            return "unknown"
    
    @staticmethod
    def validate_git_url(repo_url: str) -> bool:
        """
        驗證 Git URL 格式
        Args:
            repo_url: Git 儲存庫 URL
        Returns:
            URL 是否有效
        """
        try:
            if not repo_url:
                return False
            valid_protocols = ['https://', 'http://', 'git://']
            if not any(repo_url.startswith(protocol) for protocol in valid_protocols):
                return False
            if 'github.com' in repo_url:
                parts = repo_url.split('/')
                if len(parts) < 5:
                    return False
            return True
        except:
            return False
