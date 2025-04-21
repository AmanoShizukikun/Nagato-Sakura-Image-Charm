import requests
import os
import time
import threading
from PyQt6.QtCore import QObject, pyqtSignal


class DownloadManager(QObject):
    """下載管理器，用於從URL下載模型文件"""
    progress_signal = pyqtSignal(int, int, float) 
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.threads = []
        self.is_downloading = False
        self.cancel_requested = False
        self.current_file_path = None
    
    def cancel_download(self):
        """中止當前下載並刪除部分下載的文件"""
        if self.is_downloading:
            self.cancel_requested = True
            return True
        return False

    def download_file(self, url, filename=None, retry_count=3, retry_delay=5, num_threads=1):
        """
        從指定URL下載檔案並發出進度信號
        Args:
            url: 下載檔案的URL
            filename: 儲存的檔案名稱，若未指定則從URL取得
            retry_count: 重試次數
            retry_delay: 重試間隔(秒)
            num_threads: 使用的下載線程數
        """
        if filename is None:
            filename = url.split("/")[-1]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.cancel_requested = False 
        self.current_file_path = filename 
        attempts = 0
        while attempts < retry_count:
            try:
                response = requests.head(url, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                if total_size == 0:
                    self.single_thread_download(url, filename)
                    return
                downloaded_size = 0
                progress_lock = threading.Lock()
                start_time = time.time()
                self.is_downloading = True
                self.threads = [] 
                def download_chunk(start, end, thread_index):
                    nonlocal downloaded_size
                    headers = {"Range": f"bytes={start}-{end}"}
                    try:
                        chunk_response = requests.get(url, headers=headers, stream=True, timeout=30)
                        chunk_response.raise_for_status()
                        with open(filename, "r+b") as f:
                            f.seek(start)
                            for chunk in chunk_response.iter_content(chunk_size=8192):
                                if self.cancel_requested:
                                    return
                                if chunk:
                                    f.write(chunk)
                                    with progress_lock:
                                        downloaded_size += len(chunk)
                                        elapsed_time = time.time() - start_time
                                        speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                                        self.progress_signal.emit(downloaded_size, total_size, speed)
                    except Exception as e:
                        print(f"線程 {thread_index} 下載失敗: {str(e)}")
                with open(filename, "wb") as f:
                    f.truncate(total_size)
                chunk_size = total_size // num_threads
                for i in range(num_threads):
                    start = i * chunk_size
                    end = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
                    thread = threading.Thread(target=download_chunk, args=(start, end, i))
                    self.threads.append(thread)
                    thread.start()
                for thread in self.threads:
                    thread.join()
                self.is_downloading = False
                if self.cancel_requested:
                    self._delete_incomplete_file(filename)
                    self.finished_signal.emit(False, "下載已由用戶取消")
                    return
                if downloaded_size == total_size:
                    self.finished_signal.emit(True, f"檔案已成功下載至: {os.path.abspath(filename)}")
                else:
                    self._delete_incomplete_file(filename)
                    self.finished_signal.emit(False, f"下載不完整，已下載 {downloaded_size}/{total_size} 字節，請重試")
                return
            except requests.exceptions.RequestException as e:
                attempts += 1
                if self.cancel_requested:
                    self._delete_incomplete_file(filename)
                    self.finished_signal.emit(False, "下載已由用戶取消")
                    return
                if attempts < retry_count:
                    time.sleep(retry_delay)
                else:
                    self._delete_incomplete_file(filename)
                    self.finished_signal.emit(False, f"下載失敗: {str(e)}")
                    return

    def single_thread_download(self, url, filename):
        """
        單線程下載文件
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            start_time = time.time()
            self.is_downloading = True
            self.current_file_path = filename
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.cancel_requested: 
                        self.is_downloading = False
                        f.close()
                        self._delete_incomplete_file(filename)
                        self.finished_signal.emit(False, "下載已由用戶取消")
                        return
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        elapsed_time = time.time() - start_time
                        speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                        self.progress_signal.emit(downloaded_size, total_size, speed)
            self.is_downloading = False
            self.finished_signal.emit(True, f"檔案已成功下載至: {os.path.abspath(filename)}")
        except Exception as e:
            self.is_downloading = False
            self._delete_incomplete_file(filename)
            self.finished_signal.emit(False, f"下載失敗: {str(e)}")
            
    def _delete_incomplete_file(self, file_path):
        """刪除不完整的下載文件"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"已刪除不完整的下載文件: {file_path}")
                return True
        except Exception as e:
            print(f"刪除文件時出錯: {str(e)}")
        return False