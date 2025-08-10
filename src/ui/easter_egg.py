import sys
import math
import random
import logging
import os
import threading
import subprocess
import time
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Optional, Dict
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QApplication, QDialog)
from PyQt6.QtCore import QTimer, Qt, QPointF, QRectF, pyqtSignal, QUrl
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QFont, 
                         QPixmap, QPaintEvent, QMouseEvent, QKeyEvent, QIcon)


# 遊戲常數
GAME_WIDTH = 800  # 遊戲區域寬度
WINDOW_WIDTH = 1400  # 窗口總寬度
WINDOW_HEIGHT = 1000  # 窗口高度
PLAYER_WIDTH = 100  # 玩家板子寬度
PLAYER_HEIGHT = 100  # 玩家板子高度
PLAYER_Y = WINDOW_HEIGHT - 50  # 玩家固定Y位置
BALL_RADIUS = 12  # 彈跳球半徑
BALL_SPEED = 8  # 彈跳球初始速度
BASE_INVINCIBLE_TIME = 120  # 基礎無敵時間(幀數)
MIN_INVINCIBLE_TIME = 30  # 最低無敵時間(幀數)
BOSS_SCORE_THRESHOLD = 5000  # 召喚王的分數閾值
BOSS2_SCORE_THRESHOLD = 15000  # 召喚超級王的分數閾值


class Vector2D:
    """二維向量類"""
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def copy(self):
        return Vector2D(self.x, self.y)
    
    def mag(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        m = self.mag()
        if m > 0:
            return Vector2D(self.x / m, self.y / m)
        return Vector2D(0, 0)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    @staticmethod
    def from_angle(angle):
        return Vector2D(math.cos(angle), math.sin(angle))
    
    @staticmethod
    def random2D():
        angle = random.uniform(0, 2 * math.pi)
        return Vector2D.from_angle(angle)


class EnemyType(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    BOSS = "boss"
    BOSS_2 = "boss_2"


class BulletType(Enum):
    NORMAL = "normal"
    LASER = "laser"
    WAVE = "wave"
    TRAP = "trap"


class Player:
    """玩家類"""
    def __init__(self):
        self.position = Vector2D(GAME_WIDTH / 2, PLAYER_Y)
        self.health = 10
        self.is_invincible = False
        self.invincible_counter = 0
    
    def update(self, mouse_x: int, game_level: int):
        self.position.x = max(PLAYER_WIDTH // 2, min(mouse_x, GAME_WIDTH - PLAYER_WIDTH // 2))
        if self.is_invincible:
            self.invincible_counter += 1
            current_invincible_time = max(MIN_INVINCIBLE_TIME, BASE_INVINCIBLE_TIME - game_level)
            if self.invincible_counter >= current_invincible_time:
                self.is_invincible = False
                self.invincible_counter = 0
    
    def take_damage(self):
        if not self.is_invincible:
            self.health -= 1
            self.is_invincible = True
            self.invincible_counter = 0
            return self.health <= 0
        return False


class Ball:
    """彈跳球類"""
    def __init__(self, position: Vector2D, velocity: Vector2D):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.rotation = 0
        self.rotation_speed = random.uniform(0.1, 0.15) * (1 if random.random() > 0.5 else -1)
    
    def update(self):
        self.position = self.position + self.velocity
        self.rotation += self.rotation_speed
        if self.position.x < BALL_RADIUS or self.position.x > GAME_WIDTH - BALL_RADIUS:
            self.velocity.x *= -1
            self.position.x = max(BALL_RADIUS, min(self.position.x, GAME_WIDTH - BALL_RADIUS))
            self.rotation_speed *= -1
        if self.position.y < BALL_RADIUS:
            self.velocity.y *= -1
            self.position.y = BALL_RADIUS
            self.rotation_speed *= -1
        current_speed = self.velocity.mag()
        if abs(current_speed - BALL_SPEED) > 0.1:
            self.velocity = self.velocity.normalize() * BALL_SPEED
    
    def check_player_collision(self, player: Player):
        if (self.position.y + BALL_RADIUS >= player.position.y - PLAYER_HEIGHT // 2 and
            self.position.y - BALL_RADIUS <= player.position.y + PLAYER_HEIGHT // 2 and
            self.position.x + BALL_RADIUS >= player.position.x - PLAYER_WIDTH // 2 and
            self.position.x - BALL_RADIUS <= player.position.x + PLAYER_WIDTH // 2 and
            self.velocity.y > 0):
            return True
        return False
    
    def check_enemy_collision(self, enemy):
        distance = self.position.dist(enemy.position)
        if distance < BALL_RADIUS + enemy.size / 2:
            normal = (self.position - enemy.position).normalize()
            dot_product = self.velocity.dot(normal)
            if dot_product < 0:
                reflection = self.velocity - normal * (2 * dot_product)
                self.velocity = reflection.normalize() * BALL_SPEED
                self.position = self.position + normal * 2.0
                return True
        return False


class Enemy:
    """敵人類"""
    def __init__(self, position: Vector2D, enemy_type: EnemyType):
        self.position = position.copy()
        self.type = enemy_type
        self.attack_mode = 0
        self.attack_timer = 0
        if enemy_type == EnemyType.SMALL:
            self.size = 85
            self.health = 1
            self.velocity = Vector2D.random2D() * 3
            self.color = QColor(100, 255, 100)
        elif enemy_type == EnemyType.MEDIUM:
            self.size = 115
            self.health = 3
            self.velocity = Vector2D.random2D() * 2
            self.color = QColor(255, 255, 100)
        elif enemy_type == EnemyType.LARGE:
            self.size = 145
            self.health = 9
            self.velocity = Vector2D.random2D() * 1
            self.color = QColor(255, 100, 100)
        elif enemy_type == EnemyType.BOSS:
            self.size = 195
            self.health = 50
            self.velocity = Vector2D.random2D() * 1.5
            self.color = QColor(255, 50, 255)
        elif enemy_type == EnemyType.BOSS_2:
            self.size = 245
            self.health = 100
            self.velocity = Vector2D.random2D() * 2.0
            self.color = QColor(200, 0, 200)
            self.attack_mode = random.randint(0, 3)
    
    def update(self, frame_counter: int):
        self.position = self.position + self.velocity
        if (self.position.x < self.size / 2 or 
            self.position.x > GAME_WIDTH - self.size / 2):
            self.velocity.x *= -1
            self.position.x = max(self.size / 2, min(self.position.x, GAME_WIDTH - self.size / 2))
        if (self.position.y < self.size / 2 or 
            self.position.y > WINDOW_HEIGHT / 2):
            self.velocity.y *= -1
            self.position.y = max(self.size / 2, min(self.position.y, WINDOW_HEIGHT / 2))
        if self.type == EnemyType.BOSS_2:
            self.attack_timer += 1
            if self.attack_timer > 300:
                self.attack_mode = (self.attack_mode + 1) % 4
                self.attack_timer = 0
    
    def check_collision(self, other):
        distance = self.position.dist(other.position)
        min_dist = self.size / 2 + other.size / 2
        if distance < min_dist:
            direction = (self.position - other.position).normalize()
            overlap = min_dist - distance
            self.position = self.position + direction * (overlap * 0.5)
            other.position = other.position - direction * (overlap * 0.5)
            relative_velocity = self.velocity - other.velocity
            dot_product = relative_velocity.dot(direction)
            if dot_product > 0:
                return
            restitution = 0.8
            impulse = -(1 + restitution) * dot_product / 2
            self.velocity = self.velocity + direction * impulse
            other.velocity = other.velocity - direction * impulse
    
    def get_shoot_chance(self):
        chances = {
            EnemyType.SMALL: 0.005,
            EnemyType.MEDIUM: 0.01,
            EnemyType.LARGE: 0.015,
            EnemyType.BOSS: 0.05,
            EnemyType.BOSS_2: 0.08
        }
        return chances.get(self.type, 0.01)
    
    def get_score(self):
        scores = {
            EnemyType.SMALL: 150,
            EnemyType.MEDIUM: 250,
            EnemyType.LARGE: 500,
            EnemyType.BOSS: 1000,
            EnemyType.BOSS_2: 2000
        }
        return scores.get(self.type, 100)
    
    def get_drop_chance(self):
        chances = {
            EnemyType.SMALL: 0.1,
            EnemyType.MEDIUM: 0.2,
            EnemyType.LARGE: 0.3,
            EnemyType.BOSS: 1.0,
            EnemyType.BOSS_2: 1.0
        }
        return chances.get(self.type, 0.3)


class Bullet:
    """子彈類"""
    def __init__(self, position: Vector2D, velocity: Vector2D, 
                 color: QColor, bullet_type: BulletType):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.color = color
        self.type = bullet_type
        self.lifespan = 90
        self.activated = False
        if bullet_type == BulletType.LASER:
            self.angle = math.atan2(velocity.y, velocity.x)
        else:
            self.angle = 0
            
        if bullet_type == BulletType.NORMAL:
            self.size = 25  
        elif bullet_type == BulletType.LASER:
            self.size = 15  
            self.lifespan = 60
        elif bullet_type == BulletType.WAVE:
            self.size = 25  
        elif bullet_type == BulletType.TRAP:
            self.size = 25 
            self.lifespan = 180
    
    def update(self, frame_counter: int):
        if self.type == BulletType.NORMAL:
            self.position = self.position + self.velocity
        elif self.type == BulletType.LASER:
            self.position = self.position + self.velocity
        elif self.type == BulletType.WAVE:
            self.position = self.position + self.velocity
            self.position.x += math.sin(frame_counter * 0.2) * 1.5
        elif self.type == BulletType.TRAP:
            if not self.activated:
                self.lifespan -= 1
                if self.lifespan < 120:
                    self.activated = True
                    self.velocity = self.velocity * 25
            else:
                self.position = self.position + self.velocity
    
    def check_player_collision(self, player: Player):
        distance = self.position.dist(player.position)
        if self.type == BulletType.LASER:
            return distance < (self.size / 2 + 6)
        else:
            return distance < (self.size / 2 + 5)


class NagatoSakuraBounceGame(QWidget):
    """長門櫻彈跳球遊戲主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_resources()
        self.init_game()
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.game_loop)
        self.game_timer.start(16)
        self.frame_counter = 0
    
    def init_ui(self):
        self.setWindowTitle("長門櫻 彈跳球")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        try:
            current_directory = Path(__file__).resolve().parent.parent.parent
            icon_path = current_directory / "assets" / "icon" / "1.3.0.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
            else:
                icon_dir = current_directory / "assets" / "icon"
                if icon_dir.exists():
                    icon_files = list(icon_dir.glob("*.ico"))
                    if icon_files:
                        latest_icon = sorted(icon_files)[-1]
                        self.setWindowIcon(QIcon(str(latest_icon)))
                        logging.info(f"彩蛋遊戲使用圖示: {latest_icon.name}")
        except Exception as e:
            logging.warning(f"無法載入遊戲圖示: {e}")
        self.font = QFont("Microsoft YaHei", 12)
        self.title_font = QFont("Microsoft YaHei", 24, QFont.Weight.Bold)
        self.game_over_font = QFont("Microsoft YaHei", 36, QFont.Weight.Bold)
    
    def load_resources(self):
        """載入遊戲資源（圖片和音樂）"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            assets_path = os.path.join(project_root, "extensions", "Nagato-Sakura-Bounce-py", "data")
            self.images: Dict[str, QPixmap] = {}
            image_files = {
                'player': 'player.png',
                'ball': 'ball.png',
                'small_enemy': 'small_enemy.png',
                'medium_enemy': 'medium_enemy.png',
                'large_enemy': 'large_enemy.png',
                'boss_enemy': 'boss_enemy.png',
                'boss2_enemy': 'boss2_enemy.png',
                'background': 'background.png',
                'scoreboard': 'scoreboard.png',
                'bullet': 'bullet.png',
                'laser': 'laser.png',
                'wave': 'wave.png'
            }
            for key, filename in image_files.items():
                image_path = os.path.join(assets_path, filename)
                if os.path.exists(image_path):
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.images[key] = pixmap
                        logging.info(f"成功載入圖片: {filename}")
                    else:
                        logging.warning(f"無法載入圖片: {filename}")
                        self.images[key] = self.create_default_image(key)
                else:
                    logging.warning(f"圖片檔案不存在: {image_path}")
                    self.images[key] = self.create_default_image(key)
            self.setup_background_music(assets_path)
        except Exception as e:
            logging.error(f"載入資源時發生錯誤: {e}")
            for key in image_files.keys():
                self.images[key] = self.create_default_image(key)
    
    def create_default_image(self, image_type: str) -> QPixmap:
        """創建預設圖片"""
        size_map = {
            'player': (PLAYER_WIDTH, PLAYER_HEIGHT),
            'ball': (int(BALL_RADIUS * 2), int(BALL_RADIUS * 2)),
            'small_enemy': (90, 90),
            'medium_enemy': (120, 120),
            'large_enemy': (150, 150),
            'boss_enemy': (200, 200),
            'boss2_enemy': (250, 250),
            'background': (GAME_WIDTH, WINDOW_HEIGHT),
            'scoreboard': (WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT),
            'bullet': (20, 20),
            'laser': (30, 80),
            'wave': (30, 30)
        }
        color_map = {
            'player': QColor(200, 200, 255),
            'ball': QColor(255, 255, 255),
            'small_enemy': QColor(100, 255, 100),
            'medium_enemy': QColor(255, 255, 100),
            'large_enemy': QColor(255, 100, 100),
            'boss_enemy': QColor(255, 50, 255),
            'boss2_enemy': QColor(200, 0, 200),
            'background': QColor(0, 0, 30),
            'scoreboard': QColor(30, 0, 30),
            'bullet': QColor(255, 100, 100),
            'laser': QColor(255, 0, 0),
            'wave': QColor(0, 200, 255)
        }
        width, height = size_map.get(image_type, (50, 50))
        color = color_map.get(image_type, QColor(255, 0, 255))
        pixmap = QPixmap(width, height)
        pixmap.fill(color)
        return pixmap
    
    def setup_background_music(self, assets_path: str):
        """設置背景音樂（使用 ffmpeg）"""
        try:
            bgm_path = os.path.join(assets_path, "bgm.mp3")
            if os.path.exists(bgm_path):
                self.bgm_path = bgm_path
                self.music_process = None
                self.music_thread = None
                self.is_music_playing = False
                self.should_stop_music = False
                self.is_music_paused = False
                self.start_music_thread()
                logging.info(f"ffmpeg 背景音樂設置完成: {bgm_path}")
            else:
                logging.warning(f"背景音樂檔案不存在: {bgm_path}")
                self.bgm_path = None
        except Exception as e:
            logging.error(f"設置背景音樂時發生錯誤: {e}")
            self.bgm_path = None
    
    def start_music_thread(self):
        """啟動音樂播放線程"""
        if hasattr(self, 'bgm_path') and self.bgm_path:
            self.should_stop_music = False
            self.music_thread = threading.Thread(target=self._play_music_loop, daemon=True)
            self.music_thread.start()
            logging.info("音樂播放線程已啟動")
    
    def _play_music_loop(self):
        """音樂播放循環（在背景線程中執行）"""
        while not self.should_stop_music:
            try:
                if os.name == 'nt':  # Windows
                    cmd = [
                        'ffplay', 
                        '-v', 'quiet',
                        '-nodisp',  # 不顯示視窗
                        '-loop', '0',  # 無限循環
                        '-volume', '40',  # 音量 40%
                        self.bgm_path
                    ]
                else:  # Linux/Mac
                    cmd = [
                        'ffplay', 
                        '-v', 'quiet',
                        '-nodisp',
                        '-loop', '0',
                        '-volume', '40',
                        self.bgm_path
                    ]
                self.music_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                self.is_music_playing = True
                logging.info("ffmpeg 背景音樂開始播放")
                while self.music_process and self.music_process.poll() is None and not self.should_stop_music:
                    time.sleep(0.1)
                if not self.should_stop_music:
                    logging.info("音樂播放結束，準備重新開始...")
                    time.sleep(0.1)
            except FileNotFoundError:
                logging.error("ffplay 命令未找到，請確保 FFmpeg 已安裝並在 PATH 中")
                break
            except Exception as e:
                logging.error(f"播放音樂時發生錯誤: {e}")
                if not self.should_stop_music:
                    time.sleep(1)
            finally:
                self.is_music_playing = False
    
    def start_music(self):
        """延遲啟動音樂播放（ffmpeg 版本）"""
        try:
            if hasattr(self, 'bgm_path') and self.bgm_path and not self.is_music_playing:
                self.start_music_thread()
        except Exception as e:
            logging.warning(f"啟動音樂時發生錯誤: {e}")
    
    def restart_music(self):
        """重新啟動音樂播放（ffmpeg 版本）"""
        try:
            self.stop_music()
            time.sleep(0.1)
            self.start_music_thread()
        except Exception as e:
            logging.warning(f"重新啟動音樂時發生錯誤: {e}")
    
    def stop_music(self):
        """停止音樂播放（ffmpeg 版本）"""
        try:
            self.should_stop_music = True
            if hasattr(self, 'music_process') and self.music_process:
                try:
                    self.music_process.terminate()
                    self.music_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.music_process.kill()
                except Exception:
                    pass
                finally:
                    self.music_process = None
            if hasattr(self, 'music_thread') and self.music_thread and self.music_thread.is_alive():
                self.music_thread.join(timeout=1)
            self.is_music_playing = False
            logging.info("ffmpeg 背景音樂已停止")
        except Exception as e:
            logging.warning(f"停止音樂時發生錯誤: {e}")
    
    def pause_music(self):
        """暫停音樂播放"""
        try:
            if self.is_music_playing and not self.is_music_paused:
                self.should_stop_music = True
                if hasattr(self, 'music_process') and self.music_process:
                    try:
                        self.music_process.terminate()
                        self.music_process.wait(timeout=1)
                    except:
                        pass
                    finally:
                        self.music_process = None
                self.is_music_paused = True
                self.is_music_playing = False
                logging.info("音樂已暫停（進程已停止）")
        except Exception as e:
            logging.warning(f"暫停音樂時發生錯誤: {e}")
    
    def resume_music(self):
        """恢復音樂播放"""
        try:
            if self.is_music_paused:
                self.is_music_paused = False
                self.start_music_thread()
                logging.info("音樂已恢復（重新啟動進程）")
        except Exception as e:
            logging.warning(f"恢復音樂時發生錯誤: {e}")
    
    def init_game(self):
        self.game_over = False
        self.is_paused = False
        self.score = 0
        self.boss_score_counter = 0
        self.boss2_score_counter = 0
        self.spawn_timer = 0
        self.game_level = 1
        if hasattr(self, 'is_music_paused'):
            self.is_music_paused = False
        self.player = Player()
        self.balls: List[Ball] = []
        self.enemies: List[Enemy] = []
        self.bullets: List[Bullet] = []
        self.add_ball()
        for _ in range(3):
            self.spawn_random_enemy()
        self.mouse_x = GAME_WIDTH // 2
    
    def game_loop(self):
        """主遊戲循環"""
        if not self.game_over and not self.is_paused:
            self.update_game()
        self.frame_counter += 1
        if self.frame_counter > 600:
            self.frame_counter = 0
        self.update()
    
    def update_game(self):
        """更新遊戲邏輯"""
        self.update_game_level()
        self.player.update(self.mouse_x, self.game_level)
        if not self.balls:
            self.player.health -= 1
            if self.player.health <= 0:
                self.game_over = True
            else:
                self.add_ball()
        for i in range(len(self.balls) - 1, -1, -1):
            ball = self.balls[i]
            ball.update()
            if ball.check_player_collision(self.player):
                relative_x = (ball.position.x - self.player.position.x) / (PLAYER_WIDTH / 2)
                relative_x = max(-0.8, min(relative_x, 0.8))
                angle = -math.pi / 4 + (relative_x + 0.8) * (math.pi / 2) / 1.6
                new_dir = Vector2D.from_angle(-math.pi / 2 + angle)
                ball.velocity = new_dir * BALL_SPEED
            for j in range(len(self.enemies) - 1, -1, -1):
                enemy = self.enemies[j]
                if ball.check_enemy_collision(enemy):
                    enemy.health -= 1
                    if enemy.health <= 0:
                        self.score += enemy.get_score()
                        self.boss_score_counter += enemy.get_score()
                        self.boss2_score_counter += enemy.get_score()
                        drop_chance = enemy.get_drop_chance()
                        if (enemy.type in [EnemyType.BOSS, EnemyType.BOSS_2] or random.random() < drop_chance):
                            balls_to_add = 0
                            health_to_add = 0
                            if enemy.type == EnemyType.BOSS:
                                balls_to_add = 3
                                health_to_add = 3
                            elif enemy.type == EnemyType.BOSS_2:
                                balls_to_add = 5
                                health_to_add = 5
                            else:
                                balls_to_add = random.randint(0, 3)
                                health_to_add = random.randint(0, 3)
                            for _ in range(balls_to_add):
                                self.add_ball()
                            self.player.health += health_to_add
                        
                        self.enemies.remove(enemy)
                    break
            if ball.position.y > WINDOW_HEIGHT:
                self.balls.remove(ball)
        for i, enemy in enumerate(self.enemies):
            enemy.update(self.frame_counter)
            for j in range(i + 1, len(self.enemies)):
                enemy.check_collision(self.enemies[j])
            if random.random() < enemy.get_shoot_chance():
                self.enemy_shoot(enemy)
        for i in range(len(self.bullets) - 1, -1, -1):
            bullet = self.bullets[i]
            bullet.update(self.frame_counter)
            if not self.player.is_invincible and bullet.check_player_collision(self.player):
                if self.player.take_damage():
                    self.game_over = True
                self.bullets.remove(bullet)
                continue
            if (bullet.position.y > WINDOW_HEIGHT or bullet.position.y < 0 or bullet.position.x < 0 or bullet.position.x > GAME_WIDTH):
                self.bullets.remove(bullet)
        self.spawn_timer += 1
        spawn_interval = self.get_spawn_interval()
        if (self.spawn_timer > spawn_interval and 
            len(self.enemies) < 10 + self.game_level):
            self.spawn_random_enemy()
            self.spawn_timer = 0
        if self.boss_score_counter >= BOSS_SCORE_THRESHOLD:
            self.spawn_boss()
            self.boss_score_counter = 0
        if self.boss2_score_counter >= BOSS2_SCORE_THRESHOLD:
            self.spawn_boss2()
            self.boss2_score_counter = 0
    
    def update_game_level(self):
        """更新遊戲等級"""
        self.game_level = min(self.score // 500 + 1, 100)
    
    def get_spawn_interval(self):
        """獲取生怪間隔"""
        return max(50, 150 - self.game_level)
    
    def add_ball(self):
        """添加新球"""
        position = Vector2D(self.player.position.x, self.player.position.y - PLAYER_HEIGHT // 2 - BALL_RADIUS)
        velocity = Vector2D.from_angle(-math.pi / 2) * BALL_SPEED
        self.balls.append(Ball(position, velocity))
    
    def spawn_random_enemy(self):
        """隨機生成敵人"""
        rand = random.random()
        small_prob = max(0.1, 1 - (self.game_level * 0.01))
        medium_prob = max(0, self.game_level * 0.007)
        large_prob = max(0, self.game_level * 0.003)
        total = small_prob + medium_prob + large_prob
        small_prob /= total
        medium_prob /= total
        if rand < small_prob:
            self.spawn_enemy(EnemyType.SMALL)
        elif rand < small_prob + medium_prob:
            self.spawn_enemy(EnemyType.MEDIUM)
        else:
            self.spawn_enemy(EnemyType.LARGE)
    
    def spawn_enemy(self, enemy_type: EnemyType):
        """生成指定類型的敵人"""
        attempts = 0
        while attempts < 50:
            x = random.uniform(100, GAME_WIDTH - 100)
            y = random.uniform(100, WINDOW_HEIGHT / 2)
            valid_position = True
            size_map = {
                EnemyType.SMALL: 30,
                EnemyType.MEDIUM: 50, 
                EnemyType.LARGE: 80,
                EnemyType.BOSS: 120,
                EnemyType.BOSS_2: 160
            }
            size = size_map[enemy_type]
            for enemy in self.enemies:
                min_dist = size / 2 + enemy.size / 2 + 10
                if Vector2D(x, y).dist(enemy.position) < min_dist:
                    valid_position = False
                    break
            if valid_position:
                self.enemies.append(Enemy(Vector2D(x, y), enemy_type))
                break
            attempts += 1
    
    def spawn_boss(self):
        """生成Boss"""
        position = Vector2D(GAME_WIDTH / 2, 150)
        self.enemies.append(Enemy(position, EnemyType.BOSS))
    
    def spawn_boss2(self):
        """生成超級Boss"""
        position = Vector2D(GAME_WIDTH / 2, 200)
        self.enemies.append(Enemy(position, EnemyType.BOSS_2))
    
    def enemy_shoot(self, enemy: Enemy):
        """敵人射擊邏輯"""
        if enemy.type == EnemyType.SMALL:
            direction = Vector2D(0, 1)
            self.bullets.append(Bullet(enemy.position.copy(), direction * 3, QColor(0, 255, 0), BulletType.NORMAL))
        elif enemy.type == EnemyType.MEDIUM:
            # 三發散射
            for i in range(-1, 2):
                direction = Vector2D.from_angle(math.pi / 2 + i * 0.2)
                self.bullets.append(Bullet(enemy.position.copy(), direction * 3, QColor(255, 255, 0), BulletType.NORMAL))
        elif enemy.type == EnemyType.LARGE:
            # 圓形散射
            for i in range(9):
                direction = Vector2D.from_angle(math.pi / 2 + i * 2 * math.pi / 9)
                self.bullets.append(Bullet(enemy.position.copy(), direction * 2.5, QColor(255, 100, 0), BulletType.NORMAL))
        elif enemy.type == EnemyType.BOSS:
            # 追蹤彈
            to_player = (self.player.position - enemy.position).normalize() * 4
            self.bullets.append(Bullet(enemy.position.copy(), to_player, QColor(255, 0, 255), BulletType.NORMAL))
            # 圓形散射
            for i in range(9):
                direction = Vector2D.from_angle(self.frame_counter * 0.02 + i * 2 * math.pi / 9)
                self.bullets.append(Bullet(enemy.position.copy(), direction * 3, QColor(255, 0, 100), BulletType.NORMAL))
        elif enemy.type == EnemyType.BOSS_2:
            if enemy.attack_mode == 0:
                # 三重追蹤彈 + 螺旋彈幕
                to_player = (self.player.position - enemy.position).normalize() * 5
                # 主追蹤彈
                self.bullets.append(Bullet(enemy.position.copy(), to_player.copy(), QColor(200, 0, 200), BulletType.NORMAL))
                # 側翼追蹤彈
                offset1 = Vector2D(to_player.y, -to_player.x) * 0.3
                offset2 = Vector2D(-to_player.y, to_player.x) * 0.3
                self.bullets.append(Bullet(enemy.position.copy(), (to_player + offset1) * 0.8, QColor(200, 0, 200), BulletType.NORMAL))
                self.bullets.append(Bullet(enemy.position.copy(), (to_player + offset2) * 0.8, QColor(200, 0, 200), BulletType.NORMAL))
                # 螺旋彈幕
                for i in range(9):
                    angle = self.frame_counter * 0.03 + i * 2 * math.pi / 9
                    direction = Vector2D.from_angle(angle)
                    self.bullets.append(Bullet(enemy.position.copy(), direction * 3, QColor(230, 100, 230), BulletType.NORMAL))
            elif enemy.attack_mode == 1:
                # 雷射攻擊 - 玩家方向雷射
                laser_dir = (self.player.position - enemy.position).normalize() * 6
                self.bullets.append(Bullet(enemy.position.copy(), laser_dir, QColor(255, 0, 0), BulletType.LASER))
                # 固定方向的雷射
                laser_count = 3
                for i in range(laser_count):
                    angle = self.frame_counter * 0.01 + i * 2 * math.pi / laser_count
                    direction = Vector2D.from_angle(angle)
                    self.bullets.append(Bullet(enemy.position.copy(), direction * 5, QColor(255, 100, 100), BulletType.LASER))
            elif enemy.attack_mode == 2:
                # 波形彈幕 + 爆炸彈
                for i in range(9):
                    angle = i * 2 * math.pi / 9
                    direction = Vector2D.from_angle(angle)
                    speed = 3 + math.sin(self.frame_counter * 0.1 + i) * 0.5
                    self.bullets.append(Bullet(enemy.position.copy(), direction * speed, QColor(0, 200, 255), BulletType.WAVE))
                # 爆炸彈
                if self.frame_counter % 24 == 0:
                    x = random.uniform(100, GAME_WIDTH - 100)
                    y = random.uniform(100, WINDOW_HEIGHT * 2 / 3)
                    for i in range(9):
                        explode_angle = i * 2 * math.pi / 9
                        explode_dir = Vector2D.from_angle(explode_angle)
                        self.bullets.append(Bullet(Vector2D(x, y), explode_dir * 2, QColor(100, 200, 255), BulletType.NORMAL))
            
            elif enemy.attack_mode == 3:
                # 隨機彈幕 + 陷阱彈幕
                for i in range(12):
                    random_angle = random.uniform(0, 2 * math.pi)
                    random_speed = random.uniform(2, 4)
                    direction = Vector2D.from_angle(random_angle)
                    self.bullets.append(Bullet(enemy.position.copy(), direction * random_speed, QColor(255, 200, 0), BulletType.NORMAL))
                # 陷阱彈幕
                if self.frame_counter % 30 == 0:
                    target_x = self.player.position.x + random.uniform(-150, 150)
                    target_y = self.player.position.y + random.uniform(-100, 0)
                    target_x = max(50, min(target_x, GAME_WIDTH - 50))
                    target_y = max(50, min(target_y, WINDOW_HEIGHT - 50))
                    for i in range(7):
                        trap_angle = i * 2 * math.pi / 7
                        trap_dir = Vector2D.from_angle(trap_angle)
                        self.bullets.append(Bullet(Vector2D(target_x, target_y), trap_dir * 0.1, QColor(255, 150, 0), BulletType.TRAP))
    
    def paintEvent(self, event: QPaintEvent):
        """繪製遊戲畫面"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.game_over:
            self.draw_game(painter)
            self.draw_ui(painter)
            if self.is_paused:
                self.draw_pause_overlay(painter)
        else:
            self.draw_game_over(painter)
    
    def draw_game(self, painter: QPainter):
        """繪製遊戲場景"""
        if 'background' in self.images:
            background_img = self.images['background'].scaled(
                GAME_WIDTH, WINDOW_HEIGHT, Qt.AspectRatioMode.IgnoreAspectRatio)
            painter.drawPixmap(0, 0, background_img)
        else:
            painter.fillRect(0, 0, GAME_WIDTH, WINDOW_HEIGHT, QColor(0, 0, 30))
        if 'scoreboard' in self.images:
            scoreboard_img = self.images['scoreboard'].scaled(
                WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT, Qt.AspectRatioMode.IgnoreAspectRatio)
            painter.drawPixmap(GAME_WIDTH, 0, scoreboard_img)
        else:
            painter.fillRect(GAME_WIDTH, 0, WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT, QColor(30, 0, 30))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(GAME_WIDTH, 0, GAME_WIDTH, WINDOW_HEIGHT)
        self.draw_player(painter)
        for ball in self.balls:
            self.draw_ball(painter, ball)
        for enemy in self.enemies:
            self.draw_enemy(painter, enemy)
        for bullet in self.bullets:
            self.draw_bullet(painter, bullet)
    
    def draw_player(self, painter: QPainter):
        """繪製玩家"""
        if 'player' in self.images:
            original_img = self.images['player']
            ratio = min(PLAYER_WIDTH / original_img.width(), PLAYER_HEIGHT / original_img.height())
            new_width = int(original_img.width() * ratio)
            new_height = int(original_img.height() * ratio)
            player_img = original_img.scaled(
                new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation)
            if self.player.is_invincible and self.frame_counter % 8 < 4:
                painter.setOpacity(0.5)
            else:
                painter.setOpacity(1.0)
            offset_x = (PLAYER_WIDTH - new_width) // 2
            offset_y = (PLAYER_HEIGHT - new_height) // 2
            img_rect = QRectF(
                self.player.position.x - PLAYER_WIDTH // 2 + offset_x,
                self.player.position.y - PLAYER_HEIGHT // 2 + offset_y,
                new_width, new_height
            )
            painter.drawPixmap(img_rect, player_img, QRectF(player_img.rect()))
            painter.setOpacity(1.0)
        else:
            if self.player.is_invincible and self.frame_counter % 8 < 4:
                painter.setBrush(QBrush(QColor(200, 200, 255, 100)))
            else:
                painter.setBrush(QBrush(QColor(200, 200, 255)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            rect = QRectF(self.player.position.x - PLAYER_WIDTH // 2,self.player.position.y - PLAYER_HEIGHT // 2, PLAYER_WIDTH, PLAYER_HEIGHT)
            painter.drawRect(rect)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawEllipse(QPointF(self.player.position.x, self.player.position.y), 5, 5)
        if self.player.is_invincible:
            current_invincible_time = max(MIN_INVINCIBLE_TIME, BASE_INVINCIBLE_TIME - self.game_level)
            invincible_ratio = self.player.invincible_counter / current_invincible_time
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(QPen(QColor(50, 150, 255, 200), 3))
            remaining_ratio = 1 - invincible_ratio
            start_angle = (270 + 180) * 16
            span_angle = int(360 * 16 * remaining_ratio)
            rect = QRectF(self.player.position.x - 60, self.player.position.y - 60, 120, 120)
            painter.drawArc(rect, start_angle, span_angle)
    
    def draw_ball(self, painter: QPainter, ball: Ball):
        """繪製球"""
        if 'ball' in self.images:
            ball_img = self.images['ball'].scaled(
                int(BALL_RADIUS * 2), int(BALL_RADIUS * 2), Qt.AspectRatioMode.KeepAspectRatio)
            painter.save()
            painter.translate(ball.position.x, ball.position.y)
            painter.rotate(math.degrees(ball.rotation))
            img_rect = QRectF(-BALL_RADIUS, -BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
            painter.drawPixmap(img_rect, ball_img, QRectF(ball_img.rect()))
            painter.restore()
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.save()
            painter.translate(ball.position.x, ball.position.y)
            painter.rotate(math.degrees(ball.rotation))
            painter.drawEllipse(-BALL_RADIUS, -BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawLine(-BALL_RADIUS // 2, 0, BALL_RADIUS // 2, 0)
            painter.restore()
    
    def draw_enemy(self, painter: QPainter, enemy: Enemy):
        """繪製敵人"""
        image_key_map = {
            EnemyType.SMALL: 'small_enemy',
            EnemyType.MEDIUM: 'medium_enemy',
            EnemyType.LARGE: 'large_enemy',
            EnemyType.BOSS: 'boss_enemy',
            EnemyType.BOSS_2: 'boss2_enemy'
        }
        image_key = image_key_map.get(enemy.type, 'small_enemy')
        if image_key in self.images:
            enemy_img = self.images[image_key].scaled(int(enemy.size), int(enemy.size), Qt.AspectRatioMode.KeepAspectRatio)
            img_rect = QRectF(
                enemy.position.x - enemy.size // 2,
                enemy.position.y - enemy.size // 2,
                enemy.size, enemy.size
            )
            painter.drawPixmap(img_rect, enemy_img, QRectF(enemy_img.rect()))
        else:
            painter.setBrush(QBrush(enemy.color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            rect = QRectF(enemy.position.x - enemy.size // 2, enemy.position.y - enemy.size // 2, enemy.size, enemy.size)
            painter.drawEllipse(rect)
        
        # 繪製Boss血量環
        if enemy.type in [EnemyType.BOSS, EnemyType.BOSS_2]:
            max_health = 50.0 if enemy.type == EnemyType.BOSS else 100.0
            health_ratio = enemy.health / max_health
            radius = (enemy.size + 30) / 2.0
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(QPen(QColor(100, 100, 100), 3))
            bg_rect = QRectF(enemy.position.x - radius, enemy.position.y - radius, radius * 2, radius * 2)
            painter.drawEllipse(bg_rect)
            if enemy.type == EnemyType.BOSS:
                painter.setPen(QPen(QColor(255, 50, 50), 3))
            else:
                # Boss2根據攻擊模式顯示不同顏色
                colors = [
                    QColor(200, 0, 200),   # 紫色 - 追蹤彈幕
                    QColor(255, 0, 0),     # 紅色 - 雷射攻擊
                    QColor(0, 200, 255),   # 藍色 - 波形彈幕
                    QColor(255, 200, 0)    # 黃色 - 隨機散射
                ]
                painter.setPen(QPen(colors[enemy.attack_mode], 3))
            start_angle = (-90 + 180) * 16
            span_angle = int(360 * 16 * health_ratio)
            painter.drawArc(bg_rect, start_angle, span_angle)
    
    def draw_bullet(self, painter: QPainter, bullet: Bullet):
        """繪製子彈"""
        if bullet.type == BulletType.NORMAL:
            if 'bullet' in self.images:
                bullet_img = self.images['bullet'].scaled(
                    bullet.size, bullet.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                img_rect = QRectF(
                    bullet.position.x - bullet.size // 2,
                    bullet.position.y - bullet.size // 2,
                    bullet.size, bullet.size
                )
                painter.drawPixmap(img_rect, bullet_img, QRectF(bullet_img.rect()))
            else:
                painter.setBrush(QBrush(bullet.color))
                painter.setPen(QPen(bullet.color, 1))
                painter.drawEllipse(QPointF(bullet.position.x, bullet.position.y), bullet.size // 2, bullet.size // 2)
        elif bullet.type == BulletType.LASER:
            if 'laser' in self.images:
                laser_img = self.images['laser'].scaled(
                    20, bullet.size * 4, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                alpha = max(50, int(255 * bullet.lifespan / 60))
                painter.setOpacity(alpha / 255.0)
                painter.save()
                painter.translate(bullet.position.x, bullet.position.y)
                rotation_angle = math.degrees(bullet.angle) + 90
                painter.rotate(rotation_angle)
                img_rect = QRectF(
                    -10, -bullet.size * 2,
                    20, bullet.size * 4
                )
                painter.drawPixmap(img_rect, laser_img, QRectF(laser_img.rect()))
                painter.restore()
                painter.setOpacity(1.0)
            else:
                alpha = max(50, int(255 * bullet.lifespan / 60))
                laser_color = QColor(bullet.color.red(), bullet.color.green(), bullet.color.blue(), alpha)
                painter.setBrush(QBrush(laser_color))
                painter.setPen(QPen(laser_color, 2))
                painter.save()
                painter.translate(bullet.position.x, bullet.position.y)
                rotation_angle = math.degrees(bullet.angle) + 90
                painter.rotate(rotation_angle)
                rect = QRectF(-10, -bullet.size * 2, 20, bullet.size * 4)
                painter.drawRect(rect)
                painter.restore()
        elif bullet.type == BulletType.WAVE:
            if 'wave' in self.images:
                wave_img = self.images['wave'].scaled(
                    bullet.size, bullet.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                img_rect = QRectF(
                    bullet.position.x - bullet.size // 2,
                    bullet.position.y - bullet.size // 2,
                    bullet.size, bullet.size
                )
                painter.drawPixmap(img_rect, wave_img, QRectF(wave_img.rect()))
            else:
                painter.setBrush(QBrush(bullet.color))
                painter.setPen(QPen(bullet.color, 1))
                painter.drawEllipse(QPointF(bullet.position.x, bullet.position.y),
                                  bullet.size // 2, bullet.size // 2)
        elif bullet.type == BulletType.TRAP:
            if 'bullet' in self.images:
                bullet_img = self.images['bullet'].scaled(
                    bullet.size, bullet.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                if not bullet.activated:
                    alpha = int(150 + 105 * math.sin(self.frame_counter * 0.2))
                    scale = 0.8 + 0.4 * math.sin(self.frame_counter * 0.3)
                    painter.setOpacity(alpha / 255.0)
                    size = bullet.size * scale
                    img_rect = QRectF(
                        bullet.position.x - size // 2,
                        bullet.position.y - size // 2,
                        size, size
                    )
                    painter.drawPixmap(img_rect, bullet_img, QRectF(bullet_img.rect()))
                    painter.setOpacity(1.0)
                else:
                    img_rect = QRectF(
                        bullet.position.x - bullet.size // 2,
                        bullet.position.y - bullet.size // 2,
                        bullet.size, bullet.size
                    )
                    painter.drawPixmap(img_rect, bullet_img, QRectF(bullet_img.rect()))
            else:
                painter.setBrush(QBrush(bullet.color))
                painter.setPen(QPen(bullet.color, 1))
                if not bullet.activated:
                    alpha = int(150 + 105 * math.sin(self.frame_counter * 0.2))
                    scale = 0.8 + 0.4 * math.sin(self.frame_counter * 0.3)
                    trap_color = QColor(bullet.color.red(), bullet.color.green(), bullet.color.blue(), alpha)
                    painter.setBrush(QBrush(trap_color))
                    size = bullet.size * scale
                    painter.drawEllipse(QPointF(bullet.position.x, bullet.position.y), size // 2, size // 2)
                else:
                    painter.drawEllipse(QPointF(bullet.position.x, bullet.position.y), bullet.size // 2, bullet.size // 2)
    
    def draw_ui(self, painter: QPainter):
        """繪製遊戲UI"""
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        painter.drawText(GAME_WIDTH + 20, 50, "長門櫻 彈跳球")
        painter.setFont(self.font)
        painter.drawText(GAME_WIDTH + 20, 100, f"分數: {self.score}")
        painter.drawText(GAME_WIDTH + 20, 140, f"生命: {self.player.health}")
        painter.drawText(GAME_WIDTH + 20, 180, f"球數: {len(self.balls)}")
        painter.drawText(GAME_WIDTH + 20, 220, f"等級: {self.game_level}")
        current_invincible_time = max(MIN_INVINCIBLE_TIME, BASE_INVINCIBLE_TIME - self.game_level)
        painter.drawText(GAME_WIDTH + 20, 260, f"無敵時間: {current_invincible_time/60.0:.1f}s")
        boss_progress = min(1.0, self.boss_score_counter / BOSS_SCORE_THRESHOLD)
        painter.drawText(GAME_WIDTH + 20, 300, "BOSS 進度:")
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawRect(GAME_WIDTH + 20, 310, 300, 20)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawRect(GAME_WIDTH + 20, 310, int(300 * boss_progress), 20)
        boss2_progress = min(1.0, self.boss2_score_counter / BOSS2_SCORE_THRESHOLD)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(GAME_WIDTH + 20, 360, "超級BOSS 進度:")
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawRect(GAME_WIDTH + 20, 370, 300, 20)
        painter.setBrush(QBrush(QColor(200, 0, 200)))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawRect(GAME_WIDTH + 20, 370, int(300 * boss2_progress), 20)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(GAME_WIDTH + 20, 950, "開發者: 天野靜樹")
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(GAME_WIDTH + 20, 420, "遊戲控制:")
        painter.drawText(GAME_WIDTH + 20, 440, "空白鍵 / P - 暫停/繼續")
        painter.drawText(GAME_WIDTH + 20, 460, "ESC - 退出遊戲")
        if self.is_paused:
            painter.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.drawText(GAME_WIDTH + 20, 500, "【遊戲已暫停】")
    
    def draw_pause_overlay(self, painter: QPainter):
        """繪製暫停覆蓋層"""
        overlay_color = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, GAME_WIDTH, WINDOW_HEIGHT, overlay_color)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Microsoft YaHei", 48, QFont.Weight.Bold))
        pause_text = "遊戲暫停"
        text_rect = QRectF(0, WINDOW_HEIGHT / 2 - 120, GAME_WIDTH, 80)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, pause_text)
        painter.setFont(QFont("Microsoft YaHei", 16))
        hint_text = "按空白鍵或P鍵繼續遊戲"
        hint_rect = QRectF(0, WINDOW_HEIGHT / 2 - 20, GAME_WIDTH, 40)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter, hint_text)
        click_text = "或點擊畫面繼續"
        click_rect = QRectF(0, WINDOW_HEIGHT / 2 + 20, GAME_WIDTH, 40)
        painter.drawText(click_rect, Qt.AlignmentFlag.AlignCenter, click_text)
        exit_text = "按ESC鍵退出遊戲"
        exit_rect = QRectF(0, WINDOW_HEIGHT / 2 + 60, GAME_WIDTH, 40)
        painter.setPen(QPen(QColor(255, 200, 200)))
        painter.drawText(exit_rect, Qt.AlignmentFlag.AlignCenter, exit_text)
    
    def draw_game_over(self, painter: QPainter):
        """繪製遊戲結束畫面"""
        if 'background' in self.images:
            background_img = self.images['background'].scaled(
                GAME_WIDTH, WINDOW_HEIGHT, Qt.AspectRatioMode.IgnoreAspectRatio)
            painter.drawPixmap(0, 0, background_img)
        else:
            painter.fillRect(0, 0, GAME_WIDTH, WINDOW_HEIGHT, QColor(0, 0, 30))
        if 'scoreboard' in self.images:
            scoreboard_img = self.images['scoreboard'].scaled(
                WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT, Qt.AspectRatioMode.IgnoreAspectRatio)
            painter.drawPixmap(GAME_WIDTH, 0, scoreboard_img)
        else:
            painter.fillRect(GAME_WIDTH, 0, WINDOW_WIDTH - GAME_WIDTH, WINDOW_HEIGHT, QColor(30, 0, 30))
        
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(self.game_over_font)
        text_rect = QRectF((WINDOW_WIDTH - 600) / 2 - 200, WINDOW_HEIGHT / 2 - 100, 400, 100)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "遊戲結束")
        painter.setFont(QFont("Microsoft YaHei", 20))
        score_rect = QRectF((WINDOW_WIDTH - 600) / 2 - 200, WINDOW_HEIGHT / 2, 400, 50)
        painter.drawText(score_rect, Qt.AlignmentFlag.AlignCenter, f"最終分數: {self.score}")
        restart_rect = QRectF((WINDOW_WIDTH - 600) / 2 - 200, WINDOW_HEIGHT / 2 + 80, 400, 50)
        painter.drawText(restart_rect, Qt.AlignmentFlag.AlignCenter, "點擊重新開始")
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """處理滑鼠移動事件"""
        self.mouse_x = event.position().x()
    
    def mousePressEvent(self, event: QMouseEvent):
        """處理滑鼠點擊事件"""
        if self.game_over:
            self.restart_game()
        elif self.is_paused:
            self.toggle_pause()
    
    def keyPressEvent(self, event: QKeyEvent):
        """處理鍵盤按鍵事件"""
        if event.key() == Qt.Key.Key_Space:
            if not self.game_over:
                self.toggle_pause()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_P:
            if not self.game_over:
                self.toggle_pause()
        super().keyPressEvent(event)
    
    def toggle_pause(self):
        """切換暫停狀態"""
        if not self.game_over:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_music()
                logging.info("遊戲暫停，音樂已暫停")
            else:
                self.resume_music()
                logging.info("遊戲繼續，音樂已恢復")
            logging.info(f"遊戲{'暫停' if self.is_paused else '繼續'}")
    
    def restart_game(self):
        """重新開始遊戲"""
        if self.is_paused:
            self.is_paused = False
        self.init_game()
        if hasattr(self, 'bgm_path') and self.bgm_path:
            self.stop_music()
            time.sleep(0.1)
            self.start_music_thread()
    
    def closeEvent(self, event):
        """處理窗口關閉事件"""
        try:
            logging.info("正在關閉遊戲...")
            if hasattr(self, 'game_timer'):
                self.game_timer.stop()
                logging.info("遊戲計時器已停止")
            self.force_stop_music()
            logging.info("遊戲關閉完成")
        except Exception as e:
            logging.error(f"關閉遊戲時發生錯誤: {e}")
        super().closeEvent(event)
    
    def force_stop_music(self):
        """強制停止音樂播放（確保徹底清理）"""
        try:
            logging.info("強制停止音樂...")
            self.should_stop_music = True
            self.is_music_paused = False
            if hasattr(self, 'music_process') and self.music_process:
                try:
                    self.music_process.terminate()
                    self.music_process.wait(timeout=1)
                    logging.info("ffplay 進程已正常終止")
                except subprocess.TimeoutExpired:
                    logging.warning("ffplay 進程未正常終止，強制殺死")
                    self.music_process.kill()
                    try:
                        self.music_process.wait(timeout=1)
                    except:
                        pass
                except Exception as e:
                    logging.error(f"終止 ffplay 進程時發生錯誤: {e}")
                finally:
                    self.music_process = None
            if hasattr(self, 'music_thread') and self.music_thread and self.music_thread.is_alive():
                logging.info("等待音樂線程結束...")
                self.music_thread.join(timeout=2)
                if self.music_thread.is_alive():
                    logging.warning("音樂線程未能在時限內結束")
                else:
                    logging.info("音樂線程已結束")
            self.is_music_playing = False
            logging.info("音樂播放已完全停止")
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/f', '/im', 'ffplay.exe'], 
                                 capture_output=True, timeout=3)
                else:  # Linux/Mac
                    subprocess.run(['pkill', '-f', 'ffplay'], 
                                 capture_output=True, timeout=3)
                logging.info("已清理任何殘留的 ffplay 進程")
            except Exception as e:
                logging.debug(f"清理殘留進程時發生錯誤（可能正常）: {e}")
        except Exception as e:
            logging.error(f"強制停止音樂時發生錯誤: {e}")


class EasterEggDialog(QDialog):
    """彩蛋遊戲對話框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("長門櫻 彈跳球")
        self.setModal(True)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        try:
            current_directory = Path(__file__).resolve().parent.parent.parent
            icon_path = current_directory / "assets" / "icon" / "1.3.0.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
            else:
                icon_dir = current_directory / "assets" / "icon"
                if icon_dir.exists():
                    icon_files = list(icon_dir.glob("*.ico"))
                    if icon_files:
                        latest_icon = sorted(icon_files)[-1]
                        self.setWindowIcon(QIcon(str(latest_icon)))
                        logging.info(f"彩蛋對話框使用圖示: {latest_icon.name}")
        except Exception as e:
            logging.warning(f"無法載入對話框圖示: {e}")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.game = NagatoSakuraBounceGame()
        layout.addWidget(self.game)
        self.setLayout(layout)
        logging.info("彩蛋遊戲已啟動")
    
    def closeEvent(self, event):
        """處理對話框關閉事件"""
        try:
            logging.info("正在關閉彩蛋遊戲對話框...")
            if hasattr(self, 'game') and self.game:
                self.game.force_stop_music()
                if hasattr(self.game, 'game_timer'):
                    self.game.game_timer.stop()
                logging.info("遊戲資源已清理")
            logging.info("彩蛋遊戲對話框已關閉")
        except Exception as e:
            logging.error(f"關閉彩蛋對話框時發生錯誤: {e}")
        super().closeEvent(event)


def show_easter_egg():
    """顯示彩蛋遊戲"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    dialog = EasterEggDialog()
    dialog.exec()
if __name__ == "__main__":
    show_easter_egg()
