import sys
import math
import time
import random
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QApplication, QLabel)
from PyQt6.QtGui import QPainter, QColor, QKeyEvent, QFont, QPen, QBrush, QCursor, QPolygon, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QPoint, QRect

# --- 基本設定 ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MAP_WIDTH = 16
MAP_HEIGHT = 16
CELL_SIZE = 64 
MOUSE_SENSITIVITY = 0.0005
NAGATO_ICON_PATH = "assets/icon/1.0.2.ico"

# --- 地圖 ---
# 0 = 空地, 1 = 牆壁 (櫻花粉色), 2 = 牆壁 (深粉色)
world_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 2, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# 櫻花裝飾位置 (未使用)
cherry_blossoms = [
    (2, 1), (5, 3), (8, 2), (12, 3), (14, 2),
    (3, 13), (7, 13), (11, 13), (9, 9), (5, 7)
]

# --- 精靈位置 (長門櫻的位置) ---
NAGATO_X = 13.5 * CELL_SIZE
NAGATO_Y = 13.5 * CELL_SIZE
NAGATO_SIZE = 1.5 * CELL_SIZE  
NAGATO_INTERACTION_DISTANCE = 2.5 * CELL_SIZE

# --- 長門櫻的對話 ---
NAGATO_DIALOGUES = [
    "主人，您找到長門櫻了嗎？長門櫻一直在等您呢～ 🌸",
    "主人，這裡是長門櫻的秘密基地，您喜歡嗎？ 🌸",
    "主人，能和您一起探險真是太開心了！ 🌸",
    "主人，這些牆壁都是櫻花顏色的呢，是長門櫻特別為您準備的～ 🌸",
    "主人，您走了這麼遠的路，累不累？要不要休息一下？ 🌸",
    "主人，長門櫻很開心能和您在這裡相遇！ 🌸",
    "主人，您摸摸長門櫻的耳朵嗎？有點害羞呢... 🌸"
]

# --- Raycasting 繪圖 Widget ---
class RaycastingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.player_x = 3.5 * CELL_SIZE
        self.player_y = 3.5 * CELL_SIZE
        self.player_angle = math.pi / 4
        self.fov = math.pi / 3
        self.move_speed = 15.0
        self.rot_speed = 0.08 # 旋轉速度 (未使用，改由滑鼠控制)
        self.strafe_speed = 10.0
        self.nagato_found = False
        self.nagato_dialogue_index = 0
        self.nagato_interaction_timer = 0
        self.nagato_animation_offset = 0.0 
        self.cherry_blossom_animation = 0.0 
        self.show_victory = False
        self.victory_time = 0

        # 載入長門櫻圖示
        self.nagato_icon = QPixmap(NAGATO_ICON_PATH)
        if self.nagato_icon.isNull():
            print(f"警告：無法載入長門櫻圖示於 {NAGATO_ICON_PATH}")
            self.nagato_icon = None

        # 櫻花飄落效果
        self.falling_petals = []
        for _ in range(30):
            self.falling_petals.append({
                'x': random.uniform(0, SCREEN_WIDTH),
                'y': random.uniform(0, SCREEN_HEIGHT),
                'size': random.uniform(3, 8),
                'speed': random.uniform(1, 3),
                'wobble': random.uniform(0, 2 * math.pi),
                'wobble_speed': random.uniform(0.01, 0.05)
            })

        # 按鍵狀態
        self.key_forward = False
        self.key_backward = False
        self.key_left = False
        self.key_right = False
        self.key_interact = False

        # 滑鼠控制
        self.mouse_active = False
        self.mouse_center = QPoint(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.last_mouse_pos = self.mouse_center
        self.screen_center = self.mapToGlobal(self.mouse_center) 

        # 牆壁顏色
        self.wall_colors = {
            1: QColor(255, 182, 193),
            2: QColor(255, 105, 180),
        }
        self.dark_factor = 0.7 

        # 遊戲計時器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)
        self.timer.start(16) 
        self.last_time = time.time()
        self.show_instructions = True
        self.instructions_timer = QTimer(self)
        self.instructions_timer.setSingleShot(True)
        self.instructions_timer.timeout.connect(self.hide_instructions)
        self.instructions_timer.start(5000) 

        # 當前對話文字
        self.current_dialogue = ""
        self.dialogue_timer = QTimer(self)
        self.dialogue_timer.timeout.connect(self.update_dialogue)
        self.footstep_timer = 0 

    def hide_instructions(self):
        self.show_instructions = False

    def update_dialogue(self):
        """隨機更新長門櫻的對話"""
        if self.nagato_found and not self.show_victory:
            self.current_dialogue = random.choice(NAGATO_DIALOGUES)
            self.dialogue_timer.start(random.randint(3000, 6000))

    def update_game(self):
        """更新遊戲狀態"""
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        actual_move_speed = self.move_speed * delta_time * 10
        actual_strafe_speed = self.strafe_speed * delta_time * 10

        # 勝利畫面處理
        if self.nagato_found and self.show_victory:
            self.victory_time += delta_time
            # 更新勝利畫面的櫻花飄落
            for petal in self.falling_petals:
                petal['y'] += petal['speed'] * delta_time * 30
                petal['wobble'] += petal['wobble_speed']
                petal['x'] += math.sin(petal['wobble']) * 0.5
                if petal['y'] > SCREEN_HEIGHT:
                    petal['y'] = 0
                    petal['x'] = random.uniform(0, SCREEN_WIDTH)
            self.cherry_blossom_animation += delta_time * 2
            self.update()
            return

        # 移動計算
        move_x = 0.0
        move_y = 0.0
        if self.key_forward:
            move_x += math.cos(self.player_angle) * actual_move_speed
            move_y += math.sin(self.player_angle) * actual_move_speed
            self.footstep_timer += delta_time
        if self.key_backward:
            move_x -= math.cos(self.player_angle) * actual_move_speed
            move_y -= math.sin(self.player_angle) * actual_move_speed
            self.footstep_timer += delta_time
        if self.key_left:
            move_x += math.cos(self.player_angle - math.pi/2) * actual_strafe_speed
            move_y += math.sin(self.player_angle - math.pi/2) * actual_strafe_speed
            self.footstep_timer += delta_time
        if self.key_right:
            move_x += math.cos(self.player_angle + math.pi/2) * actual_strafe_speed
            move_y += math.sin(self.player_angle + math.pi/2) * actual_strafe_speed
            self.footstep_timer += delta_time

        # 碰撞檢測
        next_x = self.player_x + move_x
        next_y = self.player_y + move_y
        map_x = int(next_x / CELL_SIZE)
        map_y = int(next_y / CELL_SIZE)
        if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
            if world_map[map_y][map_x] == 0 or world_map[map_y][map_x] == 3: # 可通行
                self.player_x = next_x
                self.player_y = next_y
            else: # 撞牆，嘗試滑動
                 map_x_only = int((self.player_x + move_x) / CELL_SIZE)
                 map_y_only = int(self.player_y / CELL_SIZE)
                 if 0 <= map_x_only < MAP_WIDTH and 0 <= map_y_only < MAP_HEIGHT and (world_map[map_y_only][map_x_only] == 0 or world_map[map_y_only][map_x_only] == 3):
                     self.player_x += move_x
                 map_x_only = int(self.player_x / CELL_SIZE)
                 map_y_only = int((self.player_y + move_y) / CELL_SIZE)
                 if 0 <= map_x_only < MAP_WIDTH and 0 <= map_y_only < MAP_HEIGHT and (world_map[map_y_only][map_x_only] == 0 or world_map[map_y_only][map_x_only] == 3):
                     self.player_y += move_y

        # 檢查與長門櫻的距離和互動
        dist_to_nagato = math.sqrt((self.player_x - NAGATO_X)**2 + (self.player_y - NAGATO_Y)**2)
        if dist_to_nagato < NAGATO_INTERACTION_DISTANCE:
            if not self.nagato_found:
                self.nagato_found = True
                self.current_dialogue = "主人，您找到長門櫻了！長門櫻等您好久了～ 🌸"
                self.dialogue_timer.start(5000)
            if self.key_interact and not self.show_victory:
                self.show_victory = True
                self.current_dialogue = "主人～謝謝您來找長門櫻，我們一起回家吧！ ❤️"

        # 更新櫻花飄落動畫
        for petal in self.falling_petals:
            petal['y'] += petal['speed'] * delta_time * 30
            petal['wobble'] += petal['wobble_speed']
            petal['x'] += math.sin(petal['wobble']) * 0.5
            if petal['y'] > SCREEN_HEIGHT:
                petal['y'] = 0
                petal['x'] = random.uniform(0, SCREEN_WIDTH)

        # 更新長門櫻圖示動畫
        self.nagato_animation_offset = math.sin(current_time * 2) * 5
        self.cherry_blossom_animation += delta_time * 2

        # 滑鼠控制：重置滑鼠到中心
        if self.mouse_active:
            current_pos = QCursor.pos()
            if current_pos != self.screen_center:
                QCursor.setPos(self.screen_center)
        self.update() 

    def paintEvent(self, event):
        """繪製畫面"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        if self.show_victory:
            self.draw_victory_screen(painter)
            painter.end()
            return

        # 繪製天空和地板
        sky_color = QColor(135, 206, 250)
        floor_color = QColor(100, 100, 100)
        painter.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2, sky_color)
        painter.fillRect(0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2, floor_color)
        z_buffer = [float('inf')] * SCREEN_WIDTH 

        # Raycasting
        for x in range(SCREEN_WIDTH):
            camera_x = 2 * x / SCREEN_WIDTH - 1
            try: # 處理FOV計算的邊界情況
                if abs(math.cos(self.fov / 2)) < 1e-9: fov_factor = float('inf')
                else: fov_factor = 1.0 / math.tan(self.fov / 2)
                ray_angle = self.player_angle + math.atan2(camera_x, fov_factor)
            except ValueError: ray_angle = self.player_angle
            ray_angle %= (2 * math.pi)
            ray_dir_x = math.cos(ray_angle)
            ray_dir_y = math.sin(ray_angle)
            map_x = int(self.player_x / CELL_SIZE)
            map_y = int(self.player_y / CELL_SIZE)
            delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else float('inf')
            delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else float('inf')
            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (self.player_x - map_x * CELL_SIZE) / CELL_SIZE * delta_dist_x
            else:
                step_x = 1
                side_dist_x = ((map_x + 1) * CELL_SIZE - self.player_x) / CELL_SIZE * delta_dist_x
            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (self.player_y - map_y * CELL_SIZE) / CELL_SIZE * delta_dist_y
            else:
                step_y = 1
                side_dist_y = ((map_y + 1) * CELL_SIZE - self.player_y) / CELL_SIZE * delta_dist_y
            hit = 0
            side = 0
            while hit == 0:
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
                    if world_map[map_y][map_x] > 0 and world_map[map_y][map_x] < 3:
                        hit = world_map[map_y][map_x]
                else:
                    hit = 1; break 

            # 修正魚眼
            if side == 0: perp_wall_dist = (map_x * CELL_SIZE - self.player_x + (1 - step_x) * CELL_SIZE / 2) / ray_dir_x if ray_dir_x != 0 else float('inf')
            else: perp_wall_dist = (map_y * CELL_SIZE - self.player_y + (1 - step_y) * CELL_SIZE / 2) / ray_dir_y if ray_dir_y != 0 else float('inf')
            if perp_wall_dist <= 1e-5: perp_wall_dist = 1e-5
            z_buffer[x] = perp_wall_dist 

            # 計算牆高和繪製範圍
            line_height = int(SCREEN_HEIGHT / (perp_wall_dist / CELL_SIZE)) if perp_wall_dist > 0 else SCREEN_HEIGHT
            draw_start = max(0, -line_height // 2 + SCREEN_HEIGHT // 2)
            draw_end = min(SCREEN_HEIGHT - 1, line_height // 2 + SCREEN_HEIGHT // 2)

            # 選擇牆色並根據側面變暗
            wall_color = self.wall_colors.get(hit, QColor(128, 128, 128))
            if side == 1: wall_color = wall_color.darker(int(100 / self.dark_factor))

            # 繪製牆壁條帶
            wall_height = draw_end - draw_start + 1
            if wall_height > 0:
                painter.fillRect(x, draw_start, 1, wall_height, wall_color)

        # 繪製長門櫻圖示
        self.draw_nagato_sprite(painter, z_buffer)

        # 繪製櫻花飄落效果 (開啟抗鋸齒)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor(255, 192, 203, 150))
        painter.setBrush(QColor(255, 192, 203, 150))
        for petal in self.falling_petals:
            petal_size = petal['size']
            petal_x = petal['x']
            petal_y = petal['y'] + self.nagato_animation_offset * 0.3
            points = []
            for i in range(5):
                angle = 2 * math.pi * i / 5 + self.cherry_blossom_animation
                px = petal_x + math.cos(angle) * petal_size
                py = petal_y + math.sin(angle) * petal_size
                points.append(QPoint(int(px), int(py)))
            painter.drawPolygon(QPolygon(points))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # 繪製對話框和說明 (開啟抗鋸齒繪製文字)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.nagato_found and self.current_dialogue:
            painter.fillRect(50, SCREEN_HEIGHT - 100, SCREEN_WIDTH - 100, 80, QColor(0, 0, 0, 200))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("微軟正黑體", 12))
            painter.drawText(60, SCREEN_HEIGHT - 80, SCREEN_WIDTH - 120, 60, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.current_dialogue)
        if self.show_instructions:
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setFont(QFont("微軟正黑體", 12))
            painter.fillRect(10, 10, 400, 150, QColor(0, 0, 0, 150))
            painter.drawText(20, 40, "使用 W/S 前後移動, A/D 左右移動")
            painter.drawText(20, 70, "滑鼠可控制視角旋轉")
            painter.drawText(20, 100, "按下 E 鍵與長門櫻互動")
            painter.drawText(20, 130, "找到長門櫻，她在等著您！ 🌸")
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False) # 恢復

        # 繪製十字準星
        if self.mouse_active:
            painter.setPen(QColor(255, 255, 255, 120))
            painter.drawLine(SCREEN_WIDTH//2-10, SCREEN_HEIGHT//2, SCREEN_WIDTH//2+10, SCREEN_HEIGHT//2)
            painter.drawLine(SCREEN_WIDTH//2, SCREEN_HEIGHT//2-10, SCREEN_WIDTH//2, SCREEN_HEIGHT//2+10)

        painter.end()

    def draw_nagato_sprite(self, painter, z_buffer):
        """繪製長門櫻圖示"""
        if not self.nagato_icon: return
        sprite_dir_x = NAGATO_X - self.player_x
        sprite_dir_y = NAGATO_Y - self.player_y
        sprite_distance = math.sqrt(sprite_dir_x**2 + sprite_dir_y**2)
        if sprite_distance < 1: return

        # 計算螢幕大小
        sprite_base_scale = 1.0
        sprite_size = int(SCREEN_HEIGHT / (sprite_distance / CELL_SIZE) * (NAGATO_SIZE / CELL_SIZE) * sprite_base_scale)
        if sprite_size <= 5: return

        # 計算相對角度
        sprite_angle = math.atan2(sprite_dir_y, sprite_dir_x)
        relative_angle = sprite_angle - self.player_angle
        while relative_angle > math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi

        # 視野檢查
        sprite_angular_width_approx = math.atan((NAGATO_SIZE / 2) / sprite_distance) if sprite_distance > 0 else math.pi
        if abs(relative_angle) > self.fov / 2 + sprite_angular_width_approx: return

        # 計算螢幕X座標
        sprite_screen_x = int((0.5 + relative_angle / self.fov) * SCREEN_WIDTH)

        # 計算繪製參數
        draw_width = sprite_size
        draw_height = sprite_size
        draw_x = sprite_screen_x - draw_width // 2
        vertical_offset_factor = 0.1
        draw_y = int(SCREEN_HEIGHT / 2 + sprite_size * vertical_offset_factor - draw_height + self.nagato_animation_offset)

        # Z-buffer 檢查 (逐列)
        sprite_start_x = max(0, draw_x)
        sprite_end_x = min(SCREEN_WIDTH - 1, draw_x + draw_width)
        is_visible = False
        for check_x in range(sprite_start_x, sprite_end_x + 1):
            if sprite_distance <= z_buffer[check_x]:
                is_visible = True
                break 

        if is_visible:
            target_rect = QRect(draw_x, draw_y, draw_width, draw_height)
            painter.drawPixmap(target_rect, self.nagato_icon)

            # 繪製互動提示 (開啟抗鋸齒)
            tip_y = draw_y - 30
            if sprite_size > 30 and not self.show_victory and 0 <= tip_y < SCREEN_HEIGHT:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("微軟正黑體", 10))
                text = "按 E 互動" if self.nagato_found else "長門櫻 🌸"
                painter.drawText(sprite_screen_x - 40, int(tip_y), 80, 20,
                                Qt.AlignmentFlag.AlignCenter, text)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def draw_victory_screen(self, painter):
        """繪製勝利畫面"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, QColor(0, 0, 0))
        painter.setPen(QColor(255, 182, 193))
        painter.setFont(QFont("微軟正黑體", 24, QFont.Weight.Bold))
        painter.drawText(0, 100, SCREEN_WIDTH, 50, Qt.AlignmentFlag.AlignCenter, "主人找到長門櫻了！")
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("微軟正黑體", 14))
        painter.drawText(50, 200, SCREEN_WIDTH - 100, 100, Qt.AlignmentFlag.AlignCenter,
                        "長門櫻非常開心主人找到她！\n以後，長門櫻會一直陪伴在主人身旁！")

        # 繪製長門櫻圖示
        if self.nagato_icon:
            icon_size = 150
            icon_x = SCREEN_WIDTH // 2 - icon_size // 2
            icon_y = SCREEN_HEIGHT // 2 + 50 - icon_size // 2
            victory_animation_offset = math.sin(time.time() * 2) * 10
            icon_y += int(victory_animation_offset)
            painter.drawPixmap(icon_x, icon_y, icon_size, icon_size, self.nagato_icon)

        # 提示文字
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("微軟正黑體", 12))
        if self.victory_time > 1.0:
            painter.drawText(0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 30, Qt.AlignmentFlag.AlignCenter, "按下 ESC 鍵返回")

        # 繪製飄落櫻花
        painter.setPen(QColor(255, 182, 193, 150))
        painter.setBrush(QColor(255, 182, 193, 150))
        for petal in self.falling_petals:
            petal_size = petal['size'] * 1.5
            petal_x = petal['x']
            petal_y = petal['y']
            points = []
            for i in range(5):
                angle = 2 * math.pi * i / 5 + self.cherry_blossom_animation
                px = petal_x + math.cos(angle) * petal_size
                py = petal_y + math.sin(angle) * petal_size
                points.append(QPoint(int(px), int(py)))
            painter.drawPolygon(QPolygon(points))

        # 恢復渲染設定
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    def mousePressEvent(self, event):
        """處理滑鼠按下，啟用視角控制"""
        if self.show_victory: return
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_active = True
            self.screen_center = self.mapToGlobal(self.mouse_center) 
            QCursor.setPos(self.screen_center)
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif event.button() == Qt.MouseButton.RightButton: 
            self.mouse_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        pass 

    def mouseMoveEvent(self, event):
        """處理滑鼠移動，旋轉視角"""
        if self.show_victory or not self.mouse_active: return
        mouse_pos = event.pos()
        dx = mouse_pos.x() - self.mouse_center.x()
        if dx != 0:
            self.player_angle = (self.player_angle + dx * MOUSE_SENSITIVITY) % (2 * math.pi)

    def keyPressEvent(self, event: QKeyEvent):
        """處理按鍵按下"""
        if self.show_victory:
            if event.key() == Qt.Key.Key_Escape:
                parent_dialog = self.parent()
                if isinstance(parent_dialog, QDialog): parent_dialog.close()
                else: self.close()
            return

        key = event.key()
        if key == Qt.Key.Key_W: self.key_forward = True
        elif key == Qt.Key.Key_S: self.key_backward = True
        elif key == Qt.Key.Key_A: self.key_left = True
        elif key == Qt.Key.Key_D: self.key_right = True
        elif key == Qt.Key.Key_E: self.key_interact = True
        elif key == Qt.Key.Key_Escape:
            if self.mouse_active:
                self.mouse_active = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                parent_dialog = self.parent()
                if isinstance(parent_dialog, QDialog): parent_dialog.close()
                else: self.close()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """處理按鍵釋放"""
        key = event.key()
        if key == Qt.Key.Key_W: self.key_forward = False
        elif key == Qt.Key.Key_S: self.key_backward = False
        elif key == Qt.Key.Key_A: self.key_left = False
        elif key == Qt.Key.Key_D: self.key_right = False
        elif key == Qt.Key.Key_E: self.key_interact = False
        else: super().keyReleaseEvent(event)

    def leaveEvent(self, event):
        """滑鼠離開窗口時取消控制"""
        if self.mouse_active:
            self.mouse_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

# --- 主對話框類別 ---
class NagatoSakuraEasterEggDialog(QDialog):
    """長門櫻 Raycasting 彩蛋對話框"""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowTitle("長門櫻的秘密基地")
        self.setModal(True)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.raycasting_widget = RaycastingWidget(self)
        self.main_layout.addWidget(self.raycasting_widget)
        self.setFixedSize(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setStyleSheet("background-color: #000000;")

    def closeEvent(self, event):
        """關閉窗口時停止計時器並恢復游標"""
        if hasattr(self, 'raycasting_widget'):
            if self.raycasting_widget.timer: self.raycasting_widget.timer.stop()
            self.raycasting_widget.setCursor(Qt.CursorShape.ArrowCursor)
        super().closeEvent(event)

# --- 快速測試 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = NagatoSakuraEasterEggDialog()
    dialog.show()
    sys.exit(app.exec())