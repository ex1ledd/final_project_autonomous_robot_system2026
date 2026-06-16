# %%
# ====================================================================
# PROGRAM EVALUASI AKHIR SEMESTER EAS
# HYBRID A* + PURE PURSUIT + LIVE ANTI-GHOST MAPPER NAVIGATION
#
# Nama : Laurensius Duta Wicaksono
# NRP  : 5022231070
# Robot: Pioneer P3DX
# Simulator: CoppeliaSim ZeroMQ Remote API + Matplotlib Ray-Clearing Mapper
# ====================================================================

import time
import math
import heapq
import requests
import numpy as np
import matplotlib.pyplot as plt  
from collections import deque
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ====================================================================
# CONFIGURATION
# ====================================================================

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

SIMULATION_DURATION = 10000.0
CONTROL_DT = 0.05

# P3DX kinematics
WHEEL_RADIUS = 0.195 / 2.0
HALF_WHEEL_BASE = 0.318 / 2.0

# Motor limit
MAX_WHEEL_SPEED = 4.8

# Velocity limits
MAX_LINEAR_VEL = 0.45
MAX_ANGULAR_VEL = 1.8
MAX_ACCEL = 0.055
MAX_ALPHA = 0.12

# Arrival
TARGET_REACHED_DIST = 0.2

# Sensors
MAX_SENSOR_RANGE = 1
MIN_VALID_SENSOR_RANGE = 0.02
OBSTACLE_CRITICAL_DIST = 0.25
OBSTACLE_AVOID_DIST = 0.28
EMERGENCY_DIST = 0.18

# Occupancy grid
GRID_RESOLUTION = 0.12
MAP_X_MIN = -6.0
MAP_X_MAX = 6.0
MAP_Y_MIN = -6.0
MAP_Y_MAX = 6.0
OBSTACLE_INFLATION_RADIUS = 0.28

# Planning
REPLAN_INTERVAL = 0.5
PATH_DEVIATION_REPLAN_DIST = 0.65
A_STAR_GOAL_TOL_CELLS = 2

# Pure pursuit
LOOKAHEAD_BASE = 0.40
LOOKAHEAD_GAIN = 0.45
HEADING_GAIN = 1.8

# Recovery
STALL_WINDOW_SEC = 3.0
STALL_MIN_DISPLACEMENT = 0.03
RECOVERY_REVERSE_STEPS = 35
RECOVERY_ROTATE_STEPS = 45

# Sensor smoothing
SENSOR_EMA_ALPHA = 0.45


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def world_distance_to_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    if dx * dx + dy * dy == 0:
        return euclidean(point, a)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)

    closest = (ax + t * dx, ay + t * dy)
    return euclidean(point, closest)


# ====================================================================
# LLM COMMAND PARSER
# ====================================================================

class LLMCommandParser:
    def __init__(self, target_map):
        self.target_map = target_map

    def parse(self, user_input):
        prompt = (
            "You are a routing dispatcher for an autonomous mobile robot.\n"
            "Your job is only to convert natural language route commands into ordered target IDs.\n\n"
            "Available targets:\n"
            "- TARGET1: Purple / Ungu\n"
            "- TARGET2: Cyan / Light Blue\n"
            "- TARGET3: Green / Hijau\n"
            "- TARGET4: Red / Merah\n\n"
            "Rules:\n"
            "1. If user says 'via', 'through', 'pass', 'lewat', or 'mampir', the mentioned waypoint must appear before final destination.\n"
            "2. Return only comma-separated target IDs.\n"
            "3. Do not explain anything.\n"
            "4. Valid outputs example: TARGET4,TARGET2\n\n"
            f"User command: {user_input}"
        )

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 50
        }

        try:
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=8)
            response.raise_for_status()

            reply = response.json()["choices"][0]["message"]["content"]
            reply = reply.strip().upper().replace(" ", "")

            parsed = [
                token.strip()
                for token in reply.split(",")
                if token.strip() in self.target_map
            ]

            return parsed

        except Exception as e:
            print(f"[LLM ERROR] {e}")
            print("[FALLBACK] Trying simple keyword parser...")
            return self.fallback_parse(user_input)

    def fallback_parse(self, user_input):
        text = user_input.lower()
        result = []

        aliases = {
            "TARGET1": ["target1", "target 1", "ungu", "purple", "satu", "1"],
            "TARGET2": ["target2", "target 2", "cyan", "light blue", "biru muda", "dua", "2"],
            "TARGET3": ["target3", "target 3", "hijau", "green", "tiga", "3"],
            "TARGET4": ["target4", "target 4", "merah", "red", "empat", "4"],
        }

        for key, words in aliases.items():
            for word in words:
                if word in text:
                    result.append(key)
                    break

        unique = []
        for item in result:
            if item not in unique:
                unique.append(item)

        return unique


# ====================================================================
# SENSOR PROCESSOR
# ====================================================================

class SensorProcessor:
    def __init__(self, sim, sensor_handles):
        self.sim = sim
        self.sensor_handles = sensor_handles
        self.filtered = [MAX_SENSOR_RANGE for _ in range(8)]

        self.sensor_angles = [
            math.radians(90),
            math.radians(50),
            math.radians(30),
            math.radians(10),
            math.radians(-10),
            math.radians(-30),
            math.radians(-50),
            math.radians(-90),
        ]

    def read(self):
        raw = []
        for idx in range(8):
            res, dist, _, _, _ = self.sim.readProximitySensor(self.sensor_handles[idx])

            if res > 0 and MIN_VALID_SENSOR_RANGE <= dist <= MAX_SENSOR_RANGE:
                value = dist
            else:
                value = MAX_SENSOR_RANGE

            self.filtered[idx] = (
                SENSOR_EMA_ALPHA * value
                + (1.0 - SENSOR_EMA_ALPHA) * self.filtered[idx]
            )
            raw.append(value)

        return raw, list(self.filtered)

    def get_regions(self, readings):
        front = min(readings[2], readings[3], readings[4], readings[5])
        left = min(readings[0], readings[1], readings[2])
        right = min(readings[5], readings[6], readings[7])

        left_avg = np.mean(readings[0:3])
        right_avg = np.mean(readings[5:8])

        return {
            "front": front,
            "left": left,
            "right": right,
            "left_avg": left_avg,
            "right_avg": right_avg
        }


# ====================================================================
# OCCUPANCY GRID MAP (UPDATED WITH RAY-CLEARING ERASER)
# ====================================================================

class OccupancyGrid:
    def __init__(self):
        self.resolution = GRID_RESOLUTION
        self.x_min = MAP_X_MIN
        self.x_max = MAP_X_MAX
        self.y_min = MAP_Y_MIN
        self.y_max = MAP_Y_MAX

        self.width = int((self.x_max - self.x_min) / self.resolution)
        self.height = int((self.y_max - self.y_min) / self.resolution)

        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.inflation_cells = max(1, int(OBSTACLE_INFLATION_RADIUS / self.resolution))

    def world_to_grid(self, x, y):
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        return clamp(gx, 0, self.width - 1), clamp(gy, 0, self.height - 1)

    def grid_to_world(self, gx, gy):
        x = self.x_min + gx * self.resolution + self.resolution / 2.0
        y = self.y_min + gy * self.resolution + self.resolution / 2.0
        return x, y

    def is_inside(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_occupied(self, gx, gy):
        if not self.is_inside(gx, gy):
            return True
        return self.grid[gy, gx] > 0

    def free_line(self, x0, y0, x1, y1):
        """
        FIXED: Menghapus rintangan palsu sepanjang garis sensor (Ray-Clearing)
        Menyetel semua sel yang dilewati oleh berkas sinar sensor kembali menjadi KOSONG (0).
        """
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(dist / (self.resolution * 0.5)))

        for k in range(steps):
            t = k / steps
            lx = x0 + t * (x1 - x0)
            ly = y0 + t * (y1 - y0)
            gx, gy = self.world_to_grid(lx, ly)
            if self.is_inside(gx, gy):
                self.grid[gy, gx] = 0  # Hapus rintangan hantu

    def mark_obstacle(self, x, y):
        gx, gy = self.world_to_grid(x, y)

        for dy in range(-self.inflation_cells, self.inflation_cells + 1):
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                nx = gx + dx
                ny = gy + dy

                if not self.is_inside(nx, ny):
                    continue

                if math.hypot(dx, dy) <= self.inflation_cells:
                    self.grid[ny, nx] = 1

    def clear_near_robot(self, x, y, radius=0.25):
        gx, gy = self.world_to_grid(x, y)
        r_cells = int(radius / self.resolution)

        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx = gx + dx
                ny = gy + dy
                if self.is_inside(nx, ny):
                    self.grid[ny, nx] = 0


# ====================================================================
# A* GLOBAL PLANNER
# ====================================================================

class GlobalPlanner:
    def __init__(self, occ_grid):
        self.map = occ_grid

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def neighbors(self, node):
        x, y = node
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

        for dx, dy, cost in moves:
            nx = x + dx
            ny = y + dy
            if self.map.is_inside(nx, ny) and not self.map.is_occupied(nx, ny):
                yield (nx, ny), cost

    def find_nearest_free(self, cell, max_radius=10):
        if not self.map.is_occupied(cell[0], cell[1]):
            return cell

        cx, cy = cell
        for r in range(1, max_radius + 1):
            candidates = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if self.map.is_inside(nx, ny) and not self.map.is_occupied(nx, ny):
                        candidates.append((nx, ny))

            if candidates:
                candidates.sort(key=lambda p: self.heuristic(p, cell))
                return candidates[0]
        return cell

    def plan(self, start_world, goal_world):
        start = self.map.world_to_grid(start_world[0], start_world[1])
        goal = self.map.world_to_grid(goal_world[0], goal_world[1])

        start = self.find_nearest_free(start)
        goal = self.find_nearest_free(goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed: continue
            closed.add(current)

            if self.heuristic(current, goal) <= A_STAR_GOAL_TOL_CELLS:
                return self.reconstruct_path(came_from, current)

            for nxt, move_cost in self.neighbors(current):
                tentative = g_score[current] + move_cost
                if nxt not in g_score or tentative < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f = tentative + self.heuristic(nxt, goal)
                    heapq.heappush(open_set, (f, nxt))
        return []

    def reconstruct_path(self, came_from, current):
        path_cells = [current]
        while current in came_from:
            current = came_from[current]
            path_cells.append(current)

        path_cells.reverse()
        path_world = [self.map.grid_to_world(cell[0], cell[1]) for cell in path_cells]
        return self.smooth_path(path_world)

    def smooth_path(self, path):
        if len(path) <= 2: return path
        smooth = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_is_free(path[i], path[j]):
                    break
                j -= 1
            smooth.append(path[j])
            i = j
        return smooth

    def line_is_free(self, a, b):
        dist = euclidean(a, b)
        steps = max(2, int(dist / (self.map.resolution * 0.5)))

        for k in range(steps + 1):
            t = k / steps
            gx, gy = self.map.world_to_grid(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if self.map.is_occupied(gx, gy):
                return False
        return True


# ====================================================================
# LOCAL PLANNER: PURE PURSUIT + REACTIVE AVOIDANCE
# ====================================================================

class LocalPlanner:
    def __init__(self):
        self.last_waypoint_idx = 0

    def reset(self):
        self.last_waypoint_idx = 0

    def select_lookahead_point(self, robot_pos, path, current_speed):
        if not path: return None
        lookahead_dist = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * abs(current_speed)

        while self.last_waypoint_idx < len(path) - 1:
            if euclidean(robot_pos, path[self.last_waypoint_idx]) < lookahead_dist * 0.7:
                self.last_waypoint_idx += 1
            else:
                break

        for i in range(self.last_waypoint_idx, len(path)):
            if euclidean(robot_pos, path[i]) >= lookahead_dist:
                self.last_waypoint_idx = i
                return path[i]
        return path[-1]

    def compute_command(self, robot_pose, path, sensor_regions, current_speed, dist_to_goal):
        rx, ry, theta = robot_pose
        lookahead = self.select_lookahead_point((rx, ry), path, current_speed)
        if lookahead is None: return 0.0, 0.0

        heading_error = normalize_angle(math.atan2(lookahead[1] - ry, lookahead[0] - rx) - theta)
        front = sensor_regions["front"]
        angular = HEADING_GAIN * heading_error

        goal_factor = clamp(dist_to_goal / 0.8, 0.20, 1.0)
        turn_factor = clamp(1.0 - abs(heading_error) / math.radians(90), 0.25, 1.0)
        obstacle_factor = clamp((front - EMERGENCY_DIST) / (OBSTACLE_AVOID_DIST - EMERGENCY_DIST), 0.15, 1.0)

        linear = MAX_LINEAR_VEL * goal_factor * turn_factor * obstacle_factor

        if front < OBSTACLE_AVOID_DIST:
            avoid_strength = clamp((OBSTACLE_AVOID_DIST - front) / OBSTACLE_AVOID_DIST, 0.0, 1.0)
            avoid_turn = 1.2 * avoid_strength if sensor_regions["left_avg"] > sensor_regions["right_avg"] else -1.2 * avoid_strength
            angular += avoid_turn
            linear *= clamp(1.0 - 0.65 * avoid_strength, 0.12, 1.0)

        return clamp(linear, 0.0, MAX_LINEAR_VEL), clamp(angular, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)


# ====================================================================
# ROBOT CONTROLLER
# ====================================================================

class RobotController:
    def __init__(self, sim):
        self.sim = sim
        self.p3dx = sim.getObject("/PioneerP3DX")
        self.right_motor = sim.getObject("/PioneerP3DX/rightMotor")
        self.left_motor = sim.getObject("/PioneerP3DX/leftMotor")
        self.current_v = 0.0
        self.current_w = 0.0

    def pose(self):
        pos = self.sim.getObjectPosition(self.p3dx, self.sim.handle_world)
        ori = self.sim.getObjectOrientation(self.p3dx, self.sim.handle_world)
        return pos[0], pos[1], ori[2]

    def stop(self):
        self.sim.setJointTargetVelocity(self.right_motor, 0.0)
        self.sim.setJointTargetVelocity(self.left_motor, 0.0)
        self.current_v = 0.0
        self.current_w = 0.0

    def apply_slew(self, target_v, target_w):
        self.current_v += clamp(target_v - self.current_v, -MAX_ACCEL, MAX_ACCEL)
        self.current_w += clamp(target_w - self.current_w, -MAX_ALPHA, MAX_ALPHA)
        return self.current_v, self.current_w

    def drive(self, target_v, target_w):
        v, w = self.apply_slew(target_v, target_w)
        wr = (v + HALF_WHEEL_BASE * w) / WHEEL_RADIUS
        wl = (v - HALF_WHEEL_BASE * w) / WHEEL_RADIUS
        self.sim.setJointTargetVelocity(self.right_motor, clamp(wr, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        self.sim.setJointTargetVelocity(self.left_motor, clamp(wl, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

    def reverse_and_rotate(self, direction=1):
        for _ in range(RECOVERY_REVERSE_STEPS):
            self.drive(-0.12, 0.0)
            self.sim.step()
            time.sleep(0.01)
        for _ in range(RECOVERY_ROTATE_STEPS):
            self.drive(0.0, direction * 1.35)
            self.sim.step()
            time.sleep(0.01)
        self.stop()


# ====================================================================
# MAIN FSM NAVIGATION SYSTEM WITH FIXED DYNAMIC MAPPING
# ====================================================================

class NavigationSystem:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.sim.setStepping(True)
        self.robot = RobotController(self.sim)

        self.target_map = {
            "TARGET1": {"name": "Target1 (Ungu / Purple)", "handle": self.sim.getObject("/Target1")},
            "TARGET2": {"name": "Target2 (Cyan / Light Blue)", "handle": self.sim.getObject("/Target2")},
            "TARGET3": {"name": "Target3 (Hijau / Green)", "handle": self.sim.getObject("/Target3")},
            "TARGET4": {"name": "Target4 (Merah / Red)", "handle": self.sim.getObject("/Target4")},
        }

        self.sensor_handles = {idx: self.sim.getObject(f"/PioneerP3DX/ultrasonicSensor[{idx}]") for idx in range(8)}
        self.sensor_processor = SensorProcessor(self.sim, self.sensor_handles)
        self.occ_grid = OccupancyGrid()
        self.global_planner = GlobalPlanner(self.occ_grid)
        self.local_planner = LocalPlanner()
        self.llm_parser = LLMCommandParser(self.target_map)

        self.state = "WAITING_FOR_COMMAND"
        self.target_queue = []
        self.active_target_key = None
        self.current_path = []
        self.last_replan_time = 0.0
        self.position_history = deque()
        self.recovery_count = 0
        self.loop_counter = 0  

        # --- FIG INDEPENDEN GRAPHICS LIVE WINDOW ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlabel("World X (meters)")
        self.ax.set_ylabel("World Y (meters)")

    def target_position(self, target_key):
        pos = self.sim.getObjectPosition(self.target_map[target_key]["handle"], self.sim.handle_world)
        return pos[0], pos[1]

    def draw_live_map(self, robot_pos):
        self.ax.clear()
        self.ax.set_title(f"EAS Live Map | State: {self.state}")
        self.ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
        self.ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # Menampilkan bodi grid peta (daerah belum dijelajahi akan otomatis tetap putih bersih)
        self.ax.imshow(
            self.occ_grid.grid, cmap="gray_r", origin="lower",
            extent=[MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX], alpha=0.7
        )

        if len(self.current_path) > 0:
            wps = np.array(self.current_path)
            self.ax.plot(wps[:, 0], wps[:, 1], color="cyan", linewidth=2, label="A* Path", zorder=3)

        for key, info in self.target_map.items():
            tx, ty = self.target_position(key)
            color = "purple" if key=="TARGET1" else "deepskyblue" if key=="TARGET2" else "green" if key=="TARGET3" else "red"
            self.ax.scatter(tx, ty, color=color, s=120, edgecolors='black', marker="X", zorder=5)

        self.ax.scatter(robot_pos[0], robot_pos[1], color="gold", s=150, edgecolors="black", marker="o", zorder=6)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def wait_for_command(self):
        self.robot.stop()
        self.sim.setStepping(False)

        print("\n-------------------------------------------------------")
        user_input = input("Enter multi-stop route sequence: ")
        parsed = self.llm_parser.parse(user_input)

        if not parsed:
            print("[FAIL] No valid target parsed.")
            self.sim.setStepping(True)
            return

        self.target_queue = parsed
        self.active_target_key = self.target_queue.pop(0)

        print(f"[SUCCESS] Route Manifest Locked: {parsed}")
        self.current_path = []
        self.last_replan_time = 0.0
        self.local_planner.reset()
        self.position_history.clear()
        self.recovery_count = 0
        self.sim.setStepping(True)
        self.state = "INITIAL_PLANNING"

    def plan_to_active_target(self):
        rx, ry, _ = self.robot.pose()
        target = self.target_position(self.active_target_key)
        self.occ_grid.clear_near_robot(rx, ry)
        path = self.global_planner.plan((rx, ry), target)
        if not path:
            path = [(rx, ry), target]
        self.current_path = path
        self.local_planner.reset()
        self.last_replan_time = time.time()
        self.position_history.clear()
        self.state = "FOLLOW_GLOBAL_PATH"

    def should_replan(self, robot_pos, front_clearance):
        now = time.time()
        if now - self.last_replan_time < REPLAN_INTERVAL: return False
        if front_clearance < OBSTACLE_AVOID_DIST * 0.8: return True
        if len(self.current_path) >= 2:
            min_dist_to_path = min(world_distance_to_segment(robot_pos, self.current_path[i], self.current_path[i + 1]) for i in range(len(self.current_path) - 1))
            if min_dist_to_path > PATH_DEVIATION_REPLAN_DIST: return True
        return False

    def update_stall_detector(self, rx, ry):
        now = time.time()
        self.position_history.append((rx, ry, now))
        while self.position_history and now - self.position_history[0][2] > STALL_WINDOW_SEC:
            self.position_history.popleft()
        if len(self.position_history) < 10: return False
        return math.hypot(self.position_history[-1][0] - self.position_history[0][0], self.position_history[-1][1] - self.position_history[0][1]) < STALL_MIN_DISPLACEMENT

    def handle_reached_target(self):
        print(f"\n[ARRIVED] Reached {self.target_map[self.active_target_key]['name']}")
        self.robot.stop()
        if self.target_queue:
            self.active_target_key = self.target_queue.pop(0)
            self.current_path = []
            self.last_replan_time = 0.0
            self.local_planner.reset()
            self.position_history.clear()
            self.state = "INITIAL_PLANNING"
        else:
            print("[MISSION COMPLETE] All targets reached.")
            self.state = "WAITING_FOR_COMMAND"

    def emergency_control(self, sensor_regions):
        if sensor_regions["front"] < EMERGENCY_DIST:
            self.robot.drive(-0.04, 1.4 if sensor_regions["left_avg"] > sensor_regions["right_avg"] else -1.4)
            return True
        return False

    def follow_path(self):
        rx, ry, theta = self.robot.pose()
        target = self.target_position(self.active_target_key)
        dist_to_goal = euclidean((rx, ry), target)

        raw_sensor, filtered_sensor = self.sensor_processor.read()
        sensor_regions = self.sensor_processor.get_regions(filtered_sensor)

        # ====================================================================
        # FIXED LOGIC: INVERSE SENSOR MODEL RAY-CLEARING UPDATE
        # ====================================================================
        for i, dist in enumerate(filtered_sensor):
            global_angle = theta + self.sensor_processor.sensor_angles[i]
            
            # Hitung koordinat terjauh dari berkas sinar sensor ultrasonik saat ini
            ox = rx + dist * math.cos(global_angle)
            oy = ry + dist * math.sin(global_angle)
            
            # 1. Bersihkan sel di sepanjang lintasan sinar (Hapus rintangan hantu jika area kosong)
            self.occ_grid.free_line(rx, ry, ox, oy)
            
            # 2. Gambar rintangan HANYA jika sensor mendeteksi objek nyata di bawah jarak maksimum
            if dist < MAX_SENSOR_RANGE * 0.98:
                self.occ_grid.mark_obstacle(ox, oy)
                
        self.occ_grid.clear_near_robot(rx, ry)

        if dist_to_goal < TARGET_REACHED_DIST:
            self.handle_reached_target()
            return

        if self.emergency_control(sensor_regions):
            self.sim.step()
            time.sleep(CONTROL_DT)
            return

        if self.update_stall_detector(rx, ry):
            self.state = "RECOVERY"
            return

        if self.should_replan((rx, ry), sensor_regions["front"]):
            self.state = "INITIAL_PLANNING"
            return

        target_v, target_w = self.local_planner.compute_command(
            robot_pose=(rx, ry, theta), path=self.current_path, sensor_regions=sensor_regions,
            current_speed=self.robot.current_v, dist_to_goal=dist_to_goal
        )
        self.robot.drive(target_v, target_w)

        # Throttle update peta setiap 5 perulangan agar tidak mengganggu kecepatan simulasi
        self.loop_counter += 1
        if self.loop_counter % 5 == 0:
            self.draw_live_map((rx, ry))

        print(f"State: {self.state:18} | Goal: {self.active_target_key:7} | Dist: {dist_to_goal:5.2f} m | Front: {sensor_regions['front']:4.2f} m | Queue: {len(self.target_queue)}", end="\r")
        self.sim.step()
        time.sleep(CONTROL_DT)

    def recovery(self):
        print("\n[RECOVERY] Stall detected. Executing escape maneuver...")
        _, filtered_sensor = self.sensor_processor.read()
        regions = self.sensor_processor.get_regions(filtered_sensor)
        self.robot.reverse_and_rotate(direction=(1 if regions["left_avg"] > regions["right_avg"] else -1))
        self.position_history.clear()
        self.recovery_count += 1
        self.state = "INITIAL_PLANNING"

    def run(self):
        print("=======================================================")
        print(" HYBRID A* + PURE PURSUIT NAVIGATION STACK ENGAGED     ")
        print(" AI used only for route parsing, not for control loop   ")
        print("=======================================================")
        self.sim.startSimulation()
        start_time = time.time()
        try:
            while time.time() - start_time < SIMULATION_DURATION:
                if self.state == "WAITING_FOR_COMMAND": self.wait_for_command()
                elif self.state == "INITIAL_PLANNING": self.plan_to_active_target()
                elif self.state == "FOLLOW_GLOBAL_PATH": self.follow_path()
                elif self.state == "RECOVERY": self.recovery()
        finally:
            self.robot.stop()
            self.sim.stopSimulation()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    nav = NavigationSystem()
    nav.run()