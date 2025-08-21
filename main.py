from djitellopy import Tello
import threading
import pygame
import numpy as np
import queue
import time
from google import genai
from google.genai import types
import os
import cv2

"""
Controls:
  Left Stick: Up/Down = Throttle, Left/Right = Yaw
  Right Stick: Up/Down = Pitch (Forward/Back), Left/Right = Roll
  Y Button = Takeoff, B Button = Land, Select = Emergency Stop
"""

DRONE_SPEED = 100
DISPLAY_FPS = 30
WINDOW_SIZE = (960, 720)
STICK_DEADZONE = 0.1
MOVEMENT_DISTANCE = 50


class TelloDroneController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        pygame.display.set_caption("Tello Drone Controller")
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(
            WINDOW_SIZE, pygame.RESIZABLE | pygame.DOUBLEBUF
        )

        self.drone = Tello()
        self.frame_reader = None

        self.forward_back_input = 0
        self.left_right_input = 0
        self.up_down_input = 0
        self.yaw_input = 0

        self.controller = None

        self.telemetry_lock = threading.Lock()
        self.telemetry_cache = {
            "battery": 0,
            "height": 0,
            "temperature": 0,
            "flight_time": 0,
            "total_speed": 0,
        }

        self.emergency_triggered = False
        self.blocking_command_queue = queue.Queue(
            1
        )  # Limit to 1 command when pressing the button

        self.blocking_command_thread = None
        self.controller_connection_thread = None
        self.telemetry_thread = None
        self.drone_connection_thread = None
        self.autopilot_worker_thread = None

        self.stop_threads = threading.Event()
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // DISPLAY_FPS)

        self.control_update_counter = 0

    def apply_deadzone(self, value, deadzone=STICK_DEADZONE):
        """Apply deadzone to controller input."""
        if abs(value) < deadzone:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def map_stick_to_input(self, stick_value):
        """Map stick value (-1 to 1) to drone input (-100 to 100)."""
        deadzone_applied = self.apply_deadzone(stick_value)
        return int(deadzone_applied * 100)

    def update_controller_input(self):
        """Update drone velocities based on controller input (Mode 2)."""
        if not self.controller:
            return
        try:
            throttle_raw = -self.controller.get_axis(1)
            self.up_down_input = self.map_stick_to_input(throttle_raw)

            yaw_raw = self.controller.get_axis(0)
            self.yaw_input = self.map_stick_to_input(yaw_raw)

            pitch_raw = -self.controller.get_axis(4)
            self.forward_back_input = self.map_stick_to_input(pitch_raw)

            roll_raw = self.controller.get_axis(3)
            self.left_right_input = self.map_stick_to_input(roll_raw)
        except Exception as e:
            print(f"Controller input error: {e}")

    def handle_controller_buttons(self):
        """Handle controller button presses."""
        if not self.controller:
            return
        try:
            if self.controller.get_button(5):
                self.queue_command(self.drone.takeoff)

            elif self.controller.get_button(4):
                self.queue_command(self.drone.land)

            elif self.controller.get_hat(0) == (0, 1):
                self.queue_command(self.drone.move_forward(MOVEMENT_DISTANCE))

            elif self.controller.get_hat(0) == (0, -1):
                self.queue_command(self.drone.move_back(MOVEMENT_DISTANCE))

            elif self.controller.get_hat(0) == (-1, 0):
                self.queue_command(self.drone.move_left(MOVEMENT_DISTANCE))

            elif self.controller.get_hat(0) == (1, 0):
                self.queue_command(self.drone.move_right(MOVEMENT_DISTANCE))

            elif self.controller.get_button(2):
                self.autopilot_worker_thread = threading.Thread(
                    target=self.autopilot_worker, daemon=True
                )
                self.autopilot_worker_thread.start()

            elif self.controller.get_button(8) or self.controller.get_button(10):
                self.drone.emergency()
                try:
                    self.blocking_command_queue.get_nowait()  # Stop it trying to takeoff again if it was queued
                except queue.Empty:
                    pass
                self.emergency_triggered = True
                self.forward_back_input = 0
                self.left_right_input = 0
                self.up_down_input = 0
                self.yaw_input = 0
        except Exception as e:
            print(f"Controller button error: {e}")

    def run(self):
        self.start_background_threads()
        running = True
        while running:
            running = self.handle_events()
            self.render_frame()
            pygame.display.flip()
            self.clock.tick(DISPLAY_FPS)
        self.cleanup()

    def handle_events(self):
        """Process pygame events. Returns False if the app should exit."""
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT + 1:
                if self.controller:
                    self.update_controller_input()
                    self.handle_controller_buttons()

                # Only update drone controls every 3rd frame (10Hz if DISPLAY_FPS=30)
                self.control_update_counter += 1
                if self.control_update_counter >= 3:
                    self.update_drone_controls()
                    self.control_update_counter = 0
            elif event.type == pygame.QUIT:
                return False
        return True

    def render_frame(self):
        """Draw the current video frame and telemetry overlay."""
        self.screen.fill([0, 0, 0])
        if self.frame_reader:
            frame = self.frame_reader.frame
            frame_rgb = np.flipud(np.rot90(frame[:, :, :]))
            frame_surface = pygame.surfarray.make_surface(frame_rgb)

            # Get window size
            win_w, win_h = self.screen.get_size()
            # Get frame size
            frame_h, frame_w = frame.shape[:2]
            # Calculate scale to fit window while keeping aspect ratio
            scale = min(win_w / frame_w, win_h / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            # Scale the frame
            scaled_surface = pygame.transform.smoothscale(frame_surface, (new_w, new_h))
            # Center the frame
            x = (win_w - new_w) // 2
            y = (win_h - new_h) // 2
            self.screen.blit(scaled_surface, (x, y))

        # Want to see the overlay even if no frame is available
        self.draw_telemetry_overlay_pygame()

    def cleanup(self):
        """Cleanup threads and pygame resources."""
        self.stop_threads.set()
        for thread in [
            self.telemetry_thread,
            self.blocking_command_thread,
            self.controller_connection_thread,
            self.drone_connection_thread,
        ]:
            if thread and thread.is_alive():
                thread.join(timeout=1)
        pygame.quit()

    def start_background_threads(self):
        """Start telemetry, command, controller, and drone connection threads."""
        self.telemetry_thread = threading.Thread(
            target=self.telemetry_worker, daemon=True
        )
        self.blocking_command_thread = threading.Thread(
            target=self.blocking_command_worker, daemon=True
        )
        self.controller_connection_thread = threading.Thread(
            target=self.controller_connection_worker, daemon=True
        )
        self.drone_connection_thread = threading.Thread(
            target=self.drone_connection_worker, daemon=True
        )

        self.telemetry_thread.start()
        self.blocking_command_thread.start()
        self.controller_connection_thread.start()
        self.drone_connection_thread.start()

        print("Background threads started")

    def move(self):
        print("hi")

    def autopilot_worker(self):
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = "gemini-2.5-flash"
        tools = []
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        )
        if self.frame_reader:
            frame = self.frame_reader.frame
            frame_rgb = np.flipud(np.rot90(frame[:, :, :]))
            success, image_bytes = cv2.imencode(
                ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )
            if success:
                image_bytes = image_bytes.tobytes()

                print(
                    client.models.generate_content(
                        model=model,
                        contents=[
                            types.Part.from_text(
                                text="What can you see in this image? If you were a drone, how would you fly to the yellow bin? You can fly up down left right and yaw."
                            ),
                            types.Part.from_bytes(
                                data=image_bytes, mime_type="image/jpeg"
                            ),
                        ],
                        config=generate_content_config,
                    ).text
                )
        while (
            not self.emergency_triggered
            and not self.land_triggered
            and not self.takeoff_triggered
            and not self.stop_threads.is_set()
            and sum(
                [
                    self.forward_back_input,
                    self.left_right_input,
                    self.up_down_input,
                    self.yaw_input,
                ]
            )
            == 0
        ):
            try:

                print(
                    client.models.generate_content(
                        model=model, contents=contents, config=generate_content_config
                    )
                )
                # for chunk in client.models.generate_content_stream(
                #     model=model,
                #     contents=contents,
                #     config=generate_content_config,
                # ):
                #     print(
                #         chunk.text
                #         if chunk.function_calls is None
                #         else chunk.function_calls[0]
                #     )

                time.sleep(1)
            except Exception as e:
                print(f"Autopilot error: {e}")
                break

    def drone_connection_worker(self):
        """Background thread to connect to the drone and set up video stream."""
        print("Drone connection thread started")
        connected = False
        while not self.stop_threads.is_set():
            try:
                if connected:
                    time.sleep(0.5)
                    continue
                self.drone.connect()
                self.drone.set_speed(DRONE_SPEED)
                self.drone.streamoff()
                self.drone.streamon()
                self.frame_reader = self.drone.get_frame_read()
                print("Drone connected")
                connected = True
            except Exception as e:
                print(f"Drone connection failed: {e}. Retrying...")

        if connected:
            try:
                self.drone.streamoff()
                self.drone.end()
            except Exception as e:
                print(f"Error during drone cleanup: {e}")

        print("Drone connection thread stopped")

    def controller_connection_worker(self):
        """Background thread to monitor controller connection status."""
        print("Controller monitor thread started")
        pygame.joystick.init()
        while not self.stop_threads.is_set():
            if not self.controller and pygame.joystick.get_count() > 0:
                try:
                    # Can have problems if immediately start taking inputs
                    time.sleep(0.5)
                    self.controller = pygame.joystick.Joystick(0)
                    self.controller.init()
                    print(f"Controller connected: {self.controller.get_name()}")
                except Exception as e:
                    print(f"Controller error: {e}")
            elif pygame.joystick.get_count() == 0:
                self.controller = None
            time.sleep(0.1)
        print("Controller monitor thread stopped")

    def telemetry_worker(self):
        """Background thread for continuous telemetry updates."""
        print("Telemetry thread started")
        while not self.stop_threads.is_set():
            try:
                battery = self.drone.get_battery()
                height = self.drone.get_height()
                temperature = self.drone.get_temperature()
                flight_time = self.drone.get_flight_time()

                speed_x = self.drone.get_speed_x()
                speed_y = self.drone.get_speed_y()
                speed_z = self.drone.get_speed_z()
                total_speed = (speed_x**2 + speed_y**2 + speed_z**2) ** 0.5

                with self.telemetry_lock:
                    self.telemetry_cache.update(
                        {
                            "battery": battery,
                            "height": height,
                            "temperature": temperature,
                            "flight_time": flight_time,
                            "total_speed": total_speed,
                        }
                    )
            except Exception as e:
                print(f"Telemetry update error: {e}")

            time.sleep(0.5)

        print("Telemetry thread stopped")

    def blocking_command_worker(self):
        """Background thread for processing blocking drone commands."""
        print("Blocking command thread started")
        while not self.stop_threads.is_set():
            try:
                command = self.blocking_command_queue.get(timeout=0.1)
                self.emergency_triggered = False
                command()
                self.blocking_command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Blocking command execution error: {e}")
        print("Blocking command thread stopped")

    def queue_command(self, command):
        """Queue a command for execution in the blocking command thread."""
        try:
            self.blocking_command_queue.get_nowait()  # Remove any pending command
        except queue.Empty:
            pass

        self.blocking_command_queue.put(command)

    def draw_telemetry_overlay_pygame(self):
        """Draw telemetry overlay using pygame fonts."""
        win_w, win_h = self.screen.get_size()  # Get current window size

        font_medium = pygame.font.Font("DejaVuSansMono-Bold.ttf", max(16, win_h // 36))
        font_status = pygame.font.Font("DejaVuSansMono-Bold.ttf", max(40, win_h // 12))

        with self.telemetry_lock:
            telemetry = self.telemetry_cache.copy()

        # Top row
        flight_time = telemetry["flight_time"]
        time_text = font_medium.render(
            f"{flight_time // 60:02d}:{flight_time % 60:02d}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(time_text, (10, 10))

        temp_text = font_medium.render(
            f"{telemetry['temperature']:.1f}°C", True, (255, 255, 255)
        )
        temp_rect = temp_text.get_rect(center=(win_w // 2, 25))
        self.screen.blit(temp_text, temp_rect)

        battery_level = telemetry["battery"]
        battery_color = (255, 0, 0) if battery_level < 30 else (255, 255, 255)
        battery_text = font_medium.render(f"{battery_level}%", True, battery_color)
        battery_rect = battery_text.get_rect(topright=(win_w - 10, 10))
        self.screen.blit(battery_text, battery_rect)

        # Bottom left
        line_spacing = int(win_h * 0.045)  # Increased spacing, scales with window
        y_offset = win_h - (line_spacing * 2)
        controller_text = (
            f"{'Connected' if self.controller else 'Controller Disconnected'}"
        )
        controller_color = (0, 255, 0) if self.controller else (255, 0, 0)
        controller_surface = font_medium.render(controller_text, True, controller_color)
        self.screen.blit(controller_surface, (10, y_offset))
        y_offset += line_spacing

        controls_text = f"P:{self.forward_back_input:+3} R:{self.left_right_input:+3} T:{self.up_down_input:+3} Y:{self.yaw_input:+3}"
        controls_surface = font_medium.render(controls_text, True, (255, 255, 255))
        self.screen.blit(controls_surface, (10, y_offset))

        # Bottom right
        y_offset_br = win_h - (line_spacing * 2)
        height_text = font_medium.render(
            f"{telemetry['height']/100} m", True, (255, 255, 255)
        )
        height_rect = height_text.get_rect(topright=(win_w - 10, y_offset_br))
        self.screen.blit(height_text, height_rect)
        y_offset_br += line_spacing

        speed_text = font_medium.render(
            f"{telemetry['total_speed']:.1f} cm/s",
            True,
            (255, 255, 255),
        )
        speed_rect = speed_text.get_rect(topright=(win_w - 10, y_offset_br))
        self.screen.blit(speed_text, speed_rect)

        # Center overlays
        if self.emergency_triggered:
            emergency_text = font_status.render("EMERGENCY STOP", True, (255, 0, 0))
            rect = emergency_text.get_rect(center=(win_w // 2, win_h // 2))
            self.screen.blit(emergency_text, rect)

    def update_drone_controls(self):
        """Send control commands to drone (called from main thread)."""
        self.drone.send_rc_control(
            self.left_right_input,
            self.forward_back_input,
            self.up_down_input,
            self.yaw_input,
        )


if __name__ == "__main__":
    controller = TelloDroneController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        controller.cleanup()
