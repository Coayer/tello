from djitellopy import Tello
import threading
import pygame
import numpy as np
import queue

DRONE_SPEED, DISPLAY_FPS = 100, 60
WINDOW_SIZE = (960, 720)
STICK_DEADZONE = 0.1


class TelloDroneController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        pygame.display.set_caption("Tello Drone Controller")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)

        self.drone = Tello()
        self.forward_back_velocity = self.left_right_velocity = (
            self.up_down_velocity
        ) = self.yaw_velocity = 0

        # Controller support
        self.controller = None
        self.controller_connected = False
        self.setup_controller()

        # Thread-safe telemetry data
        self.telemetry_lock = threading.Lock()
        self.telemetry_cache = {
            "battery": 0,
            "height": 0,
            "temp_avg": 0,
            "flight_time": 0,
            "total_speed": 0,
            "speed_x": 0,
            "speed_y": 0,
            "speed_z": 0,
        }

        self.emergency_triggered = self.takeoff_triggered = self.land_triggered = False

        # Command queue for thread communication
        self.command_queue = queue.Queue()

        # Thread control
        self.stop_threads = threading.Event()
        self.telemetry_thread = None
        self.command_thread = None

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // DISPLAY_FPS)

    def setup_controller(self):
        """Initialize and setup game controller"""
        for attempt in range(5):
            pygame.joystick.quit()
            pygame.joystick.init()

            if pygame.joystick.get_count() > 0:
                self.controller = pygame.joystick.Joystick(0)
                self.controller.init()
                self.controller_connected = True
                print(f"Controller connected: {self.controller.get_name()}")
                print(
                    f"Axes: {self.controller.get_numaxes()}, Buttons: {self.controller.get_numbuttons()}"
                )
                break
            else:
                print(f"Attempt {attempt + 1}: No controller detected, waiting...")
                pygame.time.wait(100)

        if not self.controller_connected:
            print("No controller detected. Controller required to operate drone.")

    def apply_deadzone(self, value, deadzone=STICK_DEADZONE):
        """Apply deadzone to controller input"""
        if abs(value) < deadzone:
            return 0.0
        # Scale the remaining range to 0-1
        sign = 1 if value > 0 else -1
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def map_stick_to_velocity(self, stick_value):
        """Map stick value (-1 to 1) to drone velocity (-100 to 100)"""
        deadzone_applied = self.apply_deadzone(stick_value)
        return int(deadzone_applied * 100)

    def update_controller_input(self):
        """Update drone velocities based on controller input (Mode 2)"""
        if not self.controller_connected:
            return

        try:
            # Mode 2 Configuration
            throttle_raw = -self.controller.get_axis(1)  # Invert Y axis
            self.up_down_velocity = self.map_stick_to_velocity(throttle_raw)

            yaw_raw = self.controller.get_axis(0)
            self.yaw_velocity = self.map_stick_to_velocity(yaw_raw)

            pitch_raw = -self.controller.get_axis(4)  # Invert Y axis
            self.forward_back_velocity = self.map_stick_to_velocity(pitch_raw)

            roll_raw = self.controller.get_axis(3)
            self.left_right_velocity = self.map_stick_to_velocity(roll_raw)

        except Exception as e:
            print(f"Controller input error: {e}")

    def handle_controller_buttons(self):
        """Handle controller button presses"""
        if not self.controller_connected:
            return

        try:
            if self.controller.get_button(2):
                self.queue_command("takeoff")
                self.takeoff_triggered = True

            if self.controller.get_button(1):
                self.queue_command("land")
                self.land_triggered = True

            if self.controller.get_button(8):
                self.drone.emergency()
                self.emergency_triggered = True
                self.forward_back_velocity = self.left_right_velocity = (
                    self.up_down_velocity
                ) = self.yaw_velocity = 0

        except Exception as e:
            print(f"Controller button error: {e}")

    def run(self):
        if not self.controller_connected:
            print(
                "ERROR: No controller detected. Controller is required to operate the drone."
            )
            return

        print("Connecting to drone...")
        self.drone.connect()
        self.drone.set_speed(DRONE_SPEED)
        self.drone.streamoff()
        self.drone.streamon()
        frame_reader = self.drone.get_frame_read()

        self.start_background_threads()

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    # Update controls from controller
                    if self.controller_connected:
                        self.update_controller_input()
                        self.handle_controller_buttons()

                    self.update_drone_controls()

                elif event.type == pygame.QUIT:
                    running = False

            if not running or frame_reader.stopped:
                break

            # UI rendering (main thread responsibility)
            self.screen.fill([0, 0, 0])
            frame = frame_reader.frame

            frame_rgb = frame[:, :, :]

            frame_rgb = np.rot90(frame_rgb)
            frame_rgb = np.flipud(frame_rgb)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)

            self.screen.blit(frame_surface, (0, 0))

            self.draw_telemetry_overlay_pygame()

            pygame.display.update()
            pygame.time.wait(1000 // DISPLAY_FPS)

        # Cleanup
        self.stop_threads.set()
        self.drone.streamoff()
        self.drone.end()

        # Wait for threads to finish
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            self.telemetry_thread.join(timeout=1)
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=1)

        pygame.quit()

    def start_background_threads(self):
        """Start telemetry and command processing threads"""
        self.telemetry_thread = threading.Thread(
            target=self.telemetry_worker, daemon=True
        )
        self.command_thread = threading.Thread(target=self.command_worker, daemon=True)

        self.telemetry_thread.start()
        self.command_thread.start()

        print("Background threads started")

    def telemetry_worker(self):
        """Background thread for continuous telemetry updates"""
        print("Telemetry thread started")
        while not self.stop_threads.is_set():
            try:
                # Collect telemetry data
                battery = self.drone.get_battery()
                height = self.drone.get_height()
                temp_avg = self.drone.get_temperature()
                flight_time = self.drone.get_flight_time()

                # Try to get speed data (may not always be available)
                try:
                    speed_x = self.drone.get_speed_x()
                    speed_y = self.drone.get_speed_y()
                    speed_z = self.drone.get_speed_z()
                    total_speed = (speed_x**2 + speed_y**2 + speed_z**2) ** 0.5
                except:
                    speed_x = speed_y = speed_z = total_speed = 0

                # Thread-safe update of telemetry cache
                with self.telemetry_lock:
                    self.telemetry_cache.update(
                        {
                            "battery": battery,
                            "height": height,
                            "temp_avg": temp_avg,
                            "flight_time": flight_time,
                            "total_speed": total_speed,
                            "speed_x": speed_x,
                            "speed_y": speed_y,
                            "speed_z": speed_z,
                        }
                    )

            except Exception as e:
                print(f"Telemetry update error: {e}")

            # Update every 0.5 seconds to reduce API calls
            if not self.stop_threads.wait(0.5):
                continue
            else:
                break

        print("Telemetry thread stopped")

    def command_worker(self):
        """Background thread for processing drone commands"""
        print("Command thread started")
        while not self.stop_threads.is_set():
            try:
                # Check for commands with timeout
                command = self.command_queue.get(timeout=0.1)

                self.emergency_triggered = False

                if command["type"] == "takeoff":
                    self._execute_takeoff()
                elif command["type"] == "land":
                    self._execute_land()

                self.command_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Command execution error: {e}")

        print("Command thread stopped")

    def _execute_takeoff(self):
        """Execute takeoff in command thread"""
        try:
            print("Taking off...")
            self.drone.takeoff()
            print("Takeoff completed!")
        except Exception as e:
            print(f"Takeoff failed: {e}")
        self.takeoff_triggered = False

    def _execute_land(self):
        """Execute landing in command thread"""
        try:
            print("Landing...")
            self.drone.land()
            print("Landing completed!")
        except Exception as e:
            print(f"Landing failed: {e}")
        self.land_triggered = False

    def queue_command(self, command_type, **kwargs):
        """Queue a command for execution in the command thread"""
        command = {"type": command_type, **kwargs}
        self.command_queue.put(command)

    def draw_telemetry_overlay_pygame(self):
        """Draw telemetry overlay using pygame fonts"""
        font_medium = pygame.font.Font("DejaVuSansMono-Bold.ttf", 20)
        font_status = pygame.font.Font("DejaVuSansMono-Bold.ttf", 60)

        # Thread-safe read of telemetry data
        with self.telemetry_lock:
            telemetry = self.telemetry_cache.copy()

        # --- TOP ROW ---
        # Flight time (top left)
        flight_time = telemetry["flight_time"]
        time_text = font_medium.render(
            f"{flight_time // 60:02d}:{flight_time % 60:02d}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(time_text, (10, 10))

        # Temperature (top center)
        temp_text = font_medium.render(
            f"{telemetry['temp_avg']:.1f}°C", True, (255, 255, 255)
        )
        temp_rect = temp_text.get_rect(center=(WINDOW_SIZE[0] // 2, 25))
        self.screen.blit(temp_text, temp_rect)

        # Battery (top right)
        battery_level = telemetry["battery"]
        battery_color = (255, 0, 0) if battery_level < 30 else (255, 255, 255)
        battery_text = font_medium.render(f"{battery_level}%", True, battery_color)
        battery_rect = battery_text.get_rect(topright=(WINDOW_SIZE[0] - 10, 10))
        self.screen.blit(battery_text, battery_rect)

        # --- BOTTOM LEFT ---
        y_offset = WINDOW_SIZE[1] - 80
        # Controller status
        controller_text = f"{'Connected' if self.controller_connected else 'Controller Not Connected'}"
        controller_color = (0, 255, 0) if self.controller_connected else (255, 0, 0)
        controller_surface = font_medium.render(controller_text, True, controller_color)
        self.screen.blit(controller_surface, (10, y_offset))
        y_offset += 30

        # Current control values
        controls_text = f"P:{self.forward_back_velocity:+3} R:{self.left_right_velocity:+3} T:{self.up_down_velocity:+3} Y:{self.yaw_velocity:+3}"
        controls_surface = font_medium.render(controls_text, True, (255, 255, 255))
        self.screen.blit(controls_surface, (10, y_offset))

        # --- BOTTOM RIGHT ---
        y_offset_br = WINDOW_SIZE[1] - 80
        # Height
        height_text = font_medium.render(
            f"{telemetry['height']}m", True, (255, 255, 255)
        )
        height_rect = height_text.get_rect(topright=(WINDOW_SIZE[0] - 10, y_offset_br))
        self.screen.blit(height_text, height_rect)
        y_offset_br += 30

        # Speed
        speed_text = font_medium.render(
            f"{telemetry['total_speed']:.1f} cm/s",
            True,
            (255, 255, 255),
        )
        speed_rect = speed_text.get_rect(topright=(WINDOW_SIZE[0] - 10, y_offset_br))
        self.screen.blit(speed_text, speed_rect)

        # --- CENTER OVERLAYS ---
        # Emergency status
        if self.emergency_triggered:
            emergency_text = font_status.render("EMERGENCY STOP", True, (255, 0, 0))
            rect = emergency_text.get_rect(
                center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 - 90)
            )
            self.screen.blit(emergency_text, rect)

        # Takeoff status
        if self.takeoff_triggered:
            takeoff_text = font_status.render("TAKEOFF", True, (255, 255, 255))
            rect = takeoff_text.get_rect(
                center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 - 30)
            )
            self.screen.blit(takeoff_text, rect)

        # Land status
        if self.land_triggered:
            land_text = font_status.render("LAND", True, (255, 255, 255))
            rect = land_text.get_rect(
                center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 + 90)
            )
            self.screen.blit(land_text, rect)

    def update_drone_controls(self):
        """Send control commands to drone (called from main thread)"""
        self.drone.send_rc_control(
            self.left_right_velocity,
            self.forward_back_velocity,
            self.up_down_velocity,
            self.yaw_velocity,
        )


def main():
    print("Tello Drone Controller - Controller Required")
    print("\n=== CONTROLLER CONTROLS (Mode 2) ===")
    print("Left Stick: Up/Down = Throttle, Left/Right = Yaw")
    print("Right Stick: Up/Down = Pitch (Forward/Back)")
    print("Right Stick: Left/Right = Roll")
    print("Y Button = Takeoff")
    print("B Button = Land")
    print("Select = Emergency Stop")
    print(f"Stick Deadzone: {STICK_DEADZONE*100}%")
    print(f"Drone Speed: {DRONE_SPEED} cm/s")

    TelloDroneController().run()


if __name__ == "__main__":
    main()
