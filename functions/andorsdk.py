"""
Andor SDK3 Camera handler for Raman spectroscopy
This module provides a handler for the Andor SDK3 camera,
which is used for capturing Raman spectra.
It includes methods for initializing the camera,
acquiring frames, setting exposure, and closing the camera.
It also includes classes for controlling a laser,
a spectrometer, and a motorized stage.

Need to install andor3 sdk at https://andor.oxinst.com/downloads/
"""

from configs import *
import numpy as np

# try to import camera packages
try:
    from pylablib.devices import Andor
    import andor3

    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False


# AndorSDK3Camera is for Andor camera
class AndorSDK3Handler:
    """
    Andor SDK3 Camera handler.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cam = None
        self.sdk = None
        self.initialized = False
        self.platform = platform.system().lower()

    def initialize_camera(self):
        """
        Initialize the Andor SDK3 camera.
        Raises:
            EnvironmentError: If Andor SDK3 libraries are not available.
            RuntimeError: If camera initialization fails.
        """

        if not CAMERA_AVAILABLE:
            raise EnvironmentError(
                f"Andor SDK3 libraries not available on {self.platform}"
            )
        try:
            self.cam = Andor.AndorSDK3Camera(self.camera_index)
            self.cam.set_cooling(True)
            self.cam.set_exposure(0.01)
            self.cam.set_trigger_mode("software")
            self.sdk = andor3.AndorSDK3()
            self.sdk.open_camera(self.camera_index)
            self.initialized = True
            create_logs(
                "AndorSDK3Handler",
                "ANDORSDK",
                "Andor SDK3 camera initialized successfully.",
                status="info",
            )
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {e}")

    def acquire_frame(self):
        """
        Acquire a single frame from the camera.
        Raises:
            RuntimeError: If camera is not initialized or frame acquisition fails.
        """

        if not self.initialized:
            raise RuntimeError("Camera not initialized.")
        try:
            frame, metadata = self.cam.snap()
            return frame
        except Exception as e:
            raise RuntimeError(f"Failed to acquire frame: {e}")

    def close(self):
        """
        Close the camera and release resources.
        Raises:
            RuntimeError: If camera is not initialized or closing fails.
        """

        try:
            if self.cam:
                self.cam.close()
                self.cam = None
            if self.sdk:
                self.sdk.close_camera()
                self.sdk = None
            self.initialized = False
            create_logs(
                "AndorSDK3Handler",
                "ANDORSDK",
                "Andor SDK3 camera closed successfully.",
                status="info",
            )
        except Exception as e:
            create_logs(
                "AndorSDK3Handler",
                "ANDORSDK",
                f"Failed to close camera: {e}",
                status="error",
            )

    def set_exposure(self, exposure_time=0.01):
        """
        Set the camera exposure time.
        Args:
            exposure_time (float): Exposure time in seconds.
        Raises:
            RuntimeError: If camera is not initialized or setting exposure fails.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")
        self.cam.set_exposure(exposure_time)

    def trigger_software(self):
        if not self.initialized or self.sdk is None:
            raise RuntimeError("Camera not initialized.")
        self.sdk.send_software_trigger()


# Read ASC file function
def read_asc_file(filepath: str) -> np.ndarray:
    """
    Read an ASC file and return its data as a numpy array.
    Args:
        filepath (str): Path to the ASC file.
    Returns:
        np.ndarray: Data read from the ASC file.
    Raises:
        IOError: If file reading fails.
    """

    try:
        with open(filepath, "r") as f:
            data = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 1:
                    parts = line.strip().split(",")
                if len(parts) > 1:
                    data.append([float(x) for x in parts])
            return np.array(data)
    except Exception as e:
        raise IOError(f"Failed to read ASC: {e}")


class LaserController:
    """
    Laser control skeleton. In a real system, this might use serial commands
    to a USB/RS232 interface, for example via pyserial.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 9600):
        import serial

        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            create_logs(
                "LaserController",
                "ANDORSDK",
                f"Laser connected on {port}",
                status="info",
            )
        except Exception as e:
            self.serial = None
            create_logs(
                "LaserController",
                "ANDORSDK",
                f"Laser connection failed: {e}",
                status="error",
            )

    def laser_on(self):
        """
        Turn the laser on.
        Raises:
            RuntimeError: If serial connection is not established.
        """
        if self.serial:
            self.serial.write(b"ON\n")
            create_logs(
                "LaserController", "ANDORSDK", "Laser ON command sent.", status="info"
            )
        else:
            create_logs(
                "LaserController",
                "ANDORSDK",
                "Failed to send laser ON command: Serial connection not established.",
                status="error",
            )
            raise RuntimeError("Serial connection not established for laser control.")

    def laser_off(self):
        """
        Turn the laser off.
        Raises:
            RuntimeError: If serial connection is not established.
        """
        if self.serial:
            self.serial.write(b"OFF\n")
            create_logs(
                "LaserController", "ANDORSDK", "Laser OFF command sent.", status="info"
            )
        else:
            create_logs(
                "LaserController",
                "ANDORSDK",
                "Failed to send laser OFF command: Serial connection not established.",
                status="error",
            )
            raise RuntimeError("Serial connection not established for laser control.")

    def set_power(self, power_mW: float = 100):
        """
        Set the laser power in milliwatts.
        Args:
            power_mW (float): Power in milliwatts.
        Raises:
            RuntimeError: If serial connection is not established.
        """
        if self.serial:
            command = f"POWER {power_mW}\n".encode()
            self.serial.write(command)
            create_logs(
                "LaserController",
                "ANDORSDK",
                f"Laser power set to {power_mW} mW.",
                status="info",
            )
        else:
            create_logs(
                "LaserController",
                "ANDORSDK",
                "Failed to set laser power: Serial connection not established.",
                status="error",
            )
            raise RuntimeError("Serial connection not established for laser control.")


class SpectrometerController:
    """
    Spectrometer grating / wavelength control. Typically a USB-RS232 device.
    """

    def __init__(self, port="COM4", baudrate=9600):
        import serial

        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            create_logs(
                "SpectrometerController",
                "ANDORSDK",
                f"Spectrometer connected on {port}",
                status="info",
            )
        except Exception as e:
            self.serial = None
            create_logs(
                "SpectrometerController",
                "ANDORSDK",
                f"Spectrometer connection failed: {e}",
                status="error",
            )

    def set_wavelength(self, wavelength_nm):
        if self.serial:
            cmd = f"WAVELENGTH {wavelength_nm}\n".encode()
            self.serial.write(cmd)
            create_logs(
                "SpectrometerController",
                "ANDORSDK",
                f"Wavelength set to {wavelength_nm} nm.",
                status="info",
            )

    def get_wavelength(self):
        if self.serial:
            self.serial.write(b"GET_WAVELENGTH\n")
            result = self.serial.readline().decode().strip()
            create_logs(
                "SpectrometerController",
                "ANDORSDK",
                f"Current wavelength: {result}",
                status="info",
            )
            return result


class StageController:
    """
    Motorized x-y-z stage controller.
    Typically controlled over serial (RS232/USB).
    """

    def __init__(self, port="COM5", baudrate=9600):
        import serial

        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            create_logs(
                "StageController",
                "ANDORSDK",
                f"Stage connected on {port}",
                status="info",
            )
        except Exception as e:
            self.serial = None
            create_logs(
                "StageController",
                "ANDORSDK",
                f"Stage connection failed: {e}",
                status="error",
            )

    def move_to(self, x, y, z):
        if self.serial:
            cmd = f"MOVE {x} {y} {z}\n".encode()
            self.serial.write(cmd)
            create_logs(
                "StageController",
                "ANDORSDK",
                f"Stage moved to position ({x}, {y}, {z}).",
                status="info",
            )

    def home(self):
        if self.serial:
            self.serial.write(b"HOME\n")
            create_logs("StageController", "ANDORSDK", "Stage homed.", status="info")


class RamanMeasurementManager:
    """
    Raman measurement manager that coordinates
    laser, camera, spectrometer, stage
    """

    def __init__(self):
        self.camera = AndorSDK3Handler()
        self.laser = LaserController()
        self.spec = SpectrometerController()
        self.stage = StageController()

    def initialize_all(self):
        self.camera.initialize_camera()
        # assume the other devices are initialized on their __init__

    def single_point_measurement(self, x=0, y=0, z=0, exposure=0.05, wavelength=785):
        """
        Single point measurement procedure:
        1. move to location
        2. set wavelength
        3. turn laser on
        4. trigger camera
        5. turn laser off
        """
        self.stage.move_to(x, y, z)
        self.spec.set_wavelength(wavelength)
        self.camera.set_exposure(exposure)
        self.laser.laser_on()

        frame = self.camera.acquire_frame()
        self.laser.laser_off()
        create_logs(
            "RamanMeasurementManager",
            "ANDORSDK",
            f"Single point measurement at ({x}, {y}, {z}) with wavelength {wavelength} nm.",
            status="info",
        )
        return frame

    def shutdown(self):
        self.camera.close()
        self.laser.laser_off()
        create_logs(
            "RamanMeasurementManager",
            "ANDORSDK",
            "All devices shut down successfully.",
            status="info",
        )
