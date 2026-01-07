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

from configs.configs import *
import numpy as np
import os
import platform
import ctypes
from ctypes import c_int, c_char_p, c_float, c_long, byref, POINTER
from ._enumsdk2 import *

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


# AndorSDK2 constants
DRV_SUCCESS = 20002
DRV_NOT_INITIALIZED = 20075
DRV_ACQUIRING = 20073
DRV_IDLE = 20074
DRV_TEMPERATURE_OFF = 20034
DRV_TEMPERATURE_STABILIZED = 20036
DRV_TEMPERATURE_NOT_REACHED = 20037
DRV_TEMPERATURE_DRIFT = 20040


class AndorSDK2Handler:
    """
    Andor SDK2 Camera handler for older cameras.
    This class provides an interface to Andor cameras using the V2 SDK (atmcd32d.dll or atmcd64d.dll).
    It handles camera initialization, data acquisition, and shutdown.
    """

    def __init__(self, driver_path=r"C:\helmi\研究\raman-app\drivers"):
        """
        Initializes the handler by loading the correct SDK2 DLL based on the OS architecture.
        Args:
            driver_path (str): The path to the directory containing the Andor SDK2 DLLs.
        Raises:
            EnvironmentError: If the OS is not Windows.
            FileNotFoundError: If the required DLL is not found in the specified path.
            OSError: If the DLL fails to load.
        """
        self.sdk = None
        self.initialized = False
        self.acquiring = False
        self.detector_width = 0
        self.detector_height = 0

        if platform.system() != "Windows":
            raise EnvironmentError("Andor SDK2 is only supported on Windows.")

        is_64bit = platform.machine().endswith("64")
        dll_name = "atmcd64d.dll" if is_64bit else "atmcd32d.dll"
        dll_path = os.path.join(driver_path, dll_name)

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"Andor SDK2 driver not found at {dll_path}")

        try:
            self.sdk = ctypes.WinDLL(dll_path)
            create_logs(
                "AndorSDK2Handler",
                "ANDORSDK2",
                f"Loaded Andor SDK2 driver: {dll_name}",
                status="info",
            )
        except OSError as e:
            create_logs(
                "AndorSDK2Handler",
                "ANDORSDK2",
                f"Failed to load Andor SDK2 driver: {e}",
                status="error",
            )
            raise OSError(f"Failed to load Andor SDK2 driver: {e}")

    def _check_error(self, error_code, function_name):
        """Helper function to check for SDK errors."""
        if error_code != DRV_SUCCESS:
            raise RuntimeError(
                f"Andor SDK2 Error in {function_name}: code {error_code}"
            )

    def initialize_camera(self, camera_index: int = 0):
        """
        Initialize the Andor SDK2 camera.
        Connects to the camera, gets detector info, and sets default modes.
        Args:
            camera_index (int): The index of the camera to use (0-based).
        """
        if self.initialized:
            create_logs(
                "AndorSDK2Handler",
                "ANDORSDK2",
                "Camera already initialized.",
                status="warning",
            )
            return

        ret = self.sdk.Initialize(None)
        self._check_error(ret, "Initialize")

        num_cameras = ctypes.c_long()
        self.sdk.GetAvailableCameras(ctypes.byref(num_cameras))
        if num_cameras.value <= camera_index:
            self.sdk.ShutDown()
            raise IndexError(
                f"Camera index {camera_index} out of range. Found {num_cameras.value} cameras."
            )

        # Get detector size
        width = ctypes.c_int()
        height = ctypes.c_int()
        self.sdk.GetDetector(ctypes.byref(width), ctypes.byref(height))
        self.detector_width = width.value
        self.detector_height = height.value

        # Set default modes for spectroscopy
        self.sdk.SetAcquisitionMode(1)  # 1: Single Scan
        self.sdk.SetReadMode(
            4
        )  # 4: Image (use 0 for Full Vertical Binning if applicable)
        self.sdk.SetTriggerMode(0)  # 0: Internal trigger

        self.initialized = True
        create_logs(
            "AndorSDK2Handler",
            "ANDORSDK2",
            f"Andor SDK2 camera initialized. Detector: {self.detector_width}x{self.detector_height}",
            status="info",
        )

    def shutdown(self):
        """
        Shutdown the camera and release resources.
        """
        if not self.initialized:
            return
        self.set_cooler(False)  # Turn off cooler before shutdown
        ret = self.sdk.ShutDown()
        self._check_error(ret, "ShutDown")
        self.initialized = False
        create_logs(
            "AndorSDK2Handler",
            "ANDORSDK2",
            "Andor SDK2 camera shut down.",
            status="info",
        )

    def set_exposure(self, exposure_time_s: float):
        """
        Set the camera exposure time.
        Args:
            exposure_time_s (float): Exposure time in seconds.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")
        ret = self.sdk.SetExposureTime(ctypes.c_float(exposure_time_s))
        self._check_error(ret, "SetExposureTime")

    def start_acquisition(self):
        """
        Start the acquisition process.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")
        if self.acquiring:
            create_logs(
                "AndorSDK2Handler", "ANDORSDK2", "Already acquiring.", status="warning"
            )
            return

        self.sdk.PrepareAcquisition()
        ret = self.sdk.StartAcquisition()
        self._check_error(ret, "StartAcquisition")
        self.acquiring = True

    def wait_for_acquisition(self, timeout_s=20):
        """
        Wait for the current acquisition to complete.
        Args:
            timeout_s (int): Maximum time to wait in seconds.
        """
        if not self.acquiring:
            return

        start_time = time.time()
        while time.time() - start_time < timeout_s:
            status = ctypes.c_int()
            self.sdk.GetStatus(ctypes.byref(status))
            if status.value == DRV_IDLE:
                self.acquiring = False
                return
            time.sleep(0.05)

        self.acquiring = False
        self.sdk.AbortAcquisition()
        raise TimeoutError(f"Timeout ({timeout_s}s) waiting for acquisition to finish.")

    def get_acquired_data(self):
        """
        Get the last acquired frame from the camera buffer.
        Returns:
            np.ndarray: The acquired frame as a numpy array.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")
        if self.acquiring:
            raise RuntimeError(
                "Cannot get data while acquiring. Call wait_for_acquisition() first."
            )

        size = self.detector_width * self.detector_height
        buffer = (ctypes.c_long * size)()
        ret = self.sdk.GetAcquiredData(ctypes.byref(buffer), size)
        self._check_error(ret, "GetAcquiredData")

        return np.array(buffer, dtype=np.int32).reshape(
            (self.detector_height, self.detector_width)
        )

    def acquire_frame(self, exposure_time_s=0.1):
        """
        A convenience method to set exposure, acquire a single frame, and return it.
        Args:
            exposure_time_s (float): Exposure time in seconds.
        Returns:
            np.ndarray: The acquired frame.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")

        self.set_exposure(exposure_time_s)
        self.start_acquisition()
        self.wait_for_acquisition(timeout_s=exposure_time_s + 10)
        return self.get_acquired_data()

    def set_cooler(self, enable, target_temp_c=-20):
        """
        Enable/disable the cooler and set the target temperature.
        Args:
            enable (bool): True to turn cooler on, False to turn off.
            target_temp_c (int): The target temperature in Celsius.
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")

        if enable:
            self.sdk.SetTemperature(ctypes.c_int(target_temp_c))
            self.sdk.CoolerON()
            create_logs(
                "AndorSDK2Handler",
                "ANDORSDK2",
                f"Cooler ON, target: {target_temp_c} C.",
                status="info",
            )
        else:
            self.sdk.CoolerOFF()
            create_logs("AndorSDK2Handler", "ANDORSDK2", "Cooler OFF.", status="info")

    def get_temperature(self):
        """
        Get the current detector temperature and status.
        Returns:
            tuple: A tuple containing (current_temperature, status_string).
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized.")

        temp = ctypes.c_int()
        ret = self.sdk.GetTemperature(ctypes.byref(temp))

        status_map = {
            DRV_SUCCESS: "Stabilized",
            DRV_TEMPERATURE_OFF: "Off",
            DRV_TEMPERATURE_STABILIZED: "Stabilized",
            DRV_TEMPERATURE_NOT_REACHED: "Cooling",
            DRV_TEMPERATURE_DRIFT: "Drift",
        }
        if ret in status_map:
            status_str = status_map[ret]
        else:
            self._check_error(ret, "GetTemperature")
            status_str = "Unknown"

        return temp.value, status_str


# --- Test Run ---
if __name__ == "__main__":
    camera = AndorSDK2Handler(driver_path="C:/helmi/研究/raman-app/drivers")
    camera.test_capture()
