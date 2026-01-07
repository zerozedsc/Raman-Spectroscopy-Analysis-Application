__all__ = ["AndorSDK2"]

import asyncio
import numpy as np
from time import sleep
import os
import platform
import sys
import ctypes

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping, HasDependents
from typing import Dict, Any, List, Union
from . import atmcd


def finddllpath():
    if sys.platform == "linux":
        return "/usr/local/lib/libandor.so"
    elif sys.platform == "win32":
        if platform.machine() == "AMD64":
            dllname = "atmcd64d.dll"
        else:
            dllname = "atmcd32d.dll"
        for dirpath, dirname, filename in os.walk("C:\\"):
            if dllname in filename:
                return f"{dirpath}"
    else:
        return FileNotFoundError("cannot find appropriate library on this OS")


class AndorSDK2(HasMapping, HasMeasureTrigger, HasDependents, IsSensor, IsDaemon):
    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self.dllpath = self.finddllpath()
        self.sdk = atmcd.atmcd(userPath=self.dllpath)
        (ret) = self.sdk.Initialize("")
        sleep(
            1
        )  # could not figure why it gave a 20013 error once...I put an extra delay in here til I
        # can figure it out
        if ret != int(20002):
            (ret) = self.sdk.Initialize("")
            if ret != int(20002):
                self.logger.debug(f"init error {str(self.errorlookup(ret))}")
        self._busy = False
        self.timeout = 0.100  # may be overwritten by child

    def finddllpath(self):
        if sys.platform == "linux":
            return "/usr/local/lib/libandor.so"
        elif sys.platform == "win32":
            if platform.machine() == "AMD64":
                dllname = "atmcd64d.dll"
            else:
                dllname = "atmcd32d.dll"
            for dirpath, dirname, filename in os.walk("C:\\"):
                if dllname in filename:
                    return f"{dirpath}"
        else:
            return FileNotFoundError("cannot find appropriate library on this OS")

    async def _measure(self):
        # overrides _andor_sdk2
        timeout = self.timeout
        ret = self.sdk.StartAcquisition()
        if ret != 20002:
            self.logger.debug(f"_StartAcquisition error {str(self.errorlookup(ret))}")
        await asyncio.sleep(self.exposure_time)
        while self.busy():
            await asyncio.sleep(timeout / 10)
        ret = self._getacquireddata()
        if ret != 20002:
            self.logger.debug(f"_getacquireddata error {str(self.errorlookup(ret))}")
        pixels = np.reshape(self.buffer, self._channel_shapes["image"])
        self._gen_mappings()

        return {"image": pixels}

    def _gen_mappings(self):
        pass

    def _getacquireddata(self):
        imgsize = self.buffer_size
        carr = (ctypes.c_int * int(imgsize))()
        csize = ctypes.c_ulong(imgsize)
        ret = self.sdk.dll.GetAcquiredData(ctypes.byref(carr), csize)
        self.buffer = np.asarray(carr, dtype=int)
        return ret

    async def update_state(self):
        while True:
            (code, state) = self.sdk.GetStatus()
            if state == 20073:
                self._busy = False
            elif state == 20074:
                self._busy = False
            else:
                self.logger.debug(f"update_state error {str(self.errorlookup(state))}")
            await asyncio.sleep(0.10)

    def errorlookup(self, code):
        errordict = {
            "DRV_ERROR_CODES": 20001,
            "DRV_SUCCESS": 20002,
            "DRV_VXDNOTINSTALLED": 20003,
            "DRV_ERROR_SCAN": 20004,
            "DRV_ERROR_CHECK_SUM": 20005,
            "DRV_ERROR_FILELOAD": 20006,
            "DRV_UNKNOWN_FUNCTION": 20007,
            "DRV_ERROR_VXD_INIT": 20008,
            "DRV_ERROR_ADDRESS": 20009,
            "DRV_ERROR_PAGELOCK": 20010,
            "DRV_ERROR_PAGEUNLOCK": 20011,
            "DRV_ERROR_BOARDTEST": 20012,
            "DRV_ERROR_ACK": 20013,
            "DRV_ERROR_UP_FIFO": 20014,
            "DRV_ERROR_PATTERN": 20015,
            "DRV_ACQUISITION_ERRORS": 20017,
            "DRV_ACQ_BUFFER": 20018,
            "DRV_ACQ_DOWNFIFO_FULL": 20019,
            "DRV_PROC_UNKONWN_INSTRUCTION": 20020,
            "DRV_ILLEGAL_OP_CODE": 20021,
            "DRV_KINETIC_TIME_NOT_MET": 20022,
            "DRV_ACCUM_TIME_NOT_MET": 20023,
            "DRV_NO_NEW_DATA": 20024,
            "DRV_PCI_DMA_FAIL": 20025,
            "DRV_SPOOLERROR": 20026,
            "DRV_SPOOLSETUPERROR": 20027,
            "DRV_FILESIZELIMITERROR": 20028,
            "DRV_ERROR_FILESAVE": 20029,
            "DRV_TEMPERATURE_CODES": 20033,
            "DRV_TEMPERATURE_OFF": 20034,
            "DRV_TEMPERATURE_NOT_STABILIZED": 20035,
            "DRV_TEMPERATURE_STABILIZED": 20036,
            "DRV_TEMPERATURE_NOT_REACHED": 20037,
            "DRV_TEMPERATURE_OUT_RANGE": 20038,
            "DRV_TEMPERATURE_NOT_SUPPORTED": 20039,
            "DRV_TEMPERATURE_DRIFT": 20040,
            "DRV_TEMP_CODES": 20033,
            "DRV_TEMP_OFF": 20034,
            "DRV_TEMP_NOT_STABILIZED": 20035,
            "DRV_TEMP_STABILIZED": 20036,
            "DRV_TEMP_NOT_REACHED": 20037,
            "DRV_TEMP_OUT_RANGE": 20038,
            "DRV_TEMP_NOT_SUPPORTED": 20039,
            "DRV_TEMP_DRIFT": 20040,
            "DRV_GENERAL_ERRORS": 20049,
            "DRV_INVALID_AUX": 20050,
            "DRV_COF_NOTLOADED": 20051,
            "DRV_FPGAPROG": 20052,
            "DRV_FLEXERROR": 20053,
            "DRV_GPIBERROR": 20054,
            "DRV_EEPROMVERSIONERROR": 20055,
            "DRV_DATATYPE": 20064,
            "DRV_DRIVER_ERRORS": 20065,
            "DRV_P1INVALID": 20066,
            "DRV_P2INVALID": 20067,
            "DRV_P3INVALID": 20068,
            "DRV_P4INVALID": 20069,
            "DRV_INIERROR": 20070,
            "DRV_COFERROR": 20071,
            "DRV_ACQUIRING": 20072,
            "DRV_IDLE": 20073,
            "DRV_TEMPCYCLE": 20074,
            "DRV_NOT_INITIALIZED": 20075,
            "DRV_P5INVALID": 20076,
            "DRV_P6INVALID": 20077,
            "DRV_INVALID_MODE": 20078,
            "DRV_INVALID_FILTER": 20079,
            "DRV_I2CERRORS": 20080,
            "DRV_I2CDEVNOTFOUND": 20081,
            "DRV_I2CTIMEOUT": 20082,
            "DRV_P7INVALID": 20083,
            "DRV_P8INVALID": 20084,
            "DRV_P9INVALID": 20085,
            "DRV_P10INVALID": 20086,
            "DRV_P11INVALID": 20087,
            "DRV_USBERROR": 20089,
            "DRV_IOCERROR": 20090,
            "DRV_VRMVERSIONERROR": 20091,
            "DRV_GATESTEPERROR": 20092,
            "DRV_USB_INTERRUPT_ENDPOINT_ERROR": 20093,
            "DRV_RANDOM_TRACK_ERROR": 20094,
            "DRV_INVALID_TRIGGER_MODE": 20095,
            "DRV_LOAD_FIRMWARE_ERROR": 20096,
            "DRV_DIVIDE_BY_ZERO_ERROR": 20097,
            "DRV_INVALID_RINGEXPOSURES": 20098,
            "DRV_BINNING_ERROR": 20099,
            "DRV_INVALID_AMPLIFIER": 20100,
            "DRV_INVALID_COUNTCONVERT_MODE": 20101,
            "DRV_USB_INTERRUPT_ENDPOINT_TIMEOUT": 20102,
            "DRV_ERROR_NOCAMERA": 20990,
            "DRV_NOT_SUPPORTED": 20991,
            "DRV_NOT_AVAILABLE": 20992,
            "DRV_ERROR_MAP": 20115,
            "DRV_ERROR_UNMAP": 20116,
            "DRV_ERROR_MDL": 20117,
            "DRV_ERROR_UNMDL": 20118,
            "DRV_ERROR_BUFFSIZE": 20119,
            "DRV_ERROR_NOHANDLE": 20121,
            "DRV_GATING_NOT_AVAILABLE": 20130,
            "DRV_FPGA_VOLTAGE_ERROR": 20131,
            "DRV_OW_CMD_FAIL": 20150,
            "DRV_OWMEMORY_BAD_ADDR": 20151,
            "DRV_OWCMD_NOT_AVAILABLE": 20152,
            "DRV_OW_NO_SLAVES": 20153,
            "DRV_OW_NOT_INITIALIZED": 20154,
            "DRV_OW_ERROR_SLAVE_NUM": 20155,
            "DRV_MSTIMINGS_ERROR": 20156,
            "DRV_OA_NULL_ERROR": 20173,
            "DRV_OA_PARSE_DTD_ERROR": 20174,
            "DRV_OA_DTD_VALIDATE_ERROR": 20175,
            "DRV_OA_FILE_ACCESS_ERROR": 20176,
            "DRV_OA_FILE_DOES_NOT_EXIST": 20177,
            "DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR": 20178,
            "DRV_OA_PRESET_FILE_NOT_LOADED": 20179,
            "DRV_OA_USER_FILE_NOT_LOADED": 20180,
            "DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED": 20181,
            "DRV_OA_INVALID_FILE": 20182,
            "DRV_OA_FILE_HAS_BEEN_MODIFIED": 20183,
            "DRV_OA_BUFFER_FULL": 20184,
            "DRV_OA_INVALID_STRING_LENGTH": 20185,
            "DRV_OA_INVALID_CHARS_IN_NAME": 20186,
            "DRV_OA_INVALID_NAMING": 20187,
            "DRV_OA_GET_CAMERA_ERROR": 20188,
            "DRV_OA_MODE_ALREADY_EXISTS": 20189,
            "DRV_OA_STRINGS_NOT_EQUAL": 20190,
            "DRV_OA_NO_USER_DATA": 20191,
            "DRV_OA_VALUE_NOT_SUPPORTED": 20192,
            "DRV_OA_MODE_DOES_NOT_EXIST": 20193,
            "DRV_OA_CAMERA_NOT_SUPPORTED": 20194,
            "DRV_OA_FAILED_TO_GET_MODE": 20195,
            "DRV_OA_CAMERA_NOT_AVAILABLE": 20196,
            "DRV_PROCESSING_FAILED": 20211,
        }
        keyout = []
        code_to_str = {v: k for k, v in errordict.items()}
        return code_to_str[code]
