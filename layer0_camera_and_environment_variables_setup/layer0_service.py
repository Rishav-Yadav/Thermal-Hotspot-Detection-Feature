import logging
import time

import yaml

from .frame_acquisition import ThermalFrameAcquirer
from .metadata import ThermalFrame
from .siyi_connections import SIYIConnection
from .storage import Layer0Storage
from .thermal_calibration import ThermalCalibrationManager


class Layer0Service:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        cam = self.config["camera"]
        thermal = self.config["thermal"]

        self.conn = SIYIConnection(cam["ip"], cam["port"])
        self.calib = ThermalCalibrationManager(self.conn)
        self.acquirer = ThermalFrameAcquirer(source="mock")
        self.storage = Layer0Storage("data/layer0")
        self.params = thermal

    def start(self):
        logging.basicConfig(level=logging.INFO)

        self.conn.connect()
        self.calib.set_environment(
            emissivity=self.params["emissivity"],
            distance=self.config["distance"]["initial"],
            ambient_temp=self.params["ambient_temp"],
            humidity=self.params["humidity"],
            reflected_temp=self.params["reflected_temp"],
        )
        self.calib.enable_correction(True)
        self.calib.verify()

    def next_frame(self) -> ThermalFrame:
        temp_matrix = self.acquirer.get_frame()
        frame = ThermalFrame(
            frame_id=self.acquirer.frame_id,
            timestamp=time.time(),
            temperature_matrix=temp_matrix,
            emissivity=self.params["emissivity"],
            distance=self.config["distance"]["initial"],
            ambient_temp=self.params["ambient_temp"],
            humidity=self.params["humidity"],
            correction_enabled=True,
        )
        self.storage.save(frame)
        return frame
