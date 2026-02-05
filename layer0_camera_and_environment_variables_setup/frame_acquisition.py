import numpy as np
import time

class ThermalFrameAcquirer:
  def __init__(self, source="mock"):
    self.source = source
    self.frame_id = 0

  def get_frame(self) -> np.ndarray:
    self.frame_id +=1 

    if self.source == "mock":
      base = 80.0
      noise = np.random.normal(0,0.5,(512,640))
      hotspot = np.zeros ((512,640))
      hotspot[200:260, 300:360] = 40.0
      return base + noise +hotspot
    raise NotImplementedError("Real SIYI thermal acquisition not implemented")