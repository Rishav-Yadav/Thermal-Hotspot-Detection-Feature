import os
import json 
import numpy as np

class Layer0Storage:
  def __init__(self, base_path :str):
    self.base_path = base_path
    os.makedirs(base_path, exist_ok=True)

  def save (self, thermal_frame):
    frame_id = thermal_frame.frame_id

    np.save(os.path.join(self.base_path, f"thermal_{frame_id}.npy"), thermal_frame.temperature_matrix)

    meta = thermal_frame.__dict__.copy()
    meta.pop("temperature_matrix")

    with open(
      os.path.join(self.base_path, f"thermal_{frame_id}.json"), 'w'
    ) as f:
      json.dump(meta,f, indent=2)