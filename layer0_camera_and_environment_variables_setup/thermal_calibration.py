import logging

class ThermalCalibrationManager:
  def __init__(self,conn):
    self.conn= conn

  def set_environment(self,emissivity:float,distance:float,ambient_temp:float,humidity:float,reflected_temp:float):
    logging.info("Setting thermal environment parameters")
    payload =(int(emissivity*100).to_bytes(2,"little")+ int(distance*100).to_bytes(2,"little") + int(ambient_temp*100).to_bytes(2,"little") + int(humidity*100).to_bytes(2,"little") + int(reflected_temp*100).to_bytes(2,"little"))

    self.conn.send_command(0x3A, payload)

  def enable_correction(self, enable:bool = True):
    logging.info("Enabling Thermal correction")
    payload = b"\x01" if enable else b"\x00"
    self.conn.send_command(0x3C, payload)

    def  verify (self):
      logging.info("Verifying Thermal correction parameters")
      self.conn.send_command(0x39, b"")