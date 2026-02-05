import socket 
import logging

class SIYIConnection:
  def __init__(self,ip: str,port: int):
    self.ip= ip
    self.port= port
    self.sock= None

  def connect(self):
    logging.info("Connecting to SIYI payload...")
    self.sock= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.settimeout(1.0)
    logging.info("SIYI connection ready")

    def send_command(self, cmd_id: int, payload: bytes):
      packet = bytes([cmd_id]+payload)
      self.sock.sendto(packet, (self.ip, self.port))
  
  def receive(self,bufsize=1024):
    try:
      data, _= self.sock.recvfrom(bufsize)
      return data
    except socket.timeout:
      return None
  def close(self):
    if self.sock:
      self.sock.close()
      self.sock=None