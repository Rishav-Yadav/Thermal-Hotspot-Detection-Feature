from layer0_camera_and_environment_variables_setup.layer0_service import Layer0Service
if __name__=="__main__":
  layer0=Layer0Service("config/boiler_defaults.yaml")
  layer0.start()

  for _ in range(10):
    frame = layer0.next_frame()
    print(f"Frame {frame.frame_id} | " f"Max Temp :{frame.temperature_matrix.max():.2f} °C")