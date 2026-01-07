# from functions.andorsdk.andorsdk import *

# if __name__ == "__main__":
#     camera = AndorSDK2Handler(driver_path="C:/helmi/研究/raman-app/drivers")
#     camera.initialize_camera()

import pickle

with open(r"J:\Coding\研究\raman-app\20221215 Mgus10 B.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
# print(data)
