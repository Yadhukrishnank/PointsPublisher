import cv2
print(cv2.getBuildInformation())
# Look for "CUDA: YES" and the CUDA version
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())