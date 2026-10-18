"""打印 YOLO12s 完整网络架构"""
from ultralytics import YOLO

model = YOLO("yolo12s.pt")
info = model.info(detailed=True, verbose=True)
print(info)
