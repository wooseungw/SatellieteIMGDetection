import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 직접 YOLO 클래스를 임포트
from ultralytics import YOLO

# YOLO 모델 초기화
model = YOLO("yolo11x-obb.pt")
results = model.train(
    data="data_combine.yaml",
    seed=44,
    epochs=20,
    batch=3,
    imgsz=1024,
    patience=15,
    device="0",
    optimizer = "AdamW",
    lr0= 0.00088,
    lrf= 0.01134,
    momentum= 0.82858,
    weight_decay= 0.0006,
    warmup_epochs= 2.33952,
    warmup_momentum= 0.58988,
    box= 5.06927,
    cls= 0.56241,
    dfl= 1.36767,
    hsv_h= 0.01707,
    hsv_s= 0.68688,
    hsv_v= 0.28102,
    degrees= 0.0,
    translate= 0.11263,
    shear= 0.0,
    perspective= 0.0,
    flipud= 0.45936,
    fliplr= 0.46892,
    bgr= 0.1,
    mosaic= 0.828,
    mixup= 0.0,
    copy_paste= 0.0,
    scale = 0.8,
    dropout = 0.3,
    )
