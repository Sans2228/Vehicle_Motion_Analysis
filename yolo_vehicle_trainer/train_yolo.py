if __name__ == "__main__":
    from ultralytics import YOLO

    # Загружаем модель
    model = YOLO("yolo11n.pt")

    # Обучение
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="yolo11n-custom",
        pretrained=True,
        exist_ok = True
    )

    # Оценка модели
    metrics = model.val()