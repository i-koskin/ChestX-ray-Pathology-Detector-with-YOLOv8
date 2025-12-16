import cv2
import numpy as np
from ultralytics import YOLO
import random
from pathlib import Path


def generate_colors(n):
    """Генерирует n различных цветов в формате BGR."""
    random.seed(42)
    colors = []
    for i in range(n):
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        colors.append(color)
    return colors


def visualize_multiclass_prediction(
    model_path: str,
    image_path: str,
    output_path: str = "prediction_multiclass.jpg",
    class_names: list = None,
    conf_threshold: float = 0.25,
    box_thickness: int = 2,
    font_scale: float = 0.7,
    text_thickness: int = 1
):
    """
    Визуализация предсказаний YOLOv8 с поддержкой многоклассовой детекции.

    Args:
        model_path: путь к .pt файлу модели
        image_path: путь к входному изображению
        output_path: куда сохранить результат
        class_names: список имён классов (если None — попытается загрузить из модели)
        conf_threshold: минимальная уверенность для отображения bbox
    """
    # Загрузка модели
    model = YOLO(model_path)

    # Если имена классов не заданы — берём из модели
    if class_names is None:
        if hasattr(model.model, 'names'):
            class_names = model.model.names
        else:
            # Резервный вариант: генерируем Class_0, Class_1...
            nc = model.model.yaml.get('nc', 1) if hasattr(
                model.model, 'yaml') else 1
            class_names = [f"Class_{i}" for i in range(nc)]

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Не удалось загрузить изображение: {image_path}")

    # Предсказание
    results = model(image)
    result = results[0]

    # Генерация цветов (по одному на класс)
    colors = generate_colors(len(class_names))

    # Копия изображения для аннотаций
    annotated_image = image.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())

            # Защита от выхода за границы
            if cls_id >= len(class_names):
                label = f"Unknown({cls_id}): {conf:.2f}"
                color = (0, 0, 255)  # красный для неизвестного
            else:
                label = f"{class_names[cls_id]}: {conf:.2f}"
                color = colors[cls_id]

            # Рисуем bounding box
            cv2.rectangle(annotated_image, (x1, y1),
                          (x2, y2), color, box_thickness)

            # Размер текста
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

            # Фон для текста
            cv2.rectangle(annotated_image, (x1, y1 - text_h - 10),
                          (x1 + text_w, y1), color, -1)

            # Текст (белый)
            cv2.putText(
                annotated_image, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness
            )

            print(
                f"{class_names[cls_id] if cls_id < len(class_names) else 'Unknown'}: {conf:.2f} | ({x1}, {y1}) - ({x2}, {y2})")
    else:
        print("⚠️ Объектов не обнаружено.")

    # Сохраняем результат
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ Результат сохранён: {output_path}")


if __name__ == "__main__":

    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltare",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax"
    ]

    visualize_multiclass_prediction(
        model_path="./runs/detect/train/weights/best.pt",
        image_path="patient_chest_xray.png",
        output_path="multiclass_result.jpg",
        class_names=CLASS_NAMES,
        conf_threshold=0.25
    )
