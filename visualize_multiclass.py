import cv2
import numpy as np
from ultralytics import YOLO
import random
import json
from pathlib import Path


def generate_colors(n):
    """Генерирует n различных цветов в формате BGR."""
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]


def save_yolo_labels(boxes, class_names, image_shape, output_txt_path):
    """Сохраняет предсказания в формате YOLO (нормализованные координаты)."""
    h_img, w_img = image_shape[:2]
    with open(output_txt_path, "w") as f:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()

            # Преобразуем в центр, ширину, высоту
            x_center = (x1 + x2) / 2 / w_img
            y_center = (y1 + y2) / 2 / h_img
            w_norm = (x2 - x1) / w_img
            h_norm = (y2 - y1) / h_img

            f.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {conf:.6f}\n")


def save_predictions_json(boxes, class_names, output_json_path):
    """Сохраняет предсказания в читаемом JSON-формате."""
    predictions = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        class_name = class_names[cls_id] if cls_id < len(
            class_names) else f"Unknown({cls_id})"

        predictions.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_cxcywh": [
                round((x1 + x2) / 2, 2),
                round((y1 + y2) / 2, 2),
                round(x2 - x1, 2),
                round(y2 - y1, 2)
            ]
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def visualize_multiclass_prediction(
    model_path: str,
    image_path: str,
    results_dir: str = "results",
    class_names: list = None,
    conf_threshold: float = 0.25,
    box_thickness: int = 2,
    font_scale: float = 0.7,
    text_thickness: int = 1
):
    """
    Визуализация и сохранение результатов в папку results/.

    Сохраняет:
      - results/images/{name}.jpg       → изображение с bbox
      - results/labels/{name}.txt       → аннотации в формате YOLO + уверенность
      - results/json/{name}.json        → подробные предсказания в JSON
    """
    # Настройка путей
    results_path = Path(results_dir)
    (results_path / "images").mkdir(parents=True, exist_ok=True)
    (results_path / "labels").mkdir(parents=True, exist_ok=True)
    (results_path / "json").mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    stem = image_path.stem

    output_image = results_path / "images" / f"{stem}.jpg"
    output_txt = results_path / "labels" / f"{stem}.txt"
    output_json = results_path / "json" / f"{stem}.json"

    # Загрузка модели и изображения
    model = YOLO(model_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Не удалось загрузить изображение: {image_path}")

    # Имена классов
    if class_names is None:
        class_names = getattr(model.model, 'names', [
                              f"Class_{i}" for i in range(1)])

    # Предсказание
    results = model(image)
    result = results[0]

    # Аннотированное изображение
    annotated_image = image.copy()
    colors = generate_colors(len(class_names))

    valid_boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            if box.conf[0].item() >= conf_threshold:
                valid_boxes.append(box)

    # Визуализация на изображении
    for box in valid_boxes:
        conf = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())

        color = colors[cls_id] if cls_id < len(colors) else (0, 0, 255)
        label = f"{class_names[cls_id] if cls_id < len(class_names) else 'Unknown'}: {conf:.2f}"

        cv2.rectangle(annotated_image, (x1, y1),
                      (x2, y2), color, box_thickness)
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.rectangle(annotated_image, (x1, y1 - h - 10),
                      (x1 + w, y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)

        print(f"{label} | ({x1}, {y1}) - ({x2}, {y2})")

    if not valid_boxes:
        print("⚠️ Никаких объектов не обнаружено.")

    # Сохранение результатов
    cv2.imwrite(str(output_image), annotated_image)
    save_yolo_labels(valid_boxes, class_names, image.shape, output_txt)
    save_predictions_json(valid_boxes, class_names, output_json)

    print(f"\n✅ Результаты сохранены в папку: {results_dir}/")
    print(f"   Изображение: {output_image}")
    print(f"   Аннотации:   {output_txt}")
    print(f"   JSON:        {output_json}")


if __name__ == "__main__":
    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltrare",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax"
    ]

    visualize_multiclass_prediction(
        model_path="./runs/detect/train/weights/best.pt",
        image_path="patient_chest_xray.png",
        results_dir="results",
        class_names=CLASS_NAMES,
        conf_threshold=0.25
    )
