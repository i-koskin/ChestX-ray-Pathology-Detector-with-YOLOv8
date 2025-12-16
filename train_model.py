import argparse
from ultralytics import YOLO
from pathlib import Path


def train_model(dataset_dir: str, model_size: str, epochs: int, imgsz: int, batch: int):
    """ Обучение YOLOv8."""
    print(f"\n🚀 Обучение YOLOv8{model_size}...")
    model = YOLO(f"yolov8{model_size}.pt")
    model.train(
        data=Path(dataset_dir) / "data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fliplr=0.5,
        flipud=0.0,
        device=0 if Path("/usr/bin/nvidia-smi").exists() else "cpu"
    )
    print(f"✅ Обучение завершено. Веса: ./runs/detect/train/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн ChestX-ray детекции")
    parser.add_argument("--dataset_dir", type=str, default="chestxray_yolo",
                        help="Расположение размеченного датасета")
    parser.add_argument("--model_size", type=str, default="s",
                        choices=["n", "s", "m"], help="Версия модели YOLOv8")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir)

    train_model(str(DATASET_DIR), args.model_size,
                args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
