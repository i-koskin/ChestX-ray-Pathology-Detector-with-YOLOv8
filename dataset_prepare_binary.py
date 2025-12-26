import os
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split


def prepare_dataset_binary(
    csv_path: str,
    images_src_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Подготавливает датасет YOLO с ОДНИМ классом 'Abnormal' (все патологии объединены).

    Args:
        csv_path: путь к BBox_List_2017.csv
        images_src_dir: папка с изображениями ('./dataset/images')
        output_dir: выходная папка для YOLO-формата
        val_ratio: доля валидационной выборки (без oversampling!)
        random_seed: для воспроизводимости
    """
   # 1. Настройка путей
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Загрузка аннотаций
    df = pd.read_csv(csv_path)
    grouped = df.groupby('Image Index')

    image_ids = list(grouped.groups.keys())

    # 3. Разделение на train/val
    train_ids, val_ids = train_test_split(
        image_ids, test_size=val_ratio, random_state=random_seed
    )

    # 4. Копирование изображения и создание .txt
    def copy_image_and_label(img_id: str, split: str):
        src_img = Path(images_src_dir) / img_id
        if not src_img.exists():
            return

        # Копируем изображение
        shutil.copy(src_img, output_dir / "images" / split / img_id)

        # Создаём .txt аннотацию. Все bbox'ы имеют class_id = 0
        anns = grouped.get_group(img_id)
        w_img, h_img = Image.open(src_img).size

        label_path = output_dir / "labels" / \
            split / (Path(img_id).stem + ".txt")
        with open(label_path, "w") as f:
            for _, row in anns.iterrows():
                x = row['Bbox [x']
                y = row['y']
                w = row['w']
                h = row['h]']

                # Нормализуем
                xc = (x + w / 2) / w_img
                yc = (y + h / 2) / h_img
                wn = w / w_img
                hn = h / h_img

                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # 5. Обработка train и val
    print("\n📤 Копирование train...")
    for img_id in train_ids:
        copy_image_and_label(img_id, "train")

    print("📤 Копирование val...")
    for img_id in val_ids:
        copy_image_and_label(img_id, "val")

    # 10. Создание data.yaml
    yaml_content = """train: ./images/train
val: ./images/val
nc: 1
names:
  - Abnormal
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"✅ Бинарный датасет сохранён в: {output_dir}")
    print(f"   Изображений в train: {len(train_ids)}")
    print(f"   Изображений в val:   {len(val_ids)}")


if __name__ == "__main__":

    CSV_PATH = "./BBox_List_2017.csv"
    IMG_SRC = Path("./dataset/images")
    OUTPUT_DIR = Path("chestxray_yolo_binary")

    prepare_dataset_binary(
        csv_path=CSV_PATH,
        images_src_dir=IMG_SRC,
        output_dir=OUTPUT_DIR
    )
