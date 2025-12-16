import albumentations as A
import cv2
import numpy as np
import pandas as pd
import random
import math
from pathlib import Path
from collections import Counter


def prepare_dataset_with_oversampling_and_augmentation(
    csv_path: str,
    images_src_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    rare_classes: list = None,
    augmentation_factor: int = 2
):
    """
    Подготавливает датасет с:
      - oversampling до медианы,
      - дополнительной аугментацией для редких классов.

    Args:
        rare_classes: список классов, для которых применять доп. аугментацию
        augmentation_factor: сколько аугментированных копий создавать на оригинал
    """
    if rare_classes is None:
        rare_classes = ["Mass", "Nodule", "Pneumonia", "Pneumothorax"]

    output_path = Path(output_dir)
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    grouped = df.groupby('Image Index')
    img_to_labels = {img_id: group['Finding Label'].unique().tolist()
                     for img_id, group in grouped}

    class_counts = Counter()
    for labels in img_to_labels.values():
        for label in labels:
            class_counts[label] += 1

    target_count = int(pd.Series(list(class_counts.values())).median())
    repeat_factors = {
        cls: 1.0 if cnt >= target_count else target_count / cnt
        for cls, cnt in class_counts.items()
    }

    img_repeat_map = {
        img_id: math.ceil(max(repeat_factors[label] for label in labels))
        for img_id, labels in img_to_labels.items()
    }

    # Разделение на train/val
    random.seed(random_seed)
    all_ids = list(img_to_labels.keys())
    random.shuffle(all_ids)
    n_val = int(val_ratio * len(all_ids))
    val_ids = set(all_ids[:n_val])
    train_ids = [img for img in all_ids if img not in val_ids]

    CLASSES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax"
    ]
    class_to_id = {cls: i for i, cls in enumerate(CLASSES)}

    # === 8. Аугментации (только для редких классов) ===
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={
                "x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # сдвиг до 5%
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)
                   },                   # масштаб ±10%
            # поворот ±5°
            rotate=(-5, 5),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        )
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def save_image_and_label(img_array, bboxes, labels, split, name):
        """Сохраняет изображение и .txt файл"""
        cv2.imwrite(str(output_path / "images" / split / name), img_array)
        stem = Path(name).stem
        with open(output_path / "labels" / split / f"{stem}.txt", "w") as f:
            h, w = img_array.shape[:2]
            for (x, y, bw, bh), cls in zip(bboxes, labels):
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                wn = bw / w
                hn = bh / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    def process_original_image(img_id, split, copy_id=None):
        """Обрабатывает оригинальное изображение (без аугментации)"""
        src = Path(images_src_dir) / img_id
        if not src.exists():
            return
        img = cv2.imread(str(src))
        anns = grouped.get_group(img_id)
        bboxes = []
        labels = []
        for _, row in anns.iterrows():
            bboxes.append([row['Bbox [x'], row['y'], row['w'], row['h]']])
            labels.append(class_to_id[row['Finding Label']])

        stem = Path(img_id).stem
        suffix = f"_{copy_id}" if copy_id is not None else ""
        name = f"{stem}{suffix}{src.suffix}"
        save_image_and_label(img, bboxes, labels, split, name)

    # === 9. Обработка train: с аугментацией для редких классов ===
    print("📤 Обработка train с аугментацией редких классов...")
    train_counter = 0

    for img_id in train_ids:
        labels = img_to_labels[img_id]
        # Проверяем: содержит ли изображение редкие классы?
        has_rare = any(cls in rare_classes for cls in labels)

        # Сначала сохраняем оригинальные копии (как в oversampling)
        repeat = img_repeat_map[img_id]
        for i in range(repeat):
            process_original_image(img_id, "train", copy_id=train_counter)
            train_counter += 1

        # Если есть редкие классы — создаём доп. аугментированные копии
        if has_rare:
            src = Path(images_src_dir) / img_id
            if not src.exists():
                continue
            img = cv2.imread(str(src))
            anns = grouped.get_group(img_id)
            bboxes = []
            class_labels = []
            for _, row in anns.iterrows():
                bboxes.append([row['Bbox [x'], row['y'], row['w'], row['h]']])
                class_labels.append(row['Finding Label'])

            # Генерируем augmentation_factor аугментированных версий
            for aug_i in range(augmentation_factor):
                try:
                    augmented = augment(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_names = augmented['class_labels']
                    aug_labels = [class_to_id[cls] for cls in aug_class_names]

                    name = f"{Path(img_id).stem}_aug{aug_i}_{train_counter}{src.suffix}"
                    save_image_and_label(
                        aug_img, aug_bboxes, aug_labels, "train", name)
                    train_counter += 1
                except Exception as e:
                    print(f"⚠️ Ошибка аугментации {img_id}: {e}")

    # === 10. Обработка val ===
    print("📤 Обработка val...")
    for img_id in val_ids:
        process_original_image(img_id, "val")

    # === 11. data.yaml ===
    yaml_content = f"""train: ./images/train
val: ./images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(output_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Датасет с аугментацией сохранён в: {output_dir}")
    print(f"   Редкие классы: {rare_classes}")
    print(
        f"   Доп. аугментированных копий на редкое изображение: {augmentation_factor}")


if __name__ == "__main__":

    # Пути
    OUTPUT_DIR = Path("chestxray_yolo_2")
    IMG_SRC = Path("./dataset/images")
    CSV_PATH = "./BBox_List_2017.csv"

    prepare_dataset_with_oversampling_and_augmentation(
        csv_path=CSV_PATH,
        images_src_dir=IMG_SRC,
        output_dir=OUTPUT_DIR
    )
