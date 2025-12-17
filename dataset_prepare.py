import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import shutil
import math
import random
from PIL import Image

# Классы
CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax"
]


def prepare_dataset_with_median_oversampling(
        csv_path: str,
        images_src_dir: str,
        output_dir: str,
        val_ratio: float = 0.15,
        random_seed: int = 42
):
    """
    Подготавливает датасет YOLO с адаптивным oversampling до медианы.

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

    # Уникальные изображения и их классы
    img_to_labels = {}
    for img_id, group in grouped:
        img_to_labels[img_id] = group['Finding Label'].unique().tolist()

    # 3. Подсчёт уникальных изображений на класс
    class_counts = Counter()
    for labels in img_to_labels.values():
        for label in labels:
            class_counts[label] += 1

    print("Распределение до oversampling (уникальные изображения):")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:15}: {cnt}")

    # 4. Вычисление целевого значения — медиана
    counts_list = list(class_counts.values())
    target_count = int(pd.Series(counts_list).median())
    print(f"\n🎯 Целевое количество на класс (медиана): {target_count}")

    # 5. Вычисление коэффициентов повторения
    repeat_factors = {}
    for cls, cnt in class_counts.items():
        if cnt >= target_count:
            repeat_factors[cls] = 1.0  # не повторяем
        else:
            # Например: cnt=60, target=200 → factor ≈ 3.33 → ceil → 4
            repeat_factors[cls] = target_count / cnt

    # 6. Определение, сколько раз включать каждое изображение
    img_repeat_map = {}
    for img_id, labels in img_to_labels.items():
        # Берём максимальный фактор среди всех классов на изображении
        max_factor = max(repeat_factors[label] for label in labels)
        img_repeat_map[img_id] = math.ceil(max_factor)

    # 7. Разделение на train/val (до oversampling!)
    random.seed(random_seed)
    all_img_ids = list(img_to_labels.keys())
    random.shuffle(all_img_ids)

    n_val = int(val_ratio * len(all_img_ids))
    val_ids = set(all_img_ids[:n_val])
    train_ids = [img for img in all_img_ids if img not in val_ids]

    # Применяем oversampling к train
    train_with_repeats = []
    for img_id in train_ids:
        repeat = img_repeat_map[img_id]
        for i in range(repeat):
            train_with_repeats.append((img_id, i))  # (id, копия_номер)

    print(f"\n📊 После oversampling:")
    print(f"  Train изображений (с копиями): {len(train_with_repeats)}")
    print(f"  Val изображений (оригиналы):   {len(val_ids)}")

    # 8. Копирование изображения и создание .txt
    def copy_image_and_label(img_id: str, split: str, copy_id: int = None):
        src_img = images_src_dir / img_id
        if not src_img.exists():
            print(f"⚠️ Пропущено: {img_id} не найдено")
            return

        # Новое имя: если копия — добавляем суффикс
        stem = Path(img_id).stem
        suffix = f"_{copy_id}" if copy_id is not None else ""
        ext = src_img.suffix
        new_name = f"{stem}{suffix}{ext}"

        # Копируем изображение
        dst_img = output_dir / "images" / split / new_name
        shutil.copy(src_img, dst_img)

        # Создаём .txt аннотацию
        anns = grouped.get_group(img_id)
        w_img, h_img = Image.open(src_img).size

        label_path = output_dir / "labels" / split / f"{stem}{suffix}.txt"
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
                cls_id = CLASSES.index(row['Finding Label'])
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # 9. Обработка train и val
    print("\n📤 Копирование train...")
    for img_id, copy_id in train_with_repeats:
        copy_image_and_label(img_id, "train", copy_id)

    print("📤 Копирование val...")
    for img_id in val_ids:
        copy_image_and_label(img_id, "val")

    # 10. Создание data.yaml
    yaml_content = f"""train: ./images/train
val: ./images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Датасет сохранён в: {output_dir}")
    print(f"   Изображений в train: {len(train_with_repeats)}")
    print(f"   Изображений в val:   {len(val_ids)}")


if __name__ == "__main__":

    CSV_PATH = "./BBox_List_2017.csv"
    IMG_SRC = Path("./dataset/images")
    OUTPUT_DIR = Path("chestxray_yolo")

    prepare_dataset_with_median_oversampling(
        csv_path=CSV_PATH,
        images_src_dir=IMG_SRC,
        output_dir=OUTPUT_DIR
    )
