import os
import tarfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_images(
    csv_path: str,
    archives_dir: str,
    output_dir: str,
    max_archives: int = 12
):
    """
    Извлекает изображения, перечисленные в BBox_List_2017.csv,
    из архивов images_*.tar.gz и сохраняет в output_dir.

    Args:
        csv_path (str): Путь к BBox_List_2017.csv
        archives_dir (str): Папка с архивами images_*.tar.gz
        output_dir (str): Папка для сохранения изображений
        max_archives (int): Максимальный номер архива (обычно 12)
    """
    # Создаём выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Читаем CSV и получаем уникальные имена изображений
    df = pd.read_csv(csv_path)
    needed_images = set(df['Image Index'].unique())
    print(
        f"🎯 Найдено {len(needed_images)} уникальных изображений в аннотациях.")

    found_images = set()

    # Перебираем архивы images_001.tar.gz ... images_012.tar.gz
    for i in tqdm(range(1, max_archives + 1), desc="Поиск по архивам"):
        archive_name = f"images_{i:03d}.tar.gz"
        archive_path = Path(archives_dir) / archive_name

        if not archive_path.exists():
            print(f"⚠️ Архив {archive_name} не найден.")
            continue

        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Получаем все файлы внутри архива
                members = [m for m in tar.getmembers() if m.isfile(
                ) and m.name.endswith(('.png', '.jpg', '.jpeg'))]
                # Извлекаем имена файлов (без пути)
                archive_image_names = {Path(m.name).name: m for m in members}

                # Ищем пересечение с нужными изображениями
                to_extract = needed_images - found_images
                available_in_archive = to_extract & set(
                    archive_image_names.keys())

                if available_in_archive:
                    for img_name in available_in_archive:
                        member = archive_image_names[img_name]
                        # Извлекаем напрямую в output_dir
                        member.name = img_name  # убираем внутренний путь "images/..."
                        tar.extract(member, path=output_dir)
                        found_images.add(img_name)

                    print(
                        f"📦 Из {archive_name} извлечено {len(available_in_archive)} изображений.")

                if len(found_images) == len(needed_images):
                    print("✅ Все изображения найдены!")
                    break

        except Exception as e:
            print(f"❌ Ошибка при обработке {archive_name}: {e}")

    missing = needed_images - found_images
    if missing:
        print(f"❗ Не найдены {len(missing)} изображений:")
        for img in sorted(missing):
            print(f"  - {img}")
    else:
        print(
            f"Все {len(found_images)} изображений успешно извлечены в {output_dir}")


# ================ ЗАПУСК ================
if __name__ == "__main__":

    CSV_PATH = "BBox_List_2017.csv"
    ARCHIVES_DIR = "./dataset"
    OUTPUT_DIR = "dataset/images"

    extract_images(
        csv_path=CSV_PATH,
        archives_dir=ARCHIVES_DIR,
        output_dir=OUTPUT_DIR,
        max_archives=12  # используем 12 архивов из NIH
    )
