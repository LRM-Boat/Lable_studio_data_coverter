import os
import os.path as osp
import shutil
import json
import cv2
import numpy as np
import glob

#---------------Settings---------------------------------
VAR = "48" #Folder with data
base_dir = '/home/user/mmsegmentation/train'  #absolute path to your data folder
images_subdir = VAR + '/images'         # folder with images
MASK_FOLDER = VAR + "/annotation"       # BrushLabels masks
OUTPUT_FOLDER = VAR + "/labels"         # folder for txt and png
JSON_PATH = osp.join(base_dir, VAR, f"{VAR}_re.json") #name for normal json
OLD_JSON=osp.join(base_dir, VAR, f"{VAR}.json") #minjson from label-studio

TARGET_ROOT = "SurfaceDataset" #folder with your dataset where you want to save converted data
TARGET_IMAGES = osp.join(TARGET_ROOT, "images") 
TARGET_LABELS = osp.join(TARGET_ROOT, "labels")

os.makedirs(TARGET_IMAGES, exist_ok=True)
os.makedirs(TARGET_LABELS, exist_ok=True)

# Classes
class_to_index = {
    'background': 0,
    'gravel': 1,
    'dirt': 2,
    'asphalt': 3,
    'grass': 4,
    'obstacle': 5,
    'wall': 6
}
# Pallete:
palette = [
    [196, 123, 55],   # background
    [65, 72, 74],     # gravel
    [69, 115, 153],   # dirt
    [173, 179, 184],  # asphalt
    [126, 194, 46],   # grass
    [38, 38, 201],    # obstacle
    [2,   2,   3]     # wall
]


# ------------------------
# 1) MINSON rename
# ------------------------
with open(OLD_JSON, "r", encoding="utf-8") as f:
    annotations = json.load(f)

for ann in annotations:
    old_name = ann["image"]
    idx = old_name.find("frame_set")
    if idx != -1:
        new_name = old_name[idx:]  
        ann["image"] = new_name

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"Обновили JSON: {JSON_PATH}")


# ------------------------
# 2) Generation txt masks
# ------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)

for ann in annotations:
    task_id = ann["id"]
    image_field = ann["image"]  # "frame_set24_81.jpg" после обрезки

    # Собираем полный путь к исходной картинке
    #   base_dir + images_subdir + image_field
    image_file = osp.join(base_dir, images_subdir, image_field)

    print(f"Обрабатываем task_id={task_id}, image={image_file}")

    if not osp.exists(image_file):
        print(f"  - Файл {image_file} не найден. Пропускаем.")
        continue

    img = cv2.imread(image_file)
    if img is None:
        print(f"  - Не удалось открыть {image_file}. Пропускаем.")
        continue

    height, width, _ = img.shape

  
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    # Перебираем классы
    for class_name, class_idx in class_to_index.items():
        if class_name == 'background':
            continue

        # Ищем brush-маски
        pattern = f"task-{task_id}-annotation-*-BrushLabels-{class_name}-*.png"
        search_path = osp.join(MASK_FOLDER, pattern)
        mask_files = glob.glob(search_path)

        if not mask_files:
            continue

        for mask_file in mask_files:
            print(f"    Класс {class_name}, файл маски: {mask_file}")
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"      * Не удалось открыть {mask_file}, пропускаем.")
                continue

            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # mask==255 => class_idx
            merged_mask[mask == 255] = class_idx

  
    image_name = osp.splitext(osp.basename(image_file))[0]
    # .txt 
    out_txt_name = f"{image_name}.txt"
    out_txt_path = osp.join(OUTPUT_FOLDER, out_txt_name)
    np.savetxt(out_txt_path, merged_mask, fmt='%d')
    print(f"  - Сохранили txt-маску: {out_txt_path}")

    out_png_name = f"{image_name}.png"
    out_png_path = osp.join(OUTPUT_FOLDER, out_png_name)

    cv2.imwrite(out_png_path, merged_mask)
    print(f"  - Сохранили индексы классов в PNG: {out_png_path}")
   
    # ---------------------------
    # 3) Copy into your dataset folder 
    # ---------------------------

    
    target_img_path = osp.join(TARGET_IMAGES, osp.basename(image_file))
    shutil.copy2(image_file, target_img_path)
    print(f"  - Копируем исходное изображение в {target_img_path}")


    target_txt_path = osp.join(TARGET_LABELS, osp.basename(out_txt_path))
    shutil.copy2(out_txt_path, target_txt_path)

    target_mask_png_path = osp.join(TARGET_LABELS, osp.basename(out_png_path))
    shutil.copy2(out_png_path, target_mask_png_path)
    print(f"  - Копируем txt и png в {TARGET_LABELS}\n")

print("Завершено!")
