import os
import cv2
import numpy as np

# Палитра (индекс 0 => background, 1 => gravel, ...)
palette = [
    (196, 123, 55),    # background
    (65, 72, 74),      # gravel
    (69, 115, 153),    # dirt
    (173, 179, 184),   # asphalt
    (126, 194, 46),    # grass
    (38, 38, 201),     # obstacle
    (2, 2, 3)          # wall
]


mask_name="frame_set8_844.png"

MASK_FOLDER="SurfaceDataset/labels/"


mask_path = MASK_FOLDER+mask_name

# Предположим, что исходное изображение лежит в папке:
IMAGE_FOLDER = "SurfaceDataset/images"

# 1) Считываем PNG-маску (одноканальную)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if mask is None:
    print(f"Не удалось открыть {mask_path}")
    exit(1)

if len(mask.shape) != 2:
    print("Ошибка: маска должна быть одноканальной (H,W).")
    exit(1)

height, width = mask.shape

# 2) Формируем цветную версию маски
color_image = np.zeros((height, width, 3), dtype=np.uint8)
for class_idx, color in enumerate(palette):
    # Где mask == class_idx => ставим данный цвет
    color_image[mask == class_idx] = color

# 3) Пытаемся найти исходное изображение
#    берём имя без расширения из MASK_PATH
base_name = os.path.splitext(os.path.basename(mask_path))[0]  
# "frame_set8_844"

# предполагаем, что исходное изображение -> base_name + ".jpg"
image_path = os.path.join(IMAGE_FOLDER, base_name + ".jpg")

img = cv2.imread(image_path)
if img is None:
    print(f"Не удалось открыть исходное изображение: {image_path}")
    exit(1)

# Убедимся, что размер совпадает
if img.shape[0] != height or img.shape[1] != width:
    print("Предупреждение: Размер маски и изображения не совпадают, "
          "можно сделать resize color_image.")
    # Например, color_image = cv2.resize(color_image, (img.shape[1], img.shape[0]))

# 4) Наложение (overlay) цветной маски поверх оригинального изображения
alpha = 0.5  # степень прозрачности (0..1)
overlay = cv2.addWeighted(img, 1-alpha, color_image, alpha, 0)

# 5) Показываем результат
cv2.imshow("Original Image", img)
cv2.imshow("Seg Mask (Color)", color_image)
cv2.imshow("Overlay", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Если хотите сохранить результат:
# cv2.imwrite("overlay_result.png", overlay)
