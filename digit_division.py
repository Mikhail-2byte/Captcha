import cv2
import numpy as np
from PIL import Image
import os
from collections import defaultdict

def find_contours(image_path):
    # Загрузка изображения в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Бинаризация изображения (инвертируем цвет для удобства)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Получение размеров изображения
    height, width = thresh.shape

    # Функция для нахождения черных пикселей (рекурсивно)
    def find_connected_component(start_x, start_y, thresh, visited):
        stack = [(start_x, start_y)]
        min_x, max_x = start_x, start_x
        min_y, max_y = start_y, start_y
        while stack:
            x, y = stack.pop()
            if visited[y, x] == 0 and thresh[y, x] == 255:
                visited[y, x] = 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= x + dx < width and 0 <= y + dy < height and visited[y + dy, x + dx] == 0:
                            stack.append((x + dx, y + dy))
        return min_x, max_x, min_y, max_y

    # Поиск всех цифр
    digits = []
    visited = np.zeros_like(thresh)  # Массив для отслеживания посещённых пикселей
    for x in range(width):
        for y in range(height):
            if y + 4 < height and x + 1 < width:
                rect = thresh[y:y + 5, x:x + 2]
                if np.all(rect == 255) and visited[y, x] == 0:
                    min_x, max_x, min_y, max_y = find_connected_component(x, y, thresh, visited)
                    if (max_x - min_x) > 35:
                        mid_x = (max_x + min_x) // 2
                        if len(digits) < 6:
                            digits.append((min_x, mid_x, min_y, max_y))  # Первая половина
                        if len(digits) < 6:
                            digits.append((mid_x + 1, max_x, min_y, max_y))  # Вторая половина
                    else:
                        if len(digits) < 6:
                            digits.append((min_x, max_x, min_y, max_y))
            if len(digits) >= 6:
                break
        if len(digits) >= 6:
            break

    return digits, thresh


def save_digits(image_path, digits, output_folder, counters):
    filename = os.path.basename(image_path).split('.')[0]
    labels = filename.split('_')[0]  # Берём первые 6 символов как метки

    img = Image.open(image_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (min_x, max_x, min_y, max_y) in enumerate(digits):
        digit_img = img.crop((min_x, min_y, max_x, max_y))
        digit_img = digit_img.resize((24, 44))

        label = labels[i]  # Получаем текущую метку
        counters[label] += 1  # Увеличиваем счётчик для этой метки

        # Генерируем имя файла в формате "метка_порядковый_номер.png"
        filename = f"{label}_{counters[label]}.jpg"
        digit_img.save(os.path.join(output_folder, filename))


def process_all_images(input_folder, output_folder):
    counters = defaultdict(int)  # Словарь для подсчёта найденных цифр

    # Проходим по всем файлам в папке
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):  # Проверяем расширение файла
            image_path = os.path.join(input_folder, filename)
            digits, _ = find_contours(image_path)  # Находим цифры
            save_digits(image_path, digits, output_folder, counters)  # Сохраняем цифры


# Пример использования
input_folder = 'C:/Python work/Captcha/full_dataset/cleaned_captcha'  # Папка с исходными изображениями
output_folder = 'C:/Python work/Captcha/output_digits'  # Папка для сохранения цифр

# Обработка всех изображений в папке
process_all_images(input_folder, output_folder)
