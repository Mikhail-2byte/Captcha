import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from collections import defaultdict

# Класс для очистки изображения
class ImageCleaner:
    def __init__(self, image_path):
        # Загружаем изображение в оттенках серого
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    def clean_image(self):
        # Пороговая бинаризация
        _, binary = cv2.threshold(self.img, 120, 255, cv2.THRESH_BINARY)

        # Инверсия цветов (фон становится черным, объекты — белыми)
        inverted = cv2.bitwise_not(binary)

        # Морфологическая операция (ядро 2x2, 1 итерация)
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)

        # Удаление мелких объектов
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 10 or w < 10:  # Удаляем объекты размером меньше 10 пикселей
                cv2.drawContours(clean, [contour], -1, 0, thickness=cv2.FILLED)

        # Инвертируем обратно в черно-белый вид
        self.result = cv2.bitwise_not(clean)

    def save_image(self, output_path):
        # Сохраняем очищенное изображение
        cv2.imwrite(output_path, self.result)

# Функция для нахождения контуров цифр
def find_contours(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 60 and 10 < w < 50:
            digits.append((x, y, w, h))

    # Сортируем контуры по X-координате
    digits = sorted(digits, key=lambda b: b[0])
    return digits, thresh

# Функция для распознавания цифр с изображения
def predict_digits(image_path, model):
    # Очистка изображения перед распознаванием
    cleaner = ImageCleaner(image_path)
    cleaner.clean_image()

    # Сохранение очищенного изображения во временный файл
    cleaned_path = 'cleaned_image.jpg'
    cleaner.save_image(cleaned_path)

    # Поиск контуров цифр на очищенном изображении
    digits, _ = find_contours(cleaned_path)

    predictions = []
    for (x, y, w, h) in digits:
        img = Image.open(cleaned_path).crop((x, y, x + w, y + h)).resize((24, 44))
        img_array = np.array(img) / 255.0  # Нормализация
        img_array = img_array[np.newaxis, ..., np.newaxis]  # Форма (1, 44, 24, 1)

        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        predictions.append(predicted_label)

    result = ''.join(map(str, predictions))
    print(f'Распознанная последовательность: {result}')
    return result

# Основной код
if __name__ == "__main__":
    # Загрузка модели
    model = load_model('model.h5')

    # Путь к изображению с несколькими цифрами
    image_path = '/content/captcha.jpg'

    # Выполняем предсказание
    result = predict_digits(image_path, model)
