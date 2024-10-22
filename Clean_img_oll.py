import cv2
import numpy as np
import os

class ImageCleaner:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        # Создаем папку для сохранения, если её ещё нет
        os.makedirs(self.output_path, exist_ok=True)

    def clean_image(self, image):
        # Бинаризация изображения
        _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

        # Инверсия, чтобы фон стал черным, а объекты белыми
        inverted = cv2.bitwise_not(binary)

        # Морфологическая обработка: мягкий фильтр для удаления мелких шумов
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Поиск и удаление тонких контуров
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 10 or w < 10:
                cv2.drawContours(clean, [contour], -1, 0, thickness=cv2.FILLED)

        # Инвертируем обратно в черно-белый вид
        return cv2.bitwise_not(clean)

    def process_folder(self):
        # Обрабатываем все файлы в папке
        for filename in os.listdir(self.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_file = os.path.join(self.input_path, filename)
                output_file = os.path.join(self.output_path, filename)

                # Загружаем изображение в градациях серого
                image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Ошибка загрузки файла: {input_file}")
                    continue

                # Обрабатываем изображение
                cleaned_image = self.clean_image(image)

                # Сохраняем результат
                cv2.imwrite(output_file, cleaned_image)
                print(f"Обработано: {filename}")

# Путь до папок с изображениями
input_path = "C:/Python work/Captcha/full_dataset/processed_captcha"
output_path = "C:/Python work/Captcha/full_dataset/cleaned_captcha"

# Запуск обработки
cleaner = ImageCleaner(input_path, output_path)
cleaner.process_folder()

print("Все изображения успешно обработаны и сохранены.")
