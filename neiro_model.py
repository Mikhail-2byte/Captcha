import zipfile
import os
from PIL import Image
import numpy as np

# Путь к zip-файлу
zip_file = zipfile.ZipFile('output_digits.zip')

# Списки для хранения изображений и меток
train_images = []
train_labels = []

# Функция для проверки и извлечения метки
def extract_label(file_name):
    try:
        label = int(file_name.split('_')[0])  # Извлекаем метку до '_'
        return label
    except ValueError:
        return None  # Возвращаем None, если метка некорректная

# Извлечение файлов из архива
with zip_file as zf:
    for file in zf.namelist():
        if file.endswith(('.jpg', '.png')):
            # Пытаемся извлечь метку
            label = extract_label(os.path.basename(file))
            if label is None:
                print(f'Пропуск файла: {file} (некорректная метка)')
                continue  # Пропускаем файл, если не удалось извлечь метку

            # Загружаем изображение и нормализуем
            with zf.open(file) as img_file:
                img = Image.open(img_file).convert('L')  # Оттенки серого
                img = np.array(img) / 255.0  # Нормализация
                train_images.append(img)
                train_labels.append(label)

# Преобразуем в numpy-массивы
train_images = np.array(train_images)
train_labels = np.array(train_labels)

print(f'Извлечено {len(train_images)} изображений и {len(train_labels)} меток')


import matplotlib.pyplot as plt
import numpy as np

# Проверим, что данные загружены
print(f'Всего изображений: {len(train_images)}, меток: {len(train_labels)}')

# Функция для отображения 10 случайных изображений
def show_random_images(images, labels, n=10):
    indices = np.random.choice(len(images), n, replace=False)  # Случайные индексы
    plt.figure(figsize=(15, 5))  # Настраиваем размер окна

    for i, idx in enumerate(indices):
        plt.subplot(1, n, i + 1)  # Создаем подграфик
        plt.imshow(images[idx], cmap='gray')  # Отображаем изображение в оттенках серого
        plt.title(f'Метка: {labels[idx]}')  # Подпись с меткой
        plt.axis('off')  # Скрываем оси

    plt.show()

# Вызов функции для отображения 10 случайных изображений
show_random_images(train_images, train_labels)

from sklearn.model_selection import train_test_split
import numpy as np

# Добавляем размерность канала (1 для черно-белых изображений)
train_images = train_images[..., np.newaxis]  # Теперь форма: (116508, 44, 24, 1)

# Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

print(f'Размер обучающей выборки: {X_train.shape}')
print(f'Размер тестовой выборки: {X_test.shape}')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Создание модели
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 классов для цифр от 0 до 9
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()


# Обучение модели с выделением 10% данных на валидацию
history = model.fit(
    X_train, y_train,
    epochs=7,
    validation_split=0.1,  # 10% обучающих данных для валидации
    batch_size=32
)


model.save('model.h5')
print("Модель сохранена в формате HDF5.")



#Конвертация модели в формат onnx

from tensorflow.keras.models import load_model

model = load_model('model.h5')
print("Модель загружена из файла model.h5.")



# Загрузите модель Keras
model = tf.keras.models.load_model('/content/model.h5')

# Преобразуйте Sequential модель в функциональную, если это применимо
if isinstance(model, tf.keras.Sequential):
    inputs = tf.keras.Input(shape=(44, 24, 1))
    outputs = model(inputs)
    model = tf.keras.Model(inputs, outputs)

# Определите спецификацию входных данных spec = (tf.TensorSpec((None, 44, 24, 1), tf.float32, name="input"),)

# Определите путь для сохранения ONNX модели
output_path = "model.onnx"

# Конвертируйте модель в формат ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print(f"Модель успешно конвертирована в {output_path}")
