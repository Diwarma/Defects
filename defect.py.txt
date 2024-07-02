import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import os
import math

# --- 1. Подготовка данных ---
def load_data(data_dir):
    images = []
    labels = []
    bboxes = []
    class_names = sorted(os.listdir(os.path.join(data_dir, 'Annotations')))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, 'images', class_name)
        annotation_dir = os.path.join(data_dir, 'Annotations', class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(class_idx)

            # Загрузка аннотаций из XML-файла
            annotation_filename = image_name[:-4] + '.xml'
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            boxes = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
            bboxes.append(boxes)

    return np.array(images), np.array(labels), bboxes, class_names

def preprocess_image(image, target_size=(128, 128)):
    resized = cv2.resize(image, target_size)
    normalized = resized / 255.0
    return normalized

def create_cnn_model(num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# --- 2. Аугментация данных ---
def augment_data(images, labels, bboxes, class_names, angles_file, output_dir):
    """
    Поворачивает изображения и обновляет ограничивающие рамки.
    """
    for i, (image, label, boxes) in enumerate(zip(images, labels, bboxes)):
        angle = random.randint(-10, 10)
        rotated_image = preprocessing.image.random_rotation(image, angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

        # Обновление ограничивающих рамок
        h, w = image.shape[:2]
        cX, cY = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        
        rotated_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # Преобразование координат углов рамки
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32")
            points = np.array([cv2.transform(np.array([p]), M)[0] for p in points])
            # Нахождение новых координат рамки
            x_coords, y_coords = points[:, 0], points[:, 1]
            xmin, ymin = np.min(x_coords), np.min(y_coords)
            xmax, ymax = np.max(x_coords), np.max(y_coords)
            rotated_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

        # Сохранение повернутого изображения и аннотаций
        class_name = class_names[label]
        output_class_dir = os.path.join(output_dir, 'images', class_name)
        output_annotation_dir = os.path.join(output_dir, 'Annotations', class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        os.makedirs(output_annotation_dir, exist_ok=True)

        image_filename = f"{class_name}_{i:04d}_rotated.jpg"
        cv2.imwrite(os.path.join(output_class_dir, image_filename), rotated_image)

        # Создание XML-файла аннотации
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = class_name
        ET.SubElement(annotation, "filename").text = image_filename
        ET.SubElement(annotation, "path").text = os.path.join(output_class_dir, image_filename)
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(rotated_image.shape[1])
        ET.SubElement(size, "height").text = str(rotated_image.shape[0])
        ET.SubElement(size, "depth").text = str(rotated_image.shape[2])
        ET.SubElement(annotation, "segmented").text = "0"

        for box in rotated_boxes:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = class_name
            ET.SubElement(object, "pose").text = "Unspecified"
            ET.SubElement(object, "truncated").text = "0"
            ET.SubElement(object, "difficult").text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box[0])
            ET.SubElement(bndbox, "ymin").text = str(box[1])
            ET.SubElement(bndbox, "xmax").text = str(box[2])
            ET.SubElement(bndbox, "ymax").text = str(box[3])

        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(output_annotation_dir, image_filename[:-4] + ".xml"))

        # Запись информации о повороте в файл
        with open(angles_file, 'a') as f:
            f.write(f"{image_filename}\t{angle}\n")

# --- 3. Загрузка и обработка данных ---
data_dir = 'PCB_DATASET'
images, labels, bboxes, class_names = load_data(data_dir)

# --- 4. Аугментация данных ---
output_dir = 'PCB_DATASET_AUGMENTED' # Папка для аугментированных данных
angles_file = 'rotation_angles.txt' # Файл для записи углов поворота
augment_data(images, labels, bboxes, class_names, angles_file, output_dir)

# --- 5. Загрузка аугментированных данных ---
augmented_images, augmented_labels, _, _ = load_data(output_dir)

# --- 6. Объединение данных ---
images = np.concatenate((images, augmented_images))
labels = np.concatenate((labels, augmented_labels))

# --- 7. Предобработка данных ---
images = np.array([preprocess_image(img) for img in images])

# --- 8. Преобразование меток ---
labels = tf.keras.utils.to_categorical(labels)

# --- 9. Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# --- 10. Создание и обучение модели ---
num_classes = len(class_names)
model = create_cnn_model(num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# --- 11. Тестирование модели ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# --- 12. Пример использования модели ---
new_image = cv2.imread('test_pcb.jpg')
processed_image = preprocess_image(new_image)
processed_image = np.expand_dims(processed_image, axis=0)
processed_image = np.expand_dims(processed_image, axis=-1)

prediction = model.predict(processed_image)
predicted_class_idx = np.argmax(prediction[0])
defect_class = class_names[predicted_class_idx]
print(f'Предсказанный класс: {defect_class}') 