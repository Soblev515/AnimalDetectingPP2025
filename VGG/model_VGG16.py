import numpy as np
import time
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

num_classes = 15 
img_height, img_width = 200, 200  
epochs = 3  

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/test',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

start_time = time.time()

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs
)

end_time = time.time()
training_time = end_time - start_time

y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = validation_generator.classes

report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys(), output_dict=True)

metrics = {}
for class_name, metrics_dict in report.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        metrics[class_name] = {
            'Recall': metrics_dict['recall'],
            'Precision': metrics_dict['precision'],
            'F1': metrics_dict['f1-score']
        }

for class_name, metric_values in metrics.items():
    print(f"{class_name}: Recall={metric_values['Recall']:.2f}, Precision={metric_values['Precision']:.2f}, F1={metric_values['F1']:.2f}")

num_params = model.count_params() / 1_000_000  # Количество параметров в миллионах
images_per_second = train_generator.samples / training_time  # Изображений в секунду
accuracy = report['accuracy']  # Общая точность
num_test_images = validation_generator.samples  # Количество изображений в тестовом наборе

print(f"\nКоличество параметров: {num_params:.2f}M")
print(f"Скорость обучения: {training_time:.2f} секунд")
print(f"Изображений в секунду: {images_per_second:.2f}")
print(f"Точность: {accuracy:.2f}")
print(f"Количество изображений в тестовом наборе: {num_test_images}")
