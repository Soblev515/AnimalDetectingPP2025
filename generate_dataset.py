import os
import shutil
import random

source_dir = 'final_crop' 
train_dir = 'dataset/train'   
test_dir = 'dataset/test'   
test_ratio = 0.2           

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if os.path.isdir(class_path):
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        images = os.listdir(class_path)
        random.shuffle(images)
        
        test_size = int(len(images) * test_ratio)

        for i, image in enumerate(images):
            src_image_path = os.path.join(class_path, image)
            if i < test_size:
                shutil.copy(src_image_path, os.path.join(test_dir, class_name, image))
            else:
                shutil.copy(src_image_path, os.path.join(train_dir, class_name, image))

print("Разделение завершено!")
