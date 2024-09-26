import zipfile
import os

zip_dir = 'model_thirt_epoch'
extract_dir = 'model_thirt_epoch'

for filename in os.listdir(zip_dir):
    if filename.endswith('.zip'):
        zip_path = os.path.join(zip_dir, filename)
        folder_name = os.path.splitext(filename)[0]
        extract_folder = os.path.join(extract_dir, folder_name)
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        print(f'Файл {filename} распакован в {extract_folder}')
