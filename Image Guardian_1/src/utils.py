import os

def is_valid_image(file_path):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions
