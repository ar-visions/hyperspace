import subprocess
import os
import shutil

windows = os.name == 'nt'
subprocess.run(['make.bat' if windows else 'make', 'update'])

def copy_overlay(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dest_dir, relative_path)
        os.makedirs(dest_path, exist_ok=True)
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_path, file)
            shutil.copy2(src_file_path, dest_file_path)

src_directory = "../../hyperspace/overlays/blender/scripts"
dest_directory = "./scripts"
copy_overlay(src_directory, dest_directory)