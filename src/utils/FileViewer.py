import os
import shutil

def list_files(filepath, suffix=None, prefix=None, isdepth=True):
    files = []
    for fpathe, dirs, fs in os.walk(filepath):
        for f in fs:
            if suffix is None and prefix is None:
                files.append(os.path.join(fpathe, f))
            elif prefix is None:
                if f.endswith(suffix):
                    files.append(os.path.join(fpathe, f))
            elif suffix is None:
                if f.startswith(prefix):
                    files.append(os.path.join(fpathe, f))
            else:
                if f.startswith(prefix) and f.endswith(suffix):
                    files.append(os.path.join(fpathe, f))

        if isdepth == False:
            break
    return files

def list_filenames(filepath, suffix=None, prefix=None):
    filenames = []
    for fpathe, dirs, fs in os.walk(filepath):
        for f in fs:
            if suffix is None and prefix is None:
                filenames.append(f)
            elif prefix is None:
                if f.endswith(suffix):
                    filenames.append(f)
            elif suffix is None:
                if f.startswith(prefix):
                    filenames.append(f)
            else:
                if f.startswith(prefix) and f.endswith(suffix):
                    filenames.append(f)
        break
    return filenames

def detect_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def all_files_exist(paths):
    flag = True
    for path in paths:
        if not os.path.exists(path):
            flag = False
            break

    return flag

def clean_dir(directory):
    """Deletes all contents of the directory but keeps the directory itself."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove folder
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)
