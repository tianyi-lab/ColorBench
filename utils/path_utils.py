import os


def set_root_folder():
    CACHE_DIR = "YOUR_HF_CACHE_FOLDER"
    if not os.path.exists(CACHE_DIR):
        print(f"Not valid cache folder path: {CACHE_DIR}")
        CACHE_DIR = os.path.expanduser('~')
        print(f"Setting cache folder path to home directory: {CACHE_DIR}")

    return CACHE_DIR
