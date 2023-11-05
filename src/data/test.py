from pathlib import Path


def split_path_filename_ext(path):
    path_obj = Path(path)
    directory = path_obj.parent
    file_name = path_obj.stem
    file_extension = path_obj.suffix
    return str(directory), str(file_name), str(file_extension)


print(split_path_filename_ext("/path/to/your/file.txt"))
