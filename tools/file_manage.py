import os

def add_prefix(file_path: str, prefix: str) -> str:
    """
    Adds a prefix to the file name while maintaining its directory and extension.

    :param file_path: The original file path.
    :param prefix: The prefix to add to the file name.
    :return: The new file path with the prefixed file name.
    """
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_path = os.path.join(dir_name, f"{prefix}{name}{ext}")
    return new_file_path