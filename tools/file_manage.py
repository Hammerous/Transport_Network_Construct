import os

def add_prefix(file_path: str, prefix: str) -> str:
    """
    Append a prefix to the file name while preserving its extension.

    Args:
        file_path (str): The path of the file, including its extension.
        prefix (str): The prefix to be added to the file name.

    Returns:
        str: The modified file name with the prefix inserted before the file basename.
    """
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_path = os.path.join(dir_name, f"{prefix}{name}{ext}")
    return new_file_path

def add_affix(file_path: str, affix: str) -> str:
    """
    Append an affix to the file name while preserving its extension.

    Args:
        file_path (str): The path of the file, including its extension.
        affix (str): The affix to be added to the file name.

    Returns:
        str: The modified file name with the affix inserted before the extension.
    """
    name, ext = os.path.splitext(file_path)
    return f"{name}{affix}{ext}"