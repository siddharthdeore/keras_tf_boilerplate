import os

# for file handling converets relative path to absolute path
def absolute_file_path(rel_path):
    """
    Arguments
    ---------
    @param rel_path: relative filepath string 
    """
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, rel_path)