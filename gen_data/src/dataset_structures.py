import pathlib, shutil

def reset_dir(path: pathlib.Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
