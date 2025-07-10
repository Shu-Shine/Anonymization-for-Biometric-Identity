from pathlib import Path

# helper function to increment path
def increment_path(base_dir, name='test'):
    base = Path(base_dir)
    path = base / name
    if not path.exists():
        return path.resolve()
    else:
        n = 2
        while (base / f"{name}{n}").exists():
            n += 1
        return (base / f"{name}{n}").resolve()