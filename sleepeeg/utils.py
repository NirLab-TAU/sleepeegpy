import functools
from loguru import logger


def fix_mff_epochs_xml(path_to_mff: str, interval=(-5000, 5000), step=1000):
    """Tries to edit value of the last endTime
    tag from -5000 to 5000 with 1000 step.

    Args:
        path_to_mff: Path to mff file

    Returns:
        float | None: Delta between original endTime and fixed one.
            None if function haven't succeeded.
    """
    import re
    import shutil
    from pathlib import Path
    from itertools import chain
    from ast import literal_eval
    from mne import read_raw

    path_to_mff = Path(path_to_mff)
    values = range(interval[0], interval[1] + step, step)
    shutil.copyfile(
        path_to_mff / "epochs.xml",
        path_to_mff / "epochs_old.xml",
    )
    for value in values:
        with open(path_to_mff / "epochs.xml", "r+") as epochs_f:
            orig_xml = epochs_f.read()
            matches = re.findall(r"<endTime>(\d+)<\/endTime>", orig_xml)
            new_xml = orig_xml.replace(
                matches[-1], str(literal_eval(matches[-1]) + value)
            )
            epochs_f.seek(0)
            epochs_f.write(new_xml)
            epochs_f.truncate()
        try:
            read_raw(path_to_mff, verbose="ERROR")
            return value
        except:
            with open(path_to_mff / "epochs.xml", "w") as epochs_f:
                epochs_f.write(orig_xml)
    return None


def logger_wraps(*, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            logger_ = logger.opt(depth=1)
            logger_.log(
                level,
                f"Entering '{self.__class__.__name__}.{name}' (args={args}, kwargs={kwargs})",
            )
            result = func(self, *args, **kwargs)

            return result

        return wrapped

    return wrapper
