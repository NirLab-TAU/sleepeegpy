import functools
from loguru import logger


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


def plot_topomap(
    data,
    axis,
    info=None,
    shrink_cbar=0.6,
    topomap_args=None,
    cbar_args=None,
):
    from matplotlib.pyplot import colorbar
    from mne.viz import plot_topomap

    topomap_args = topomap_args or dict()
    topomap_args.setdefault("size", 5)
    topomap_args.setdefault("show", False)
    im, cn = plot_topomap(
        data,
        info,
        axes=axis,
        **topomap_args,
    )

    cbar_args = cbar_args or dict()
    cbar_args.setdefault("shrink", shrink_cbar)
    cbar_args.setdefault("orientation", "vertical")
    colorbar(
        im,
        ax=axis,
        **cbar_args,
    )
