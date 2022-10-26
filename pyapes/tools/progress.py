#!/usr/bin/env python
"""Check simulation progress.
"""
from rich.progress import BarColumn
from rich.progress import filesize
from rich.progress import Progress as RichProgress
from rich.progress import ProgressColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.text import Text
from tqdm.std import tqdm


class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""

    def __init__(self, unit_divisor=1000):
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        total = int(task.total)
        unit, suffix = filesize.pick_unit_and_suffix(total, [""], 1)

        precision = 0 if unit == 1 else 1
        return Text(
            f"{completed/unit:,.{precision}f}/{total/unit:,.{precision}f} {suffix}",
            style="progress.download",
        )


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_divisor=1000):
        self.unit = unit
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed

        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 0 if unit == 1 else 1

        return Text(
            f"{speed/unit:,.{precision}f} {suffix}{self.unit}/s",
            style="progress.data.speed",
        )


rich_progress_config = (
    "[bold yellow][progress.description]{task.description}[/bold yellow]"
    "[progress.percentage]{task.percentage:>4.0f}%",
    BarColumn(bar_width=30),
    FractionColumn(unit_divisor=1000),
    "[",
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    ",",
    RateColumn(unit="it", unit_divisor=1000),
    "]",
)


class RichPBar(tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""

    # TODO: @classmethod: write()?
    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.
        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs["gui"] = True
        kwargs["disable"] = False
        progress = kwargs.pop("progress", None)

        super().__init__(*args, **kwargs)

        d = self.format_dict

        # Allows custom config
        if progress is None:
            progress = rich_progress_config

        self._prog = RichProgress(*progress, transient=not self.leave)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        super().close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        self._prog.update(
            self._task_id, completed=self.n, description=self.desc
        )

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.
        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        self._prog.reset(self._task_id, total=total)
        super().reset(total=total)
