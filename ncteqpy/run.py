from __future__ import annotations

import argparse
import subprocess
from typing import Any, Callable, Sequence, cast

import simple_slurm as ss

from ncteqpy.settings import Settings


def intrange(arg: str) -> int | range:
    """Parse a string as an integer or a range of integers."""
    ints = arg.split("-")
    if len(ints) == 1:
        return int(ints[0])
    elif len(ints) == 2:
        return range(int(ints[0]), int(ints[1]) + 1)
    else:
        raise argparse.ArgumentTypeError(f"Invalid range {arg}")


class Runs:

    _num_runs: int
    _settings_input: dict[int, Settings] | Callable[[int], Settings]
    _settings: dict[int, Settings] | None = None

    _args: argparse.Namespace | None
    _indices: list[int | range]
    _indices_split: list[list[int | range]]
    _indices_flat: list[int]

    MAX_ARRAY_TASKS = 2000

    batch: ss.Slurm | list[ss.Slurm] | None = None

    def __init__(
        self,
        num_runs: int,
        description: str,
        settings: dict[int, Settings] | Callable[[int], Settings],
        slurm_batch_kwargs: dict[str, Any] | None = None,
        slurm_cmds: str | Sequence[str] | None = None,
        parse_args: bool = True,
        indices: list[int | range] | None = None,
    ) -> None:
        """Class to start nCTEQ++ runs or submit them to SLURM.

        Parameters
        ----------
        num_runs : int
            Maximum number of runs. Used for input validation
        description : str
            Description shown in the CLI
        settings : dict[int, Settings] | Callable[[int], Settings]
            Dictionary or function to get a `Settings` object for a given run index. The `Settings` object is written to a file when starting the run.
        slurm_batch_kwargs : dict[str, Any]
            Keyword arguments passed to `simple_slurm.Slurm`, resembling the options in a SLURM sbatch script. A minimal run should include `ntasks`, `cpus_per_task`, `partition`, `job_name`, and `time` (and maybe `mail-type` and `mail-user`).
        slurm_cmds : str | Sequence[str] | None
            Arguments passed to `simple_slurm.Slurm.add_cmd`
        parse_args : bool
            Whether to obtain the run indices from the CLI
        indices : list[int | range] | None
            Run indices to use. If `parse_args` is False and `indices` is None, all indices from 0 to `num_runs` are used. To pass `indices` that are not None, `parse_args` must be False.
        """

        self._num_runs = num_runs

        if parse_args:
            if indices is not None:
                raise ValueError("If parse_args is True, indices must be None")

            self.parse_args(num_runs, description)

        else:
            self._indices = [range(num_runs)] if indices is None else indices

        self._indices_split = [[]]
        for i in self._indices:
            current_max = len(self._indices_split) * self.MAX_ARRAY_TASKS
            prev_max = current_max - self.MAX_ARRAY_TASKS

            if isinstance(i, int):
                chunk, i_new = divmod(i, self.MAX_ARRAY_TASKS)

                if chunk >= len(self._indices_split):
                    self._indices_split.extend(
                        [] for _ in range(chunk - len(self._indices_split) + 1)
                    )

                self._indices_split[chunk].append(i_new)
            else:
                chunk_start, i_start_new = divmod(i.start, self.MAX_ARRAY_TASKS)
                chunk_stop, i_stop_new = divmod(i.stop, self.MAX_ARRAY_TASKS)

                if chunk_stop >= len(self._indices_split):
                    self._indices_split.extend(
                        [] for _ in range(chunk_stop - len(self._indices_split) + 1)
                    )

                # if the whole range is inside a single chunk
                if chunk_start == chunk_stop:
                    self._indices_split[chunk_start].append(
                        range(i_start_new, i_stop_new)
                    )
                # if the range spans multiple chunks
                else:
                    self._indices_split[chunk_start].append(
                        range(i_start_new, self.MAX_ARRAY_TASKS)
                    )
                    for j in range(chunk_start + 1, chunk_stop):
                        self._indices_split[chunk_start + j].append(
                            range(0, self.MAX_ARRAY_TASKS)
                        )
                    self._indices_split[chunk_stop].append(range(0, i_stop_new))

        self._indices_flat = [
            j for i in self._indices for j in (i if isinstance(i, range) else [i])
        ]
        self._settings_input = settings

        if slurm_batch_kwargs is not None and slurm_cmds is not None:
            self.init_slurm(slurm_batch_kwargs, slurm_cmds)

    def parse_args(self, num_runs: int, description: str) -> None:
        """Parse the command line arguments."""

        parser = argparse.ArgumentParser(description=description)
        subparsers = parser.add_subparsers(dest="command")

        command_settings = subparsers.add_parser(
            "settings", help="write the settings of a run to a file"
        )
        command_run = subparsers.add_parser("run", help="start a run")
        command_script = subparsers.add_parser(
            "script", help="print the SLURM sbatch script"
        )
        command_submit = subparsers.add_parser("submit", help="submit a run to SLURM")

        for subcommand in command_settings, command_run, command_script, command_submit:
            subcommand.add_argument(
                "i",
                type=intrange,
                # choices=range(0, len(fit_names)), # TODO: validation
                nargs="*",
                metavar="i",
                help=f"index of the run (0-{num_runs - 1})",
            )

        args = parser.parse_args()
        self._args = args

        self._indices = cast(list[int | range], args.i) if args.i else [range(num_runs)]

    def init_slurm(
        self, batch_kwargs: dict[str, Any], cmds: str | Sequence[str]
    ) -> None:
        """Create the SLURM batch script(s)."""

        if len(self.indices_split) > 1:
            self.batch = []
            for j, indices in enumerate(self.indices_split):
                self.batch.append(
                    ss.Slurm(
                        array=indices,
                        **batch_kwargs,
                    )
                )
                for cmd in cmds:
                    self.batch[-1].add_cmd(
                        cmd.replace(
                            ss.Slurm.SLURM_ARRAY_TASK_ID,  # pyright: ignore[reportAttributeAccessIssue]
                            f"$(( {ss.Slurm.SLURM_ARRAY_TASK_ID} + {j * self.MAX_ARRAY_TASKS} ))",  # pyright: ignore[reportAttributeAccessIssue]
                        )
                    )
        else:
            self.batch = ss.Slurm(
                array=self.indices,
                **batch_kwargs,
            )
            for cmd in cmds:
                self.batch.add_cmd(cmd)

    def run(self, i: int) -> None:
        """Run the fit with the settings at index `i`."""
        self.settings[i].write()
        subprocess.run(["./MainFit", self.settings[i].write_path])

    @property
    def script(self) -> str | list[str] | None:
        if isinstance(self.batch, ss.Slurm):
            return self.batch.script()
        elif isinstance(self.batch, list):
            return [batch.script() for batch in self.batch]
        else:
            return None

    def submit(self) -> None:
        """Submit the SLURM batch script."""
        if self.batch is not None:
            if isinstance(self.batch, list):
                for batch in self.batch:
                    batch.sbatch()
            elif isinstance(self.batch, ss.Slurm):
                self.batch.sbatch()
        else:
            raise ValueError(
                "SLURM batch script not initialized. Did you pass `slurm_batch_kwargs` and `slurm_cmds` to the constructor?"
            )

    def exec(self, i: int | None = None) -> None:
        """Execute the command specified in the CLI."""

        if self.args is None:
            raise ValueError("Need to parse arguments to execute the command")

        indices = self.indices_flat if i is None else [i]

        match self.args.command:
            case "script":
                if isinstance(self.script, list):
                    for s in self.script:
                        print(s)
                else:
                    print(self.script)

            case "submit":
                self.submit()

            case "settings":
                for j in indices:
                    self.settings[j].write()
                    print(f"Settings written to {self.settings[j].write_path}")

            case "run":
                for j in indices:
                    self.run(j)

            case _:
                raise ValueError("Invalid command")

    @property
    def num_runs(self) -> int:
        return self._num_runs

    @property
    def settings(self) -> dict[int, Settings]:
        if self._settings is None:
            if isinstance(self._settings_input, dict):
                self._settings = {i: self._settings_input[i] for i in self.indices_flat}
            else:
                self._settings = {i: self._settings_input(i) for i in self.indices_flat}

        return self._settings

    @property
    def args(self) -> argparse.Namespace | None:
        return self._args

    @property
    def indices(self) -> list[int | range]:
        return self._indices

    @property
    def indices_split(self) -> list[list[int | range]]:
        """`Runs.indices` split into chunks between two integer multiples of `Runs.MAX_ARRAY_TASKS`, i.e., chunk i contains the indices between `i * Runs.MAX_ARRAY_TASKS` and `(i + 1) * Runs.MAX_ARRAY_TASKS` modulo `Runs.MAX_ARRAY_TASKS`. This is needed for submitting multiple jobs when the array indices exceed the maximum allowed by SLURM."""
        return self._indices_split

    @property
    def indices_flat(self) -> list[int]:
        return self._indices_flat
