from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, TypedDict, cast

import sympy as sp

import ncteqpy.jaml as jaml
from ncteqpy.cuts import Cuts
from ncteqpy.jaml import YAMLType
from ncteqpy.kinematic_variables import label_to_kinvar
from ncteqpy.labels import kinvars_yaml_to_py

_SETTINGS_FILE_HEADER = (
    "# !! File is writable by ncteqpy (changing this line will make it non-writable) !!"
)


class Settings(jaml.YAMLWrapper):
    _datasets: list[Path] | None = None
    _all_parameters: list[str] | None = None
    _open_parameters: list[str] | None = None
    _closed_parameters: list[str] | None = None
    _cuts: Cuts | None = None

    write_path: os.PathLike[str]

    yaml_overwrites: dict[str, dict[str, YAMLType]] = {}
    """YAML fields that will be overwritten when calling `Settings.write`. Values can also be added by calling `Settings.add_yaml_overwrite`."""

    yaml_comments: set[int] = set()
    """Lines that will be commented out when calling `Setings.write` (or uncommented if they are already commented out). The line numbers refer to the YAML file at `Settings.path`. Values can also be added by calling `Settings.yaml_add_comment`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        write_path: str | os.PathLike[str] | None = None,
        yaml_overwrites: dict[str, dict[str, YAMLType]] = {},
        yaml_comments: set[int] = set(),
        cache_path: str | os.PathLike[str] = Path("./.jaml_cache/"),
        retain_yaml: bool = False,
    ) -> None:
        """Wrapper class of a YAML settings file

        Parameters
        ----------
        path : str | os.PathLike[str]
            Path to the YAML settings file
        yaml_overwrites : dict[str, dict[str, YAMLType]]
            YAML fields that will be overwritten when calling `Settings.write`
        yaml_comments : set[int]
            Lines that will be commented out when calling `Setings.write` (or uncommented if they are already commented out)
        cache_path : str | os.PathLike[str]
            Path to the directory where cached variables should be stored. By default "./.jaml_cache"
        retain_yaml : bool, optional
            True if the raw YAML data should be stored in `Settings.yaml` after loading, otherwise False. By default False
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError("Please pass the file path to a YAML settings file")

        super().__init__(path, cache_path, retain_yaml)

        if write_path is not None:
            self.write_path = Path(write_path)
        else:
            self.write_path = path

        self.yaml_overwrites = yaml_overwrites
        self.yaml_comments = yaml_comments

    def add_yaml_overwrite(self, key: list[str], value: YAMLType) -> None:
        """Add a YAML value to `Settings.yaml_overwrites`, which will overwrite the previous one when `Settings.write` is called.

        Parameters
        ----------
        keys : list[str]
            Key to the value that will be overwritten. Must be of length 2
        value : YAMLType
            New value
        """
        if len(key) != 2:
            raise ValueError("Key must have length 2")
        jaml.nested_set(self.yaml_overwrites, key, value)

    def add_yaml_comments(self, lines: int | Iterable[int]) -> None:
        """Add line numbers to `Settings.yaml_comments`, which will be either commented or uncommented when `Settings.write` is called.

        Parameters
        ----------
        lines : int | Iterable[int]
            Line number(s) that will be commented or uncommented.
        """
        if not isinstance(lines, Iterable):
            lines = [lines]

        self.yaml_comments.update(lines)

    # FIXME: handle multiline fields
    # TODO: add commenting functionality
    def write(self, path: str | os.PathLike[str] | None = None) -> None:
        """Write out a new version of the settings file, first toggling commentation of the lines in `Settings.yaml_comments` and then overwriting values contained in `Settings.yaml_overwrites`. This preserves whitespace and comments.

        Parameters
        ----------
        path : str | os.PathLike[str] | None, optional
            The path where to write the settings file, by default None. If this is None, the settings file given is written to `Settings.write_path`. Note that the latter points to the original file if `write_path` is not passed to the `Run` constructor. Existing files can only be overwritten if the first line is given by

            `# !! File is writable by ncteqpy (changing this line will make it non-writable) !!`

        Raises
        ------
        PermissionError
            If an existing file that is to be overwritten does not contain the header mentioned above as the first line.
        SyntaxError
            If the settings file at `self.path` cannot be read.
        """

        # paths is only initialized with one file in __init__
        assert len(self.paths) == 1

        # if path is None, we try to overwrite self.paths[0] if it includes _SETTINGS_FILE_HEADER
        path = Path(self.write_path) if path is None else Path(path)

        lines = self.paths[0].read_text().splitlines()

        offset_lineno = 0
        if not lines[0] == _SETTINGS_FILE_HEADER:
            lines.insert(0, _SETTINGS_FILE_HEADER)
            offset_lineno = 1

        current_tags: list[str] = []
        current_cols: list[str] = []
        i_level = 0
        for i, line in enumerate(lines):
            # offset by offset_lineno (to correct adding the header as the first line) and by 1 since line numbers start at 1
            lineno = i - offset_lineno + 1

            if not line or line.isspace():
                continue

            if lineno in self.yaml_comments:
                line_lstrip = line.lstrip()
                whitespace = line[: -len(line_lstrip)]
                if line_lstrip:
                    if line_lstrip.startswith("# "):
                        line_lstrip = line_lstrip[2:]
                    elif line_lstrip.startswith("#"):
                        line_lstrip = line_lstrip[1:]
                    else:
                        line_lstrip = "# " + line_lstrip

                    # we set `line` in case it is modified below and `lines[i]` in case of a continue below
                    line = whitespace + line_lstrip
                    lines[i] = line

            # first take care of comments, since colons in them don't identify tags
            p = line.rpartition("#")
            if p[1]:
                if p[0].isspace():
                    continue
                comment = p[1] + p[2]
                line = line.removesuffix(comment)
            else:
                comment = ""

            # we look for the leftmost colon that is followed by newline (p[2] == "") or whitespace
            # FIXME? this doesn't cover the case where a line is just a string that contains a colon followed by whitespace (but this is not correct YAML syntax anyway)
            p = line.partition(":")
            while p[2] and not p[2][0].isspace():
                p = line.partition(":")
            # print(p)

            # case that there is no colon in the current line
            if not p[1]:
                continue

            # case that the only colon is part of a string
            if p[2]:
                if not p[2][0].isspace():
                    raise SyntaxError(
                        f"Only colon in line {lineno} is part of string, are you missing a space?"
                    )

            tag = p[0].lstrip()
            leading_whitespace = p[0][: -len(tag)]
            try:
                i_level = current_cols.index(leading_whitespace) if current_cols else 0
            except ValueError:
                i_level += 1

            # limit to 2 levels, since the settings files only go 2 levels deep
            if i_level < 2:
                current_cols = current_cols[:i_level]
                current_cols.append(
                    (current_cols[-1] + leading_whitespace)
                    if current_cols
                    else leading_whitespace
                )
                current_tags = current_tags[:i_level]
                current_tags.append(tag)

            if len(current_tags) == 2 and jaml.nested_in(
                self.yaml_overwrites, current_tags
            ):
                val = line.removeprefix(leading_whitespace + tag + ":")

                new_val = jaml.to_str(
                    jaml.nested_get(self.yaml_overwrites, current_tags)
                )
                new_val = val.replace(val.strip(), new_val)
                lines[i] = leading_whitespace + tag + ":" + new_val + comment

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))

    @property
    def datasets(self) -> list[Path]:
        if self._datasets is None or self._yaml_changed():
            pattern = jaml.Pattern({"DataSets": {"DataFiles": None}})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            datasets_str = cast(
                list[str],
                jaml.nested_get(yaml, ["DataSets", "DataFiles"], raise_keyerror=True),
            )

            self._datasets = [Path(s) for s in datasets_str]

        return self._datasets

    @property
    def cuts(self) -> Cuts:
        if self._cuts is None or self._yaml_changed():
            pattern = jaml.Pattern({"Cuts": None})
            yaml = self._load_yaml(pattern)

            assert isinstance(yaml, dict)

            yaml_to_cut: dict[str, type[sp.Rel]] = {
                "MINCUT": sp.StrictGreaterThan,
                "MAXCUT": sp.StrictLessThan,
                "MINCUTEQ": sp.GreaterThan,
                "MAXCUTEQ": sp.StrictLessThan,  # bug in ncteqpp-2.0
            }

            class ByType(TypedDict):
                Type: str
                Value: float
                KinVar: str
                Cut: str

            by_type: dict[str, sp.Rel] = {}

            for cut_yaml in cast(dict[str, dict[str, list[ByType]]], yaml)["Cuts"][
                "ByType"
            ]:
                cut = yaml_to_cut[cut_yaml["Cut"]](
                    label_to_kinvar[kinvars_yaml_to_py[cut_yaml["KinVar"]]],
                    cut_yaml["Value"],
                )
                t = cut_yaml["Type"]
                by_type[t] = cut if t not in by_type else by_type[t] & cut

            class ByID(TypedDict):
                IDs: list[int]
                Value: float
                KinVar: str
                Cut: str

            by_id: dict[int, sp.Rel] = {}

            for cut_yaml in cast(dict[str, dict[str, list[ByID]]], yaml)["Cuts"][
                "ByID"
            ]:
                cut = yaml_to_cut[cut_yaml["Cut"]](
                    label_to_kinvar[kinvars_yaml_to_py[cut_yaml["KinVar"]]],
                    cut_yaml["Value"],
                )
                for id in cut_yaml["IDs"]:
                    by_id[id] = cut if id not in by_id else by_id[id] & cut

            self._cuts = Cuts(by_type, by_id)

        return self._cuts

    def _read_parameters(self) -> None:
        # read all lines inside the FitParams tag
        tag = "FitParams"
        with open(self.paths[0], "r") as f:

            inside_tag = False
            indentation = -1
            lines = []
            for line in f:
                line_stripped = line.lstrip()

                if not inside_tag:

                    if line_stripped.startswith(tag + ":"):
                        inside_tag = True
                        indentation = len(line) - len(line_stripped)

                else:
                    line_without_indentation = line[indentation:]

                    # break when encountering a non-comment line containing a tag that is not at a higher or at the same level of indentation as the FitParams tag
                    is_comment = line.lstrip().startswith("#")
                    is_empty = line.isspace()
                    is_same_level = len(line_without_indentation) == len(line.lstrip())
                    is_higher_level = not line[:indentation].isspace()

                    if (
                        not is_comment
                        and not is_empty
                        and (is_higher_level or is_same_level)
                    ):
                        break

                    lines.append(line)

        self._all_parameters = []
        self._open_parameters = []
        self._closed_parameters = []

        pattern = re.compile(
            r"(#*)\s*\[(\w+)\s*,\s*\[[\-0-9.,\s]*\]\s*,\s*(?:FREE|BOUNDED)\]"
        )  # pain
        for line in lines:
            for match in pattern.finditer(line):
                hash = match.group(1)
                param = match.group(2)

                self._all_parameters.append(param)
                if hash:
                    self._closed_parameters.append(param)
                else:
                    self._open_parameters.append(param)

        assert set(self._all_parameters) == set(
            self._open_parameters + self._closed_parameters
        )

    @property
    def all_parameters(self) -> list[str]:
        if self._all_parameters is None or self._yaml_changed():
            self._read_parameters()

        assert self._all_parameters is not None

        return self._all_parameters

    @property
    def open_parameters(self) -> list[str]:
        if self._open_parameters is None or self._yaml_changed():
            self._read_parameters()

        assert self._open_parameters is not None

        return self._open_parameters

    @property
    def closed_parameters(self) -> list[str]:
        if self._closed_parameters is None or self._yaml_changed():
            self._read_parameters()

        assert self._closed_parameters is not None

        return self._closed_parameters
