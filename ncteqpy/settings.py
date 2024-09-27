from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import ncteqpy.jaml as nc_jaml

_SETTINGS_FILE_HEADER = (
    "# !! File is writable by ncteqpy (changing this line will make it non-writable) !!"
)


class Settings(nc_jaml.YAMLWrapper):
    _datasets: list[Path] | None = None
    _yaml_overwrites: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str] = Path("./.jaml_cache/"),
        retain_yaml: bool = False,
    ) -> None:
        """Wrapper class of a YAML settings file

        Parameters
        ----------
        path : str | os.PathLike[str]
            Path to the YAML settings file
        cache_path : str | os.PathLike[str]
            Path to the directory where cached variables should be stored. By default ./.jaml_cache
        retain_yaml : bool, optional
            True if the raw YAML data should be stored in `yaml` after loading, otherwise False. By default False
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError("Please pass the file path to a YAML settings file")

        super().__init__(path, cache_path, retain_yaml)

    def add_overwrite(self, key: list[str], value: nc_jaml.YAMLType) -> None:
        """Add a YAML value that is overwritten when `Settings.write` is called.

        Parameters
        ----------
        keys : list[str]
            Key to the value that will be overwritten. Must be of length 2
        value : nc_jaml.YAMLType
            New value
        """
        if len(key) != 2:
            raise ValueError("Key must have length 2")
        nc_jaml.nested_set(self._yaml_overwrites, key, value)

    # FIXME: handle multiline fields
    # TODO: add commenting functionality
    def write(self, path: str | os.PathLike[str] | None = None) -> None:
        """Write out a new version of the settings file containing overwritten values set by `Settings.add_overwrite`. This preserves whitespace and comments.

        Parameters
        ----------
        path : str | os.PathLike[str] | None, optional
            The path where to write the settings file, by default None. If this is None, the original settings file given by `self.path` is overwritten. Note that an existing file can only be overwritten if the first line is given by

            `# !! File is writable by ncteqpy (changing this line will make it non-writable) !!`

        Raises
        ------
        PermissionError
            If an existing file that is to be overwritten does not contain the header mentioned above as the first line.
        SyntaxError
            If the settings file at `self.path` cannot be read.
        """
        if path is None:
            assert len(self.paths) == 1
            path = self.paths[0]
        else:
            path = Path(path)

        offset_lineno = 0

        if path.exists():
            lines = path.read_text().splitlines()

            if not lines[0] == _SETTINGS_FILE_HEADER:
                raise PermissionError(
                    f"File at {path} is not writable by ncteqpy. If you intend write to it anyway, add the following as the first line of the file:\n{_SETTINGS_FILE_HEADER}"
                )
        else:
            # paths is only initialized with one file in __init__
            assert len(self.paths) == 1

            lines = self.paths[0].read_text().splitlines()

            if not lines[0] == _SETTINGS_FILE_HEADER:
                lines.insert(0, _SETTINGS_FILE_HEADER)
                offset_lineno = 1

        current_tags: list[str] = []
        current_cols: list[str] = []
        i_level = 0
        for i, line in enumerate(lines):
            # if skip_to_next and line.startswith("".join(current_cols)):

            if not line or line.isspace():
                continue

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
                        f"Only colon in line {i - offset_lineno} is part of string, are you missing a space?"
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

            if len(current_tags) == 2 and nc_jaml.nested_in(
                self._yaml_overwrites, current_tags
            ):
                # print([current_cols, current_tags])
                val = line.removeprefix(leading_whitespace + tag + ":")
                # print(val)
                # print(val.strip())
                new_val = str(
                    nc_jaml.nested_get(self._yaml_overwrites, current_tags)
                ).replace("'", '"')
                if new_val in ("True", "False"):
                    new_val = new_val.lower()
                new_val = val.replace(val.strip(), new_val)
                lines[i] = leading_whitespace + tag + ":" + new_val + comment

        path.write_text("\n".join(lines))

    @property
    def datasets(self) -> list[Path]:
        if self._datasets is None or self._yaml_changed():
            pattern = nc_jaml.Pattern({"DataSets": {"DataFiles": None}})
            yaml = self._load_yaml(pattern)

            datasets_str = cast(
                list[str],
                nc_jaml.nested_get(
                    yaml, ["DataSets", "DataFiles"], raise_keyerror=True
                ),
            )

            self._datasets = [Path(s) for s in datasets_str]

        return self._datasets
