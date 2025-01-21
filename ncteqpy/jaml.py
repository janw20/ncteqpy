from __future__ import annotations

import functools
import hashlib
import io
import os
import pathlib
import pickle
import sys
from typing import IO, Any, Iterable, Iterator, Sequence, TypeAlias, TypeVar, cast

import pandas as pd
import yaml
import yaml.scanner
from yaml.composer import Composer, ComposerError
from yaml.constructor import SafeConstructor
from yaml.events import *
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.parser import Parser
from yaml.reader import Reader
from yaml.resolver import Resolver
from yaml.scanner import Scanner

try:
    from yaml._yaml import CParser

    use_CParser = True
except ImportError:
    print(
        "Warning: using the slower Python parser instead of the C parser. Install libyaml for faster parsing."
    )
    use_CParser = False

# TODO: provide way to get TypedDict from JSON schema
YAMLLeaf: TypeAlias = str | int | float | bool | None
T = TypeVar("T")
_YAMLNode: TypeAlias = dict[str, T] | list[T]
YAMLType: TypeAlias = _YAMLNode["YAMLType"] | YAMLLeaf
YAMLNode: TypeAlias = _YAMLNode[YAMLType]

_PatternNode: TypeAlias = dict[str | None, T] | list[T]
PatternType: TypeAlias = _PatternNode["PatternType"] | YAMLLeaf
PatternNode: TypeAlias = _PatternNode[PatternType]


def nested_get(
    d: YAMLType,
    keys: Iterable[str | int],
    default: YAMLType = None,
    raise_keyerror: bool = False,
) -> YAMLType:
    for i, key in enumerate(keys):
        if (
            isinstance(d, dict)
            and isinstance(key, str)
            and key in d
            or isinstance(d, list)
            and isinstance(key, int)
        ):
            d = d[key]  # type: ignore[index] # this is correct but mypy says otherwise
        elif raise_keyerror:
            raise KeyError(f"Key {key} not found at level {i} of dictionary")
        else:
            return default
    return d


def nested_set(d: YAMLType, keys: Iterable[str | int], value: YAMLType) -> None:
    keys_iter = iter(keys)

    i = -1
    key_next = next(keys_iter)

    # if we want keys to be Iterable instead of Sequence we can't use a for loop since we need to treat the last key differently
    while True:
        try:
            key = key_next
            key_next = next(keys_iter)
            value_default_next: YAMLNode = [] if isinstance(key_next, int) else {}
            i += 1
        except StopIteration:
            break

        if isinstance(d, dict) and isinstance(key, str):
            if not key in d:
                d[key] = value_default_next.copy()

            d = d[key]
        elif isinstance(d, list) and isinstance(key, int):
            if key >= len(d):
                for j in range(key - len(d) + 1):
                    d.append(value_default_next.copy())
                # d.extend((key - len(d) + 1) * [value_default_next.copy()])
            d = d[key]
        else:
            raise ValueError(f"Could not index with key {key} at level {i}")

    if isinstance(d, dict) and isinstance(key, str):
        d[key] = value
    elif isinstance(d, list) and isinstance(key, int):
        d[key] = value
    else:
        raise ValueError(f"Could not index with key {key} at level {i}")


def nested_in(d: YAMLType, keys: Iterable[str | int]) -> bool:
    for key in keys:
        if isinstance(d, dict) and isinstance(key, str) and key in d:
            d = d[key]
        elif isinstance(d, list) and isinstance(key, int) and key < len(d):
            d = d[key]
        else:
            return False
    return True


def nested_iter(d: YAMLType) -> Iterator[tuple[str | int, ...]]:
    def helper(d: YAMLType, keys: list[str | int]) -> Iterator[tuple[str | int, ...]]:
        if isinstance(d, dict):
            for key, value in d.items():
                keys.append(key)
                yield from helper(value, keys)
                keys.pop()
        elif isinstance(d, list):
            for i, value in enumerate(d):
                keys.append(i)
                yield from helper(value, keys)
                keys.pop()
        else:
            yield tuple(keys)

    return helper(d, [])


def nested_items(d: YAMLType) -> Iterator[tuple[tuple[str | int, ...], YAMLLeaf]]:
    def helper(
        d: YAMLType, keys: list[str | int]
    ) -> Iterator[tuple[tuple[str | int, ...], YAMLLeaf]]:
        if isinstance(d, dict):
            for key, value in d.items():
                keys.append(key)
                yield from helper(value, keys)
                keys.pop()
        elif isinstance(d, list):
            for i, value in enumerate(d):
                keys.append(i)
                yield from helper(value, keys)
                keys.pop()
        else:
            yield (tuple(keys), d)

    return helper(d, [])


def to_str(y: YAMLType) -> str:
    if isinstance(y, str):
        res = f'"{y}"'
    elif isinstance(y, bool):
        res = str(y).lower()
    else:
        res = str(y)

    return res


class Pattern:
    _pattern: PatternType

    def __init__(self, pattern: PatternType) -> None:
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"Pattern(pattern={self.pattern})"

    @property
    def pattern(self) -> PatternType:
        return self._pattern

    @pattern.setter
    def pattern(self, value: PatternType, verify: bool = True) -> None:
        def _verify(p: PatternType) -> None:
            if p is None:
                return

            if isinstance(p, dict):
                for key in p:
                    if not isinstance(key, str) and key is not None:
                        raise ValueError(
                            f"Keys must be of type str or None, but found key '{key}' of type {type(key).__name__}"
                        )
                    else:
                        _verify(p[key])
            elif isinstance(p, list):
                if len(p) != 1:
                    raise ValueError(
                        f"Sequences must be of length 1, but found sequence of length {len(p)}:\n{p}"
                    )
                else:
                    _verify(p[0])
            else:
                raise ValueError(
                    f"Values must be a dict, a list or None, but found value '{p}' of type {type(p).__name__}"
                )

        if verify:
            _verify(value)
        self._pattern = value

    def matched_by(self, d: YAMLType) -> bool:
        for key in nested_iter(d):
            if not nested_in(d, key):  # FIXME: match None as key in pattern
                return False
        return True

    def difference(self, d: PatternType) -> Pattern:
        diff: PatternType = {}  # TODO: case self.pattern is not a dict
        for key in nested_iter(self.pattern):
            print(key)
            if not nested_in(d, key):
                print("not in pattern")
                nested_set(diff, key, None)
        return Pattern(diff)


class YAMLWrapper:

    paths: list[pathlib.Path]
    """Path to the wrapped YAML file(s) or to the directory containing the wrapped YAML files"""
    cache_path: pathlib.Path
    """Path to the pickle cache directory"""
    retain_yaml: bool
    """True if the raw YAML data should be stored in `yaml` after loading, otherwise False"""

    _mtimes: list[float]
    """Timestamp of the time the wrapped YAML file(s) pointed to by `path` was/were modified. Used for determining if the wrapped YAML file(s) changed"""
    _yaml: YAMLType = None
    """Wrapped raw YAML, only different from `None` if `retain_yaml` is True"""

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        cache_path: str | os.PathLike[str] = pathlib.Path("./.jaml_cache/"),
        retain_yaml: bool = False,
    ) -> None:
        """Wrapper class of a YAML file with convenience functions and pickling functionality.

        Parameters
        ----------
        path : str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
            Path to YAML file(s) or a directory (possibly recursively) containing YAML files
        retain_yaml : bool, optional
            True if the raw YAML data should be stored in `yaml` after loading, otherwise False. By default False
        """
        if isinstance(paths, (str, os.PathLike)):
            self.paths = [pathlib.Path(paths)]
        else:
            self.paths = [pathlib.Path(p) for p in paths]

        for p in self.paths:
            p.resolve(strict=True)  # raise if path doesn't exist
        self._mtimes = self._path_mtimes()
        self.cache_path = pathlib.Path(cache_path)
        self.retain_yaml = retain_yaml
        if retain_yaml:
            self._yaml = {}

    @property
    def yaml(self) -> YAMLType:
        """Wrapped raw YAML, only different from `None` if `retain_yaml` is True"""
        return self._yaml

    def _load_yaml(
        self, pattern: Pattern | None
    ) -> YAMLType | list[tuple[pathlib.Path, YAMLType]]:
        """Loads the YAML file(s) pointed to by `path`. If a `pattern` is given, the YAML file is loaded with `jaml.safe_load`, otherwise PyYAMLs `safe_load` function is used

        Parameters
        ----------
        pattern : Pattern | None
            Pattern describing which fields in the wrapped YAML file(s) should be loaded

        Returns
        -------
        YAMLType | list[tuple[pathlib.Path, YAMLType]]
            The loaded YAML data if `self.path` points to a YAML file or a directory that contains just one file, or a list of tuples containing the path to the YAMl file and the loaded YAML data if `self.path` points to a directory

        Raises
        ------
        NotImplementedError
            Retaining the raw YAML data is not yet implemented if the root node is not a mapping/dict, so an exception is raised if this is tried
        """

        def load_yaml_file(path: pathlib.Path, pattern: Pattern | None) -> YAMLType:
            with open(path) as f:
                if pattern is None:
                    return cast(YAMLType, yaml.safe_load(f))
                else:
                    if self.retain_yaml:
                        diff = pattern.difference(self.yaml)
                        new_yaml: YAMLType = safe_load(f, diff.pattern)

                        if isinstance(self._yaml, dict) and isinstance(new_yaml, dict):
                            self._yaml = self._yaml | new_yaml
                        else:
                            raise NotImplementedError(
                                "Retaining not implemented for non-dictionaries"
                            )

                        return new_yaml
                    else:
                        return cast(YAMLType, safe_load(f, pattern.pattern))

        res: list[tuple[pathlib.Path, YAMLType]] = []
        for path in self.paths:
            if path.is_dir():
                for p in path.glob("**/*.yaml"):
                    try:
                        res.append((p, load_yaml_file(p, pattern)))
                    except yaml.scanner.ScannerError:
                        pass
            elif path.is_file():
                res.append((path, load_yaml_file(path, pattern)))
            else:
                raise ValueError("All paths in self.paths must be a file or directory")

        return res[0][1] if len(res) == 1 else res

    def _path_mtimes(self) -> list[float]:
        res = []

        for path in self.paths:
            if path.is_file():
                res.append(path.stat().st_mtime)
            elif path.is_dir():
                res.append(
                    functools.reduce(
                        lambda t, p: max(t, p.stat().st_mtime),
                        path.glob("**/*.yaml"),
                        0.0,
                    )
                )
            else:
                raise ValueError("All paths in self.paths must be a file or directory")

        return res

    def _pickle_path(self, name: str) -> pathlib.Path:
        """Path to the cached pickle file

        Parameters
        ----------
        name : str
            Name of the pickle file without extension

        Returns
        -------
        pathlib.Path
            Path to the pickle file (may not exist)
        """

        # Python version in the pickle path to support different versions and hash of the path to support caching different files simultaneously (resolve first so relative and absolute paths give the same hash)
        return pathlib.Path(
            self.cache_path
            / f"{sys.version_info.major}.{sys.version_info.minor}/{hashlib.sha256(str(self.paths).encode(), usedforsecurity=False).hexdigest()}/{name}.pkl"
        )

    def _pickle(self, variable: object, name: str) -> None:
        """Pickle a `variable` and store it in the cache directory

        Parameters
        ----------
        variable : object
            Variable to be pickled
        name : str
            Name of the pickle file without extension
        """
        self._pickle_path(name).parent.mkdir(parents=True, exist_ok=True)
        (self.cache_path / ".gitignore").write_text(
            "# Automatically created by jaml\n*\n"
        )
        if isinstance(variable, pd.DataFrame):
            variable.to_pickle(self._pickle_path(name))
        else:
            with open(self._pickle_path(name), "wb") as f:
                pickle.dump(variable, f)

    def _has_valid_pickle(self, name: str) -> bool:
        """Checks if the pickle file associated with `name` exists and the wrapped YAML file pointed to by `path` was not modified in the meantime

        Parameters
        ----------
        name : str
            Name of the pickle file without extension

        Returns
        -------
        bool
            True if a valid pickle file exists, False otherwise
        """
        return self._pickle_path(name).is_file() and self._pickle_path(
            name
        ).stat().st_mtime > min(self._path_mtimes())

    def _unpickle(self, name: str) -> object | None:
        """Unpickle a variable stored in the cache directory

        Parameters
        ----------
        name : str
            Name of the pickle file without extension

        Returns
        -------
        object | None
            Unpickled variable if the pickle file was valid, `None` otherwise
        """
        if self._has_valid_pickle(name):
            try:
                return cast(object, pd.read_pickle(self._pickle_path(name)))
            except:
                return None
        else:
            return None

    def _yaml_changed(self) -> bool:
        """Check if the wrapped YAML file pointed to by `path` was modified during runtime

        Returns
        -------
        bool
            True if the wrapped YAML file was modified, False otherwise
        """
        prev_mtime = self._mtimes
        self._mtimes = self._path_mtimes()
        return prev_mtime < self._mtimes


class PatternComposer(Composer):

    def __init__(self, pattern: dict) -> None:
        super().__init__()
        self.pattern = pattern

    # changes to Composer.compose_node: pass on the pattern
    def compose_document(self) -> Node | None:
        # Drop the DOCUMENT-START event.
        self.get_event()

        # Compose the root node.
        node = self.compose_node(None, None, self.pattern)

        # Drop the DOCUMENT-END event.
        self.get_event()

        self.anchors = {}
        return node

    # changes to Composer.compose_node: pass on the pattern
    def compose_node(
        self,
        parent: Node | None,
        index: ScalarNode | int | None,
        pattern: dict | None = None,
    ) -> Node | None:
        if self.check_event(AliasEvent):
            event = self.get_event()
            anchor = event.anchor
            if anchor not in self.anchors:
                raise ComposerError(
                    None, None, "found undefined alias %r" % anchor, event.start_mark
                )
            return self.anchors[anchor]
        event = self.peek_event()
        anchor = event.anchor
        if anchor is not None:
            if anchor in self.anchors:
                raise ComposerError(
                    "found duplicate anchor %r; first occurrence" % anchor,
                    self.anchors[anchor].start_mark,
                    "second occurrence",
                    event.start_mark,
                )
        self.descend_resolver(parent, index)
        if self.check_event(ScalarEvent):
            node = self.compose_scalar_node(anchor)
        elif self.check_event(SequenceStartEvent):
            node = self.compose_sequence_node(anchor, pattern)
        elif self.check_event(MappingStartEvent):
            node = self.compose_mapping_node(anchor, pattern)
        self.ascend_resolver()
        return node

    # changes to Composer.compose_sequence_node: if pattern is None, pass None (i.e. default argument, otherwise pass on the dict in the list)
    def compose_sequence_node(
        self, anchor: dict[Any, Node], pattern: dict = None
    ) -> SequenceNode:
        # print("seq", pattern)
        start_event = self.get_event()
        tag = start_event.tag
        if tag is None or tag == "!":
            tag = self.resolve(SequenceNode, None, start_event.implicit)
        node = SequenceNode(
            tag, [], start_event.start_mark, None, flow_style=start_event.flow_style
        )
        if anchor is not None:
            self.anchors[anchor] = node
        index = 0
        while not self.check_event(SequenceEndEvent):
            if pattern is None:
                node.value.append(self.compose_node(node, index))
            else:
                node.value.append(self.compose_node(node, index, pattern[0]))
            # print("seq i", node.value[-1])
            index += 1
        end_event = self.get_event()
        node.end_mark = end_event.end_mark
        return node

    def compose_mapping_node(
        self, anchor: dict[Any, Node], pattern: dict = None
    ) -> MappingNode:
        # print("map", pattern)
        start_event = self.get_event()
        tag = start_event.tag
        if tag is None or tag == "!":
            tag = self.resolve(MappingNode, None, start_event.implicit)
        node = MappingNode(
            tag, [], start_event.start_mark, None, flow_style=start_event.flow_style
        )
        if anchor is not None:
            self.anchors[anchor] = node
        while not self.check_event(MappingEndEvent):
            # item_key is always a ScalarNode
            item_key = self.compose_node(node, None)
            if isinstance(item_key, ScalarNode):
                # if pattern is None, compose everything that is on a lower level
                if pattern is None:
                    item_value = self.compose_node(node, item_key)
                    node.value.append((item_key, item_value))
                # if pattern is not None and the key is in the pattern, compose the value
                elif item_key.value in pattern or None in pattern:
                    item_value = self.compose_node(
                        node,
                        item_key,
                        pattern[item_key.value if item_key.value in pattern else None],
                    )
                    node.value.append((item_key, item_value))
                # if the key is not in the pattern, skip the rest of the value node, i.e. one event for a scalar ...
                elif self.check_event(ScalarEvent):
                    self.get_event()
                # ... or until the lower level mappings are through
                elif self.check_event(MappingStartEvent):
                    level = 1
                    self.get_event()
                    while not level == 0:
                        if self.check_event(MappingStartEvent):
                            level += 1
                        elif self.check_event(MappingEndEvent):
                            level -= 1
                        self.get_event()
                # or until the lower level sequences are through
                elif self.check_event(SequenceStartEvent):
                    level = 1
                    self.get_event()
                    while not level == 0:
                        if self.check_event(SequenceStartEvent):
                            level += 1
                        elif self.check_event(SequenceEndEvent):
                            level -= 1
                        self.get_event()
                else:
                    raise ComposerError(
                        None,
                        None,
                        f"expected a ScalarEvent after MappingStartEvent but got {self.peek_event()}",
                    )
            else:
                raise ComposerError(
                    None,
                    None,
                    f"expected a ScalarEvent after MappingStartEvent but got {self.peek_event()}",
                )
        end_event = self.get_event()
        node.end_mark = end_event.end_mark
        return node


if use_CParser:

    class PatternCSafeLoader(PatternComposer, CParser, SafeConstructor, Resolver):

        def __init__(self, stream: IO, pattern: dict) -> None:
            PatternComposer.__init__(self, pattern)
            CParser.__init__(self, stream)
            SafeConstructor.__init__(self)
            Resolver.__init__(self)


class PatternSafeLoader(
    Reader, Scanner, Parser, PatternComposer, SafeConstructor, Resolver
):

    def __init__(self, stream: IO, pattern: dict) -> None:
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        PatternComposer.__init__(self, pattern)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)


def safe_load(stream: IO, pattern: dict) -> Any:
    if use_CParser:
        loader = PatternCSafeLoader(stream, pattern)
    else:
        loader = PatternSafeLoader(stream, pattern)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


# TODO finish
def yaml_get_node_names(file: io.FileIO, level: int) -> list[str]:
    loader = CSafeLoader2(file)

    result = []
    level = 0
    try:
        while loader.check_event():
            event = loader.get_event()
            if isinstance(event, yaml.MappingStartEvent):
                level += 1
            elif isinstance(event, yaml.MappingEndEvent):
                level -= 1
            elif isinstance(event, yaml.ScalarEvent):
                if level == 1:
                    result.append(event.value)
    finally:
        loader.dispose()

    return result
