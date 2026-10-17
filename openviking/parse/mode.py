# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared add-resource parsing modes."""

from dataclasses import dataclass
from enum import Enum
from pathlib import PureWindowsPath
from typing import Literal, TypeAlias

from openviking_cli.exceptions import InvalidArgumentError


class ParseMode(str, Enum):
    """Controls whether parsed Markdown is split into multiple files."""

    DEFAULT = "default"
    NO_SPLIT = "no_split"


ParseModeInput: TypeAlias = ParseMode | Literal["default", "no_split"]


def normalize_parse_mode(value: str | ParseMode) -> ParseMode:
    """Return a validated resource parsing mode."""
    try:
        return ParseMode(value)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError("parse_mode must be one of: default, no_split.") from exc


@dataclass(frozen=True)
class PathParsePolicy:
    """Native modes for import-relative prefixes, longest matching prefix first."""

    default: ParseMode
    overrides: tuple[tuple[str, ParseMode], ...]

    @classmethod
    def from_options(cls, split_content: bool, overrides: object) -> "PathParsePolicy":
        if not isinstance(overrides, dict):
            raise InvalidArgumentError("args.parse_mode_overrides must be an object.")
        entries = []
        for prefix, mode in overrides.items():
            if (
                not isinstance(prefix, str)
                or not prefix
                or prefix.startswith("/")
                or PureWindowsPath(prefix).drive
                or "\\" in prefix
                or any(part in {"", ".", ".."} for part in prefix.split("/"))
                or any(ord(char) < 32 or ord(char) == 127 for char in prefix)
            ):
                raise InvalidArgumentError(
                    "args.parse_mode_overrides prefixes must be nonempty relative POSIX paths "
                    "without empty, '.' or '..' segments, drives, backslashes or control characters."
                )
            try:
                entries.append((prefix, normalize_parse_mode(mode)))
            except InvalidArgumentError as exc:
                raise InvalidArgumentError(
                    "args.parse_mode_overrides modes must be one of: default, no_split."
                ) from exc
        return cls(
            ParseMode.DEFAULT if split_content else ParseMode.NO_SPLIT,
            tuple(sorted(entries, key=lambda entry: len(entry[0]), reverse=True)),
        )

    def split_content(self, relative_path: str) -> bool:
        for prefix, mode in self.overrides:
            if relative_path == prefix or relative_path.startswith(prefix + "/"):
                return mode is ParseMode.DEFAULT
        return self.default is ParseMode.DEFAULT
