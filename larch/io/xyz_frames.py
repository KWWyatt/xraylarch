"""
Utilities for extracting individual frames from multi-frame XYZ trajectory files.

The :func:`extract_xyz_frames` helper reads a standard XYZ trajectory that
contains multiple frames and writes selected frames to individual XYZ files.
The module also provides a small command-line interface::

    python -m larch.io.xyz_frames trajectory.xyz output_dir --frames 0 10 25
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class XYZFrame:
    """Representation of a single XYZ frame."""

    index: int
    natoms: int
    comment: str
    atoms: Sequence[str]


class XYZFrameFormatError(RuntimeError):
    """Raised when an XYZ file cannot be parsed."""


def _iter_xyz_frames(stream: Iterable[str]) -> Iterator[XYZFrame]:
    """
    Yield frames from a multi-frame XYZ stream.

    Parameters
    ----------
    stream
        An iterable returning lines of the XYZ file (typically a file object).

    Yields
    ------
    XYZFrame
        Parsed frame information.

    Raises
    ------
    XYZFrameFormatError
        If the XYZ file structure is invalid (missing lines or malformed atom
        count headers).
    """

    frame_index = 0
    line_iter = iter(stream)
    for line in line_iter:
        header = line.strip()
        if not header:
            # Skip blank lines between frames (non-standard but defensive).
            continue
        try:
            natoms = int(header)
        except ValueError as exc:
            raise XYZFrameFormatError(
                f"Expected integer atom count at frame {frame_index}, got {header!r}"
            ) from exc

        try:
            comment = next(line_iter)
        except StopIteration as exc:
            raise XYZFrameFormatError(
                f"Unexpected end of file after atom count for frame {frame_index}"
            ) from exc

        atoms: List[str] = []
        for atom_idx in range(natoms):
            try:
                atom_line = next(line_iter)
            except StopIteration as exc:
                raise XYZFrameFormatError(
                    f"Unexpected end of file reading atom {atom_idx} of frame {frame_index}"
                ) from exc
            atoms.append(atom_line.rstrip("\n"))

        yield XYZFrame(
            index=frame_index,
            natoms=natoms,
            comment=comment.rstrip("\n"),
            atoms=tuple(atoms),
        )
        frame_index += 1


def _write_xyz_frame(target: Path, frame: XYZFrame) -> None:
    """Write a single frame to ``target`` in XYZ format."""

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{frame.natoms}\n")
        handle.write(frame.comment)
        handle.write("\n")
        for atom_line in frame.atoms:
            handle.write(atom_line)
            handle.write("\n")


def _normalise_frame_numbers(
    frames: Sequence[int], one_based: bool
) -> List[int]:
    """
    Normalise user-supplied frame indices.

    Parameters
    ----------
    frames
        Sequence of frame numbers supplied by the user.
    one_based
        Whether to treat the input frame numbers as one-based indices.

    Returns
    -------
    list[int]
        Sorted list of unique, zero-based frame indices.
    """

    unique: Set[int] = set()
    for value in frames:
        if one_based:
            value -= 1
        if value < 0:
            raise ValueError(f"Frame indices must be non-negative, got {value}")
        unique.add(value)
    return sorted(unique)


def _load_frames_from_file(path: Path) -> List[int]:
    """Load frame numbers from an arbitrary text file."""

    text = path.read_text(encoding="utf-8")
    tokens = re.split(r"[,\s]+", text.strip())
    frames: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            frames.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid frame number {token!r} in {path}") from exc
    return frames


def extract_xyz_frames(
    source: Path | str,
    frame_numbers: Sequence[int],
    output_dir: Path | str,
    *,
    one_based: bool = False,
    zero_pad: int | None = None,
    prefix: str = "frame_",
    suffix: str = ".xyz",
    overwrite: bool = False,
) -> List[Path]:
    """
    Extract selected frames from a multi-frame XYZ trajectory.

    Parameters
    ----------
    source
        Path to the source XYZ trajectory.
    frame_numbers
        Iterable of frame numbers to extract. Duplicate indices are ignored.
    output_dir
        Destination directory. Created if it does not exist.
    one_based
        If ``True``, treat ``frame_numbers`` as one-based indices.
    zero_pad
        Custom zero-padding width for generated filenames. Defaults to the
        number of digits in the maximum frame index (minimum width of 3).
    prefix
        Filename prefix for generated files.
    suffix
        Filename suffix for generated files.
    overwrite
        If ``True``, existing files with the same name will be overwritten.

    Returns
    -------
    list[pathlib.Path]
        Paths to the written XYZ files.
    """

    src_path = Path(source)
    if not src_path.is_file():
        raise FileNotFoundError(f"XYZ trajectory not found: {src_path}")

    normalised_frames = _normalise_frame_numbers(frame_numbers, one_based=one_based)
    if not normalised_frames:
        raise ValueError("At least one frame number must be supplied.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_index = max(normalised_frames)
    pad_width = zero_pad if zero_pad is not None else max(3, len(str(max_index)))

    frames_to_find = set(normalised_frames)
    written_entries: List[Tuple[int, Path]] = []
    found_frames: Set[int] = set()

    with src_path.open("r", encoding="utf-8") as handle:
        for frame in _iter_xyz_frames(handle):
            if frame.index in frames_to_find:
                output_path = out_dir / f"{prefix}{frame.index:0{pad_width}d}{suffix}"
                if output_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite existing file {output_path}. "
                        "Use overwrite=True to force replacement."
                    )
                _write_xyz_frame(output_path, frame)
                written_entries.append((frame.index, output_path))
                found_frames.add(frame.index)
                if found_frames == frames_to_find:
                    break

    missing = frames_to_find - found_frames
    if missing:
        raise ValueError(
            f"Requested frames not present in {src_path}: {sorted(missing)}"
        )

    # Ensure returning paths are ordered by frame index.
    return [path for _, path in sorted(written_entries, key=lambda item: item[0])]


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Configure CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Extract selected frames from a multi-frame XYZ trajectory."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the multi-frame XYZ trajectory file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where extracted frames will be written.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="Frame numbers to extract (space-separated).",
    )
    parser.add_argument(
        "--frames-file",
        type=Path,
        help="Optional text file containing frame numbers (whitespace or comma separated).",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret supplied frame numbers as one-based indices.",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=None,
        help="Zero-padding width for output filenames (default: auto).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Prefix for output filenames (default: frame_).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".xyz",
        help="Suffix for output filenames (default: .xyz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    return parser.parse_args(argv)


def _derive_frame_numbers(args: argparse.Namespace) -> List[int]:
    """Combine frame numbers from CLI arguments."""

    frames: List[int] = []
    if args.frames:
        frames.extend(args.frames)
    if args.frames_file:
        frames.extend(_load_frames_from_file(args.frames_file))
    if not frames:
        raise ValueError(
            "No frame numbers provided. Use --frames or --frames-file to specify frames."
        )
    return frames


def main(argv: Sequence[str] | None = None) -> List[Path]:
    """CLI entry point."""

    args = _parse_arguments(argv)
    frame_numbers = _derive_frame_numbers(args)
    paths = extract_xyz_frames(
        args.source,
        frame_numbers,
        args.output_dir,
        one_based=args.one_based,
        zero_pad=args.zero_pad,
        prefix=args.prefix,
        suffix=args.suffix,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()

