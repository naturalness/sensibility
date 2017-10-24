#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2017 Eddie Antonio Santos <easantos@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
from os import PathLike
from pathlib import Path

assert os.symlink in os.supports_dir_fd


def symlink_within_dir(
        *, directory: PathLike, source: PathLike, target: Path
) -> None:
    """
    Creates a symbolic link (symlink) relative to a directory.
    """
    # Clobber the existing symlink.
    if target.exists():
        target.unlink()
    fd = os.open(os.fspath(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.symlink(source, target, dir_fd=fd)  # type: ignore
    except FileExistsError:
        pass
    finally:
        os.close(fd)


def test_symlink() -> None:
    import uuid
    import tempfile
    from pathlib import Path

    # Create some unique content.
    unique_contents = f"{uuid.uuid1()}\n"
    with tempfile.TemporaryDirectory() as tempdirname:
        directory = Path(tempdirname)
        with open(directory / 'source', 'w') as f:
            f.write(unique_contents)

        assert not (directory / 'target').exists()
        symlink_within_dir(directory=directory,
                           source=directory / 'source',
                           target=directory / 'target')

        # Check to see if it's got the unique contents.
        assert (directory / 'target').is_symlink()
        with open(directory / 'target') as f:
            assert f.read() == unique_contents


def clamp(x: float, lower=0., upper=1.) -> float:
    """
    Clamps a float to within a range (default [0, 1]).
    """
    from math import isnan
    if x <= lower:
        return lower
    elif x >= upper:
        return upper
    elif isnan(x):
        raise FloatingPointError('clamp is undefined for NaN')
    return x


class Timer:
    def __enter__(self) -> 'Timer':
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc_info) -> None:
        self.end = time.monotonic()

    @property
    def seconds(self) -> float:
        return self.end - self.start
