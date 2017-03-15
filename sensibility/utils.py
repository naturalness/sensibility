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
from os import PathLike

assert os.symlink in os.supports_dir_fd  # type: ignore # noqa https://github.com/python/typeshed/issues/1003


def symlink_within_dir(
        *, directory: PathLike, source: PathLike, target: PathLike
) -> None:
    """
    Creates a symbolic link (symlink) relative to a directory.
    """
    fd = os.open(os.fspath(directory), os.O_RDONLY | os.O_DIRECTORY)  # type: ignore # noqa https://github.com/python/typeshed/pull/991
    try:
        os.symlink(source, target, dir_fd=fd)  # type: ignore # noqa https://github.com/python/typeshed/pull/991
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
