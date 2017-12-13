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

"""

Allows one to set the language prior to running any of the scripts:

Usage:

    sensibility [-l LANGUAGE] <command> [<args>]
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from sensibility._paths import REPOSITORY_ROOT

bin_dir = REPOSITORY_ROOT / 'bin'


def main() -> None:
    assert bin_dir.is_dir()
    args = parse_args()

    # Set up the environment
    env: Dict[str, str] = {}
    env.update(os.environ)

    # Set the language if defined.
    if args.language is not None:
        env.update(SENSIBILITY_LANGUAGE=args.language)

    if args.subcommand:
        run_subcommand(args.subcommand, env)
    else:
        list_commands()
        sys.exit(-1)


def run_subcommand(command, env) -> None:
    bin, args = get_bin_and_argv(command)
    if not bin.exists():
        usage_error("Unknown executable:", bin)
    os.execve(str(bin.absolute()), args, env)


def list_commands() -> None:
    print("Please specify a subcommand:\n", file=sys.stderr)
    for bin in bin_dir.rglob('*'):
        if bin.is_dir() or not is_executable(bin):
            continue
        bin = bin.relative_to(bin_dir)
        subcommand = ' '.join(bin.parts)
        print(f"\t{subcommand}", file=sys.stderr)


def get_bin_and_argv(command: List[str]) -> Tuple[Path, List[str]]:
    """
    Returns the absolute path to the binary, AND the argument vector,
    including argv[0] (the command name).
    """
    first_comp, = command[:1]
    # XXX: Only supports one-level subcommands
    if (bin_dir / first_comp).is_dir():
        return bin_dir / first_comp / command[1], command[1:]
    else:
        return bin_dir / first_comp, command


def is_executable(path: Path) -> bool:
    # access() is deprecated, but we're using it anyway!
    return os.access(path, os.X_OK)


def parse_args(argv=sys.argv):
    """
    Roll my own parse because argparse will swallow up arguments that don't
    belong to it.
    """
    argv = argv[1:]
    args = SimpleNamespace()
    args.language = None
    args.subcommand = None

    # Parse options one by one.
    while argv:
        arg = argv.pop(0)
        if arg in ('-l', '--language'):
            args.language = argv.pop(0)
        elif arg.startswith('--language='):
            _, args.language = arg.split('=', 1)
        elif arg.startswith('-'):
            usage_error(f"Unknown argument {arg!r}")
        else:
            args.subcommand = [arg] + argv[:]
            break
    return args


def usage_error(*args):
    print(f"{sys.argv[0]}:", *args, file=sys.stderr)
    sys.exit(2)


if __name__ == '__main__':
    main()
