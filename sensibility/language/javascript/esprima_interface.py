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
Provides an interface to Esprima, implemented in Node.JS.
"""

import atexit
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import IO, Any, Iterator, Optional

import zmq  # type: ignore

here = Path(__file__).parent.absolute()
esprima_bin = here / 'esprima-interface'
assert esprima_bin.exists()


class Server:
    """
    Provides a convenient interface for accessing the Esprima interface
    server.

    All that's required is the path of the UNIX domain socket where the
    already instantiated server is running.
    """

    def __init__(self, socket_path: Path, process) -> None:
        context = zmq.Context()
        self._socket = context.socket(zmq.REQ)
        # Wait for a maximum of 5 seconds before forfeiting the request.
        self._socket.RCVTIMEO = 5000
        self._socket.connect(f"ipc://{socket_path}")
        self._process = process

    def check_syntax(self, source: bytes) -> bool:
        return self._communicate(b'c', source)

    def tokenize(self, source: bytes) -> Iterator[Any]:
        return self._communicate(b't', source)

    def exit(self) -> None:
        response = self._communicate(b'x')
        assert response is True
        # Remove this reference right here to:
        #  - prevent wonkiness when shutting down
        #  - cause errors if a future request is made.
        del self._process

    def _communicate(self, type_code: bytes, payload: bytes=None) -> Any:
        self._ensure_alive()
        self._socket.send(type_code + payload if payload else type_code)
        response = self._socket.recv()
        return json.loads(response)

    def _ensure_alive(self) -> None:
        if self._process.poll() is not None:
            raise RuntimeError('Server process is dead.')


# A global instance of the parse server.
_instance: Optional[Server] = None


def get_server() -> Server:
    """
    Retrieves the global server instance.
    """
    if _instance is None:
        return start()
    return _instance


def start() -> Server:
    """
    Starts the global server instance.
    """
    global _instance
    argv0 = str(here / 'esprima-interface')
    socket_path = Path(f'/tmp/esprima-server.{os.getpid()}')
    proc = subprocess.Popen([argv0, '--server', f"ipc://{socket_path}"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            shell=False)

    _instance = Server(socket_path, proc)

    @atexit.register
    def cleanup():
        _instance.exit()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            warnings.warn("Socket {socket_path} already deleted?")

    return _instance


def tokenize(file_obj: IO[bytes]) -> Iterator[Any]:
    """
    Tokenizes a (real!) bytes file using Esprima.
    """
    status = subprocess.run([str(esprima_bin)],
                            check=True,
                            stdin=file_obj,
                            stdout=subprocess.PIPE)
    return json.loads(status.stdout.decode('UTF-8'))


def check_syntax(source_file: IO[bytes]) -> bool:
    status = subprocess.run((str(esprima_bin), '--check-syntax'),
                            check=False,
                            stdin=source_file,
                            stdout=subprocess.PIPE)
    return status.returncode == 0


# Tests the server.
if __name__ == '__main__':
    from sensibility.language.javascript import javascript
    # Time how many times I can tokenize the source code of the server itself.
    parser = get_server()
    import timeit
    with open(here / 'index.js', 'rb') as source_file:
        source = source_file.read()

    timer = timeit.Timer('parser.tokenize(source)', globals=globals())
    samples = timer.repeat(number=1000)
    print("Using server:", *sorted(samples))

    timer = timeit.Timer('javascript.tokenize(source)', globals=globals())
    samples = timer.repeat(number=1000)
    print("Using original:", *sorted(samples))
