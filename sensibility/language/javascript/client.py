#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import warnings
import atexit
import os
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import zmq  # type: ignore

here = Path(__file__).parent.absolute()


class ParseServer:
    def __init__(self, socket_path: Path) -> None:
        context = zmq.Context()
        socket = self._socket = context.socket(zmq.REQ)
        socket.connect(f"ipc://{socket_path}")

    def check_syntax(self, source: bytes) -> bool:
        return self._communicate(b'c', source)

    def tokenize(self, source: bytes) -> Any:
        return self._communicate(b't', source)

    def exit(self) -> None:
        response = self._communicate(b'x')
        assert response is True

    def _communicate(self, type_code: bytes, payload: bytes=None) -> Any:
        self._socket.send(type_code + payload if payload else type_code)
        response = self._socket.recv()
        return json.loads(response)


# A global instance of the parse server.
_instance: Optional[ParseServer] = None


def get_instance() -> ParseServer:
    if _instance is None:
        return start()
    return _instance


def start() -> ParseServer:
    global _instance
    argv0 = str(here / 'esprima-interface')
    socket_path = Path(f'/tmp/esprima-server.{os.getpid()}')
    subprocess.Popen([argv0, '--server', f"ipc://{socket_path}"],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL,
                     shell=False)

    _instance = ParseServer(socket_path)

    @atexit.register
    def cleanup():
        _instance.exit()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            warnings.warn("Socket {socket_path} already deleted?")

    return _instance


if __name__ == '__main__':
    from sensibility.language.javascript import javascript
    # Time how many times I can tokenize the source code of the server itself.
    parser = get_instance()
    import timeit
    with open(here / 'index.js', 'rb') as source_file:
        source = source_file.read()

    timer = timeit.Timer('parser.tokenize(source)', globals=globals())
    samples = timer.repeat(number=1000)
    print("Using server:", *sorted(samples))

    timer = timeit.Timer('javascript.tokenize(source)', globals=globals())
    samples = timer.repeat(number=1000)
    print("Using original:", *sorted(samples))
