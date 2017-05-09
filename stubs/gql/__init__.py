#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import AnyStr
from .transport import Transport

class Document:
    ...

def gql(request: AnyStr) -> Document: ...

class Client:
    def __init__(self, transport: Transport) -> None: ...
    def execute(self, query: Document) -> object: ...
