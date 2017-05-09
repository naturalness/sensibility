#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from . import Transport

class RequestsHTTPTransport(Transport):
    def __init__(self, url: str , auth=None, use_json: bool=False, timeout=None, **kwargs) -> None: ...
