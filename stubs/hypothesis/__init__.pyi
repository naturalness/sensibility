#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Callable, NewType

class Strategy: ...

def given(*strats: Strategy) -> Callable[..., Callable]: ...
