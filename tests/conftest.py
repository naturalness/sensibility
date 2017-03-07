#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# http://doc.pytest.org/en/latest/example/simple.html
import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
        help="run slow tests")
