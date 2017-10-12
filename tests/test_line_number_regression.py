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

from sensibility.edit import Insertion, Deletion
from sensibility.evaluation.distance import determine_fix_event
from sensibility.language import language


# This example caused a crash due to the edit's line number being
# calculated on the wrong revision of the file.
#
# This example has been modified slightly, but the line numbers and token
# indices are the same.
# id: 11608/193044
error_file = b"""
/**
 * ##### # ########### ## ##### ############## ####.
 *
 * @###### (#### ####)
 * @####### (# ####### ###### ## # ####)
 */
public class DaysAlivePrint
{



    public static void  main(String[] args)
    {
        Day birthday = new Day(1951, 5, 25);
        Day today = new Day(2012, 7, 23);
        int days = today.daysFrom(birthday);
        System.out.println(days);
    }


    }
}"""

fixed_file = b"""
public class DaysAlivePrint
{


     public static void  main(String[] args)
    {
        Day birthday = new Day(1951, 5, 25);
        Day today = new Day(2012, 7, 23);
        int days = today.daysFrom(birthday);
        System.out.println(days); //print result
    }



}"""


def setup():
    language.set('java')


def test_calculates_line_numbers_from_source(c) -> None:
    event = determine_fix_event(error_file, fixed_file)
    assert isinstance(event.fix, Deletion)
    # One of the curly braces at the end of the ERROR file.
    assert event.fix.original_token == language.vocabulary.to_index(c('}'))
    assert event.line_no in {19, 22, 23}


def test_calculates_first_line_number() -> None:
    """
    Tests to ensure position of line number doesn't "wrap around" to the
    bottom.
    """

    error = b"""
    HelloWorld {
        string greet() {
            return "hello, world!";

        }
    }"""

    fixed = b"""
    class HelloWorld {
        string greet() {
            return "hello, world!";

        }
    }"""

    event = determine_fix_event(error, fixed)
    assert isinstance(event.fix, Insertion)
    assert event.fix.index == 0
    # It used to think this was line number zero, because it wrapped around to
    # the end of the file.
    assert event.line_no == 2
