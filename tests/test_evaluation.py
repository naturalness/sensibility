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


canonical_error = br"""                   //  1
class Hello {                               //  2
    public void main(String args[]) {       //  3
        if (args.length < 3) // {           --   4
            System.err.println("No good!"); //  5
            System.exit(2);                 //  6
        }                                   //  7
        System.out.println("Good!");        //  8
        System.exit(0);                     //  9
    }                                       // 10
}                                           // 11
"""

fixed = br"""                               //  1
class Hello {                               //  2
    public void main(String args[]) {       //  3
        if (args.length < 3) {              //   4
            System.err.println("No good!"); //  5
            System.exit(2);                 //  6
        }                                   //  7
        System.out.println("Good!");        //  8
        System.exit(0);                     //  9
    }                                       // 10
}                                           // 11
"""


def setup():
    from sensibility.language import language
    language.set_language('java')


def test_evaluation() -> None:
    from sensibility.evaluation.evaluate import (
        Evaluation, EvaluationFile, EvaluationResult, LSTMPartition, Mistake,
    )
    from sensibility.evaluation.distance import determine_fix_event

    evaluation = Evaluation('mistake', LSTMPartition(3))
    event = determine_fix_event(canonical_error, fixed)
    mistake = Mistake('example', canonical_error, event)
    actual = evaluation.evaluate_file(mistake)

    assert 'lstm3' == actual.model
    assert 'mistake' == actual.mode
    assert 10 == actual.n_lines
    assert 56 == actual.n_tokens
    assert event.mistake == actual.error
    assert actual.fixed
