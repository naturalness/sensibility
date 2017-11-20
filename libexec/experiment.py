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
Defines a way to add an experiment to a Makefile.
"""

from pathlib import Path

from makefile import Command


base_dir = Path('.')

# All commands use the `sensibility` wrapper.
sensibility = Command('sensibility', '-l', 'java')
# The training command.
train = sensibility('train-lstm')
# The evaluation command.
evaluate = sensibility('evaluate')


class Experiment:
    def __init__(self, *, name, language, training_configs, eval_configs):
        self.name = name
        self.training_configs = training_configs
        self.eval_configs = eval_configs
        self.model_dir = base_dir / 'models' / language
        self.eval_dir = base_dir / 'evaluation' / language / 'results'

    def add_to_makefile(self, make):
        # Side-effect: create the directories if they don't exist.
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        models = []
        results = []

        # Create rules for the training models.
        for config in self.training_configs:
            forwards_model = self.model_dir / config.hashed_slug.with_suffix('.forwards')
            backwards_model = self.model_dir / config.hashed_slug.with_suffix('.backwards')

            make.rule(forwards_model).set_recipe(
                train(forwards=True, output_dir='$@', **config),
            )
            make.rule(backwards_model).set_recipe(
                train(backwards=True, output_dir='$@', **config)
            )
            models.append((forwards_model, backwards_model))

        # Creates rules for evaluating models.
        for config in self.eval_configs:
            # Evaluate each and every one of the models created in the last step.
            for forwards, backwards in models:
                assert forwards.stem == backwards.stem
                # The filename will indicate the kind of fix attempted.
                eval_file = (
                    (self.eval_dir / f"fix-{config.fix}-{forwards.stem}").with_suffix('.sqlite3')
                )
                make.rule(eval_file).depends_on(forwards, backwards, 'mistakes.txt').set_recipe(
                    evaluate(o='$@',
                             # The model directory without the extension.
                             models=forwards.parent / forwards.stem,
                             mistakes_list='mistakes.txt',
                             **config)
                )
                results.append(eval_file)

        # Create the phony rule so that you can just run:
        #   make -j -k experiments
        # and do EVERYTHING!
        make.phony_rule(f'{self.name}-experiments').depends_on(*results)

        # Make a phony rule for models.
        make.phony_rule(f'{self.name}-models').depends_on(
            *(f for f, b in models), *(b for f, b in models)
        )
