#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2016, 2017 Eddie Antonio Santos <easantos@ualberta.ca>
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
Places files into partitions attempting to equalize the token length.

                            ,                       :::
                                     ,:::::::        ::::,
       ,,,,;;;;;;;;;,,                 ,:::::::,     ::::::,
   ,,,;::::::::::::;,,,                ,::::::::,    :::::,,,
  ;:::::::::::,,,,,                    ,::::::::,,   :::::,,,        +
 ,:::::::,,,,                          ::::::::,,,   ::::,,,      ,
 ,::::;,,,                             :::::::,,,   ~::::,,,
 ,::::::,                              :::::::,,,   ,::::,,         ,:::,
  :::::::,                             ::::::,,, :::::::,,,        :::::::,
  :::::::::,                           ::::::,,::::,::::,,I        :::::::::,
  ,:::::::::,,                 :,      :::::,,::::,,:::,,,        ,:::::::::,,
   :::::::::,,,                ::::    :::::,,:::,,,:::,,,        ,::::::::,,,,
    ::::::::,,,      ::::,,   ::::::, +::::,,:::,,,,:::,,,        :::::::,,,,
    ::::::::,,,    ,:::::,,  ::::::,:,,::::,,:::,,, ::::,,,:,,    :::::::,,,
     :::::::,,,   ,::::::,, ,:::::,,::,::::,,::::,,~:::::::,,,    ::::::,,,
      :::::::,, ::::::::,,, ::::::,,::,:::,,,:::::::,:::::,,,     :::::,,,    +
      ,::::::,::::::::::,,, :::::,,,::,:::,,,::::::,,,::::,,,     :::::,,,
       ::::::::::::,::,,,   ::::,,, ::,::::,,::::::,,,I,,,,,      ,:::,,,
        :::::::::,,,,,      ::::,,  ::,::::::,,,,,,,,::::::I      ~:::,,,
         ::::::,,,,,       ,:::::, :::,:::::,,,  :,? ,:::::,,,     :::,,
     I   ,:::::,,,        ,+::::::::::,,:::,,,,      ?:::::::::,,  :::,,
          :::::::,          :::::::::,,, :::::, ::::::::::,,,,,,,,+::,,,
           :::::::,          ::::::::,,,  ::::::,:::::::::,,,      ,:,,,
            ::::::,,         `;::::,,,,  ,:::::,,::::,::::,,        ::,,
            ,::::::,,        , ,,,,,,,  ::,,,,,,,:::,,::::,,   ::,, ,:,,
             ::::::,,            ,,~    :::::     ,,,,:::::, ,::,,,, :,,
              ::::::,,                  :::::::,    ,:,::::::::,,,  , ,,
              ,:::::,,                  ,::::::,,   ::,:::::::,,,   ,::
               :::::,,I                  ::::::,,  ,:,,,:::::,,,    :::::::
               ,::::,,,                  ,::::::,   ,,,  +,,,,,     :::::::,,
         ,       ,,,,,,                   ::::::::::       ,,      :::::::,,,
                                          ,:::::::,,,,             :::::,,,,,
                                           ::::::,,,,               ,,,,,,,
        I                                  ,::::,,,
                              ?             ,:,,,,                    +
                                              ,,,                         ,
                                                                           +


    Partions:   |  0   |   1   |   2   |   3   |   4   |

    Each partition is made of the following sets:

    Sets:   | train set | validate set | test set |

"""

import argparse
import heapq
import random
import sys
import warnings

from pathlib import Path
from functools import partial, total_ordering
from typing import Set

from tqdm import tqdm

from sensibility.miner.models import RepositoryID
from sensibility.miner.corpus import Corpus
from sensibility._paths import EVALUATION_DIR
from sensibility.language import language

PARTITIONS = 5
stderr = partial(print, file=sys.stderr)


class Split:
    "Define training/validation/test splits."
    def __init__(self, split: str) -> None:
        train, validate, test = (int(n) for n in split.split('/'))
        assert train + validate + test == 100
        self.train = train / 100.0
        self.validate = validate / 100.0
        self.test = test / 100.0


parser = argparse.ArgumentParser(
    description='Divides the corpus into partitions.'
)
parser.add_argument('-o', '--output-dir', type=Path, default=EVALUATION_DIR)
parser.add_argument('-s', '--split', type=Split, default=Split('80/10/10'))
parser.add_argument('-f', '--overwrite', type=bool, default=False)


def main() -> None:
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    overwrite: bool = args.overwrite

    # Magically get the corpus.
    # TODO: from cmdline arguments
    corpus = Corpus(read_only=True)

    @total_ordering
    class Partition:
        def __init__(self, number: int) -> None:
            self.n_tokens = 0
            self.number = 0
            self.repos: Set[RepositoryID] = set()
            self.seen: Set[str] = set()

        def __eq__(self, other):
            return self.n_tokens == other.n_tokens

        def __lt__(self, other) -> bool:
            return self.n_tokens < other.n_tokens

        def add_repo(self, repo: RepositoryID, tokens: int) -> None:
            self.repos.add(repo)
            self.n_tokens += tokens

        def create_sets(self) -> None:
            splits = args.split
            repos = list(self.repos)
            random.shuffle(repos)

            n_train = int(splits.train * len(repos))
            self.training_repos = repos[:n_train]

            n_validate = int(splits.validate * len(repos))
            point = n_train + n_validate
            self.validation_repos = repos[n_train:point]

            n_test = int(splits.test * len(repos))
            self.test_repos = repos[point:]

        def save_to(self, path: Path) -> None:
            # Create the path if it doesn't exist.
            directory = path / str(self.number)
            directory.mkdir(parents=True, exist_ok=True)

            for set_name in 'training', 'validation', 'test':
                self._commit_set(set_name, directory)

        def _commit_set(self, set_name: str, directory: Path) -> None:
            repos = getattr(self, f"{set_name}_repos")
            # Add all hashes from these repos to this fold.
            hashes: Set[str] = set()
            for repo in repos:
                for filehash in corpus.get_hashes_in_repo(repo):
                    # Make sure we're not adding duplicates!
                    if filehash in self.seen:
                        warnings.warn(f'Already saw {filehash} in partition'
                                      f'{self.number} when adding to {set_name}')
                        continue
                    hashes.add(filehash)

            # Create the output file by shuffling the hashes.
            hash_list = list(hashes)
            random.shuffle(hash_list)
            with open(directory / set_name, 'w') as set_file:
                for filehash in hash_list:
                    print(filehash, file=set_file)

            # Add all the seen hashes.
            self.seen.update(hashes)

    # We maintain a priority queue of partitions. At the top of the heap is the
    # partition with the fewest tokens.
    heap = [Partition(part_no) for part_no in range(PARTITIONS)]
    heapq.heapify(heap)

    def pop():
        return heapq.heappop(heap)

    def push(partition: Partition):
        return heapq.heappush(heap, partition)

    # This is kinda dumb, but:
    # Shuffle a list of ALL repositories...
    stderr('Shuffling repositories...')
    shuffled_repos = list(corpus.get_repositories_with_n_tokens())
    random.shuffle(shuffled_repos)

    # Assign a repository to each fold...
    for repo, n_tokens in tqdm(shuffled_repos):
        # Assign this project to the smallest partition
        partition = pop()
        partition.add_repo(repo, n_tokens)
        push(partition)

    # Output structure
    #   {output_dir}/{language}/partitions/{partition_no}/{set}
    stderr('Shuffling writing sets for each partition...')
    for partition in tqdm(heap):
        partition.create_sets()
        partition.save_to(output_dir / language.id / 'partitions')
