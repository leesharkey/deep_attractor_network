#!/usr/bin/env python3
# Copyright 2019 Lee Sharkey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""
Combines all the CSV files into one
-----------------------------------

"""
import argparse
from lib import utils

def collate_CSVs():
    parser = argparse.ArgumentParser(description='Collates all CSVs containing'+
                                     'experimental settings and results into' +
                                     ' one CSV file.'
                                     )
    cgroup = parser.add_argument_group('CSV collation settings')
    cgroup.add_argument('--directory_str', type=str, default='exps',
                        help='The name of the directive (relative to the ' +
                             'current directory) of the directory with the ' +
                             'CSVs. Default: %(default)s.',
                        required=False)
    cgroup.add_argument('--base_csv_name', type=str, default='params_and_results',
                        help='The name of the final destination file for the '+
                        'combined CSVs. Default: %(default)s.',
                        required=False)
    cgroup.add_argument('--remove_old_csvs', action='store_true',
                        help='Moves all the old csvs to the archive folder ' +
                             'when they\'ve been added to the main csv.')
    parser.set_defaults(remove_old_csvs=False)

    args = parser.parse_args()
    if not vars(args)['base_csv_name'].endswith('.csv'):
        vars(args)['base_csv_name'] += '.csv'

    utils.combine_all_csvs(args.directory_str,
                           args.base_csv_name,
                           args.remove_old_csvs)

if __name__ == '__main__':
    collate_CSVs()
