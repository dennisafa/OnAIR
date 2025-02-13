# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
CSV Parser
"""

import os
import pandas as pd

from onair.data_handling.on_air_data_source import OnAirDataSource
from onair.src.util.print_io import *
from onair.data_handling.parser_util import *

class DataSource(OnAirDataSource):

    def process_data_file(self, data_file):
        self.sim_data = self.parse_csv_data(data_file)
        self.frame_index = 0

##### INITIAL PROCESSING ####
    def parse_csv_data(self, data_file):
        #Read in the data set
        dataset = pd.read_csv(data_file, delimiter=',', header=0, dtype=str)
        dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

        #Initialize the entire data dictionary
        all_data = []
        for index, row in dataset.iterrows():
            rowVals = floatify_input(list(row))
            all_data.append(floatify_input(list(row)))

        return all_data

    def parse_meta_data_file(self, meta_data_file, ss_breakdown):
        return extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown)

##### GETTERS ##################################

    def get_vehicle_metadata(self):
        return self.all_headers, self.binning_configs['test_assignments']

    # Get the data at self.index and increment the index
    def get_next(self):
        self.frame_index = self.frame_index + 1
        return self.sim_data[self.frame_index - 1]

    # Return whether or not the index has finished traveling through the data
    def has_more(self):
        return self.frame_index < len(self.sim_data)
