# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""
Execution Engine, which sets configs and sets up the simulation
"""

import os
import configparser
import importlib
import ast
import shutil
from distutils.dir_util import copy_tree
from time import gmtime, strftime

from ..run_scripts.sim import Simulator

class ExecutionEngine:
    def __init__(self, config_file='', run_name='', save_flag=False):

        # Init Housekeeping
        self.run_name = run_name
        self.config_filepath = config_file

        # Init Flags
        self.IO_Flag = False
        self.Dev_Flag = False
        self.Viz_Flag = False

        # Init Paths
        self.dataFilePath = ''
        self.telemetryFile = ''
        self.fullTelemetryFileName = ''
        self.metadataFilePath = ''
        self.metaFile = ''
        self.fullMetaDataFileName = ''
        self.benchmarkFilePath = ''
        self.benchmarkFiles = ''
        self.benchmarkIndices = ''

        # Init parsing/sim info
        self.parser_file_name = ''
        self.simDataSource = None
        self.sim = None

        # Init plugins
        self.knowledge_rep_plugin_dict = ['']
        self.learners_plugin_dict = ['']
        self.planners_plugin_dict = ['']
        self.complex_plugin_dict = ['']

        self.save_flag = save_flag
        self.save_name = run_name

        if config_file != '':
            self.init_save_paths()
            self.parse_configs(config_file)
            self.parse_data(self.parser_file_name, self.fullTelemetryFileName, self.fullMetaDataFileName)
            self.setup_sim()

    def parse_configs(self, config_filepath):
        config = configparser.ConfigParser()

        if len(config.read(config_filepath)) == 0:
            raise FileNotFoundError(f"Config file at '{config_filepath}' could not be read.")

        try:
            ## Parse Required Data: Telementry Data & Configuration
            self.dataFilePath = config['DEFAULT']['TelemetryDataFilePath']
            self.telemetryFile = config['DEFAULT']['TelemetryFile'] # Vehicle telemetry data
            self.fullTelemetryFileName = os.path.join(self.dataFilePath, self.telemetryFile)
            self.metadataFilePath = config['DEFAULT']['TelemetryMetadataFilePath']
            self.metaFile = config['DEFAULT']['MetaFile'] # Config for vehicle telemetry
            self.fullMetaDataFileName = os.path.join(self.metadataFilePath, self.metaFile)

            ## Parse Required Data: Names
            self.parser_file_name = config['DEFAULT']['ParserFileName']

            ## Parse Required Data: Plugins
            self.knowledge_rep_plugin_dict = self.parse_plugins_dict(config['DEFAULT']['KnowledgeRepPluginDict'])
            self.learners_plugin_dict = self.parse_plugins_dict(config['DEFAULT']['LearnersPluginDict'])
            self.planners_plugin_dict = self.parse_plugins_dict(config['DEFAULT']['PlannersPluginDict'])
            self.complex_plugin_dict = self.parse_plugins_dict(config['DEFAULT']['ComplexPluginDict'])

            ## Parse Optional Data: Flags
            ## 'RUN_FLAGS' must exist, but individual flags return False if missing
            self.IO_Flag = config['RUN_FLAGS'].getboolean('IO_Flag')
            self.Dev_Flag = config['RUN_FLAGS'].getboolean('Dev_Flag')
            self.Viz_Flag = config['RUN_FLAGS'].getboolean('Viz_Flag')

        except KeyError as e:
            new_message = f"Config file: '{config_filepath}', missing key: {e.args[0]}"
            raise KeyError(new_message) from e

        ## Parse Optional Data: Benchmarks
        try:
            self.benchmarkFilePath = config['DEFAULT']['BenchmarkFilePath']
            self.benchmarkFiles = config['DEFAULT']['BenchmarkFiles'] # Vehicle telemetry data
            self.benchmarkIndices = config['DEFAULT']['BenchmarkIndices']
        except:
            pass

    def parse_plugins_dict(self, config_plugin_dict):
        ## Parse Required Data: Plugin name to path dict
        ast_plugin_dict = self.ast_parse_eval(config_plugin_dict)
        if isinstance(ast_plugin_dict.body, ast.Dict):
            temp_plugin_dict = ast.literal_eval(config_plugin_dict)
        else:
            raise ValueError(f"Plugin dict {config_plugin_dict} from {self.config_filepath} is invalid. It must be a dict.")

        for plugin_file in temp_plugin_dict.values():
            if not(os.path.exists(plugin_file)):
                raise FileNotFoundError(f"In config file '{self.config_filepath}' Plugin path '{plugin_file}' does not exist.")
        return temp_plugin_dict

    def parse_data(self, parser_file_name, data_file_name, metadata_file_name, subsystems_breakdown=False):
        data_source_spec = importlib.util.spec_from_file_location('data_source', parser_file_name)
        data_source_module = importlib.util.module_from_spec(data_source_spec)
        data_source_spec.loader.exec_module(data_source_module)
        self.simDataSource = data_source_module.DataSource(data_file_name, metadata_file_name, subsystems_breakdown)

    def setup_sim(self):
        self.sim = Simulator(self.simDataSource,
                             self.knowledge_rep_plugin_dict,
                             self.learners_plugin_dict,
                             self.planners_plugin_dict,
                             self.complex_plugin_dict)
        try:
            fls = ast.literal_eval(self.benchmarkFiles)
            fp = os.path.dirname(os.path.realpath(__file__)) + '/../..' + self.benchmarkFilePath
            bi = ast.literal_eval(self.benchmarkIndices)
            self.sim.set_benchmark_data(fp, fls, bi)
        except:
            pass

    def run_sim(self):
        self.sim.run_sim(self.IO_Flag, self.Dev_Flag, self.Viz_Flag)
        if self.save_flag:
            self.save_results(self.save_name)

    def init_save_paths(self):
        save_path = os.environ['RESULTS_PATH']
        temp_save_path = os.path.join(save_path, 'tmp')
        temp_models_path = os.path.join(temp_save_path, 'models')
        temp_diagnosis_path = os.path.join(temp_save_path, 'diagnosis')

        self.delete_save_paths()
        os.mkdir(temp_save_path)
        os.mkdir(temp_models_path)
        os.mkdir(temp_diagnosis_path)

        os.environ['ONAIR_SAVE_PATH'] = save_path
        os.environ['ONAIR_TMP_SAVE_PATH'] = temp_save_path
        os.environ['ONAIR_MODELS_SAVE_PATH'] = temp_models_path
        os.environ['ONAIR_DIAGNOSIS_SAVE_PATH'] = temp_diagnosis_path

    def delete_save_paths(self):
        save_path = os.environ['RESULTS_PATH']
        sub_dirs = os.listdir(save_path)
        if 'tmp' in sub_dirs:
            try:
                shutil.rmtree(save_path + '/tmp')
            except OSError as e:
                print("Error: %s : %s" % (save_path, e.strerror))

    def save_results(self, save_name):
        complete_time = strftime("%H-%M-%S", gmtime())
        save_path = os.environ['ONAIR_SAVE_PATH'] + '/saved/' + save_name + '_' + complete_time
        os.mkdir(save_path)
        copy_tree(os.environ['ONAIR_TMP_SAVE_PATH'], save_path)

    def set_run_param(self, name, val):
        setattr(self, name, val)

    def ast_parse_eval(self, config_list):
        return ast.parse(config_list, mode='eval')
