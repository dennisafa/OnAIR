""" Test Time Sync Functionality """
import os
import unittest

from mock import MagicMock
import src.data_handling.time_synchronizer as time_synchronizer
from src.data_handling.time_synchronizer import TimeSynchronizer

# __init__ tests
def test_init_does_not_set_instance_default_values_when_calls_to_init_sync_data_and_sort_data_do_not_raise_exceptions(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data')
    mocker.patch.object(cut, 'sort_data')

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 1
    assert cut.sort_data.call_args_list[0].args == (arg_dataFrames, )
    assert hasattr(cut, 'ordered_sources') == False
    assert hasattr(cut, 'ordered_fused_headers') == False
    assert hasattr(cut, 'ordered_fused_tests') == False
    assert hasattr(cut, 'indices_to_remove') == False
    assert hasattr(cut, 'offsets') == False
    assert hasattr(cut, 'sim_data') == False

def test_init_sets_instance_default_values_when_call_to_init_sync_data_raises_exception(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data', side_effect=Exception())
    mocker.patch.object(cut, 'sort_data')

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 0
    assert cut.ordered_sources == []
    assert cut.ordered_fused_headers == []
    assert cut.ordered_fused_tests == []
    assert cut.indices_to_remove == []
    assert cut.offsets == {}
    assert cut.sim_data == []

def test_init_sets_instance_default_values_when_call_to_sort_data_raises_exception(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_dataFrames = MagicMock()
    arg_test_configs = MagicMock()

    cut = TimeSynchronizer.__new__(TimeSynchronizer)

    mocker.patch.object(cut, 'init_sync_data')
    mocker.patch.object(cut, 'sort_data', side_effect=Exception())

    # Act
    cut.__init__(arg_headers, arg_dataFrames, arg_test_configs)

    # Assert
    assert cut.init_sync_data.call_count == 1
    assert cut.init_sync_data.call_args_list[0].args == (arg_headers, arg_test_configs)
    assert cut.sort_data.call_count == 1
    assert cut.sort_data.call_args_list[0].args == (arg_dataFrames, )
    assert cut.ordered_sources == []
    assert cut.ordered_fused_headers == []
    assert cut.ordered_fused_tests == []
    assert cut.indices_to_remove == []
    assert cut.offsets == {}
    assert cut.sim_data == []

class TestTimeSynchronizer(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        self.TS = TimeSynchronizer()

    def test_init_empty_sync_data(self):
        self.assertEqual(self.TS.ordered_sources, [])
        self.assertEqual(self.TS.ordered_fused_headers, [])
        self.assertEqual(self.TS.ordered_fused_tests, [])
        self.assertEqual(self.TS.indices_to_remove, [])
        self.assertEqual(self.TS.offsets, {})
        self.assertEqual(self.TS.sim_data, [])

    def test_init_sync_data(self):
        hdrs = {'test_sample_01' : ['TIME', 'hdr_A', 'hdr_B'],
                'test_sample_02' : ['TIME', 'hdr_C']}        
        
        # Even if you give configs with ss assignments, they should not be here at the binner stage 
        configs = {'test_assignments': {'test_sample_01': [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]],
                                        'test_sample_02': [[['SYNC', 'TIME']], [['NOOP']]]}, 
                   'description_assignments': {'test_sample_01': ['Time', 'No description', 'No description']}}

        self.TS.init_sync_data(hdrs, configs) 

        self.assertEqual(self.TS.ordered_fused_tests, [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]])
        self.assertEqual(self.TS.ordered_sources, ['test_sample_01', 'test_sample_02'])
        self.assertEqual(self.TS.ordered_fused_headers, ['TIME', 'hdr_A', 'hdr_B', 'hdr_C'])
        self.assertEqual(self.TS.indices_to_remove, [0,3])
        self.assertEqual(self.TS.offsets, {'test_sample_01': 0, 'test_sample_02': 3})
        self.assertEqual(self.TS.sim_data, [])

    def test_sort_data(self):

        self.TS.ordered_fused_tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']], [['NOOP']]]
        self.TS.ordered_sources = ['test_sample_01', 'test_sample_02']
        self.TS.ordered_fused_headers = ['TIME', 'hdr_A', 'hdr_B', 'hdr_C']
        self.TS.indices_to_remove =[0,3]
        self.TS.offsets = {'test_sample_01': 0, 'test_sample_02': 3}
        self.TS.unclean_fused_hdrs = ['TIME', 'hdr_A', 'hdr_B', 'TIME', 'hdr_C']

        data = {'1234' : {'test_sample_01' : ['1234','202','0.3'],
                          'test_sample_02' : ['1234','0.3']},
                '2235' : {'test_sample_02' : ['2235','202']},
                '1035' : {'test_sample_01' : ['1035','202','0.3'],
                          'test_sample_02' : ['1035','0.3']},
                '1305' : {'test_sample_01' : ['1005','202','0.3']},
                '1350' : {'test_sample_01' : ['1350','202','0.3'],
                          'test_sample_02' : ['1350','0.3']}}

        self.TS.sort_data(data)

        self.assertEqual(self.TS.sim_data, [['1035', '202', '0.3', '0.3'], 
                                             ['1234', '202', '0.3', '0.3'], 
                                             ['1305', '202', '0.3', '-'], 
                                             ['1350', '202', '0.3', '0.3'], 
                                             ['2235', '-', '-', '202']])

    def test_remove_time_headers(self):
        hdrs_list = ['A', 'B', 'time', 'TIME', 'C', 'D']
        indices, clean_hdrs_list = self.TS.remove_time_headers(hdrs_list)
        self.assertEqual(clean_hdrs_list, ['A', 'B','C', 'D'])

    def test_remove_time_datapoints(self):
        data = ['1', '2', '3', '4']
        indices_to_remove = [0, 2]
        clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
        self.assertEqual(clean_data, ['2', '4'])

        data = [['list1'], [], ['list3', 'list3'], []]
        indices_to_remove = [0, 2]
        clean_data = self.TS.remove_time_datapoints(data, indices_to_remove)
        self.assertEqual(clean_data, [[],[]])

    def test_get_spacecraft_metadata(self):
        return 

    def test_get_sim_data(self):
        return
        
if __name__ == '__main__':
    unittest.main()
