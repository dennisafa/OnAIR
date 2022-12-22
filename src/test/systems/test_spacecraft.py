""" Test Spacecraft Functionality """
import pytest
from mock import MagicMock
import src.systems.spacecraft as spacecraft
from src.systems.spacecraft import Spacecraft

# __init__ tests
def test_Spacecraft__init__asserts_when_len_given_headers_is_not_eq_to_len_given_tests(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = []
    fake_len.append(pytest.gen.randint(0, 100)) # arbitrary, from 0 to 100 size
    fake_len.append(fake_len[0])
    while fake_len[1] == fake_len[0]: # need a value not equal for test to pass
        fake_len[1] = pytest.gen.randint(0, 100) # arbitrary, same as fake_len_headers

    cut = Spacecraft.__new__(Spacecraft)

    mocker.patch('src.systems.spacecraft.len', side_effect=fake_len)
    # Act
    with pytest.raises(AssertionError) as e_info:
        cut.__init__(arg_headers, arg_tests)
    
    # Assert
    assert spacecraft.len.call_count == 2
    call_list = set({})
    [call_list.add(spacecraft.len.call_args_list[i].args) for i in range(len(spacecraft.len.call_args_list))]
    assert call_list == {(arg_headers, ), (arg_tests, )}
    assert e_info.match('')

def test_Spacecraft__init__sets_status_to_Status_with_str_MISSION_and_headers_to_given_headers_and_test_suite_to_TelemetryTestSuite_with_given_headers_and_tests_and_curr_data_to_all_empty_step_len_of_headers(mocker):
    # Arrange
    arg_headers = MagicMock()
    arg_tests = MagicMock()

    fake_len = pytest.gen.randint(0, 100) # arbitrary, 0 to 100 items
    fake_status = MagicMock()
    fake_test_suite = MagicMock()

    cut = Spacecraft.__new__(Spacecraft)

    mocker.patch('src.systems.spacecraft.len', return_value=fake_len)
    mocker.patch('src.systems.spacecraft.Status', return_value=fake_status)
    mocker.patch('src.systems.spacecraft.TelemetryTestSuite', return_value=fake_test_suite)
    
    # Act
    cut.__init__(arg_headers, arg_tests)

    # Assert
    assert spacecraft.Status.call_count == 1
    assert spacecraft.Status.call_args_list[0].args == ('MISSION', )
    assert cut.status == fake_status
    assert cut.headers == arg_headers
    assert spacecraft.TelemetryTestSuite.call_count == 1
    assert spacecraft.TelemetryTestSuite.call_args_list[0].args == (arg_headers, arg_tests)
    assert cut.test_suite == fake_test_suite
    assert cut.curr_data == ['-'] * fake_len

# NOTE: commonly each optional arg is tested, but because their sizes must be equal testing both at once
def test_Spacecraft__init__default_given_headers_and_tests_are_both_empty_list(mocker):
    # Arrange
    cut = Spacecraft.__new__(Spacecraft)

    mocker.patch('src.systems.spacecraft.Status')
    mocker.patch('src.systems.spacecraft.TelemetryTestSuite')
    
    # Act
    cut.__init__()

    # Assert
    assert cut.headers == []
    assert spacecraft.TelemetryTestSuite.call_count == 1
    assert spacecraft.TelemetryTestSuite.call_args_list[0].args == ([], [])
    assert cut.curr_data == ['-'] * 0

# update tests
def test_Spacecraft_update_does_not_set_any_curr_data_when_given_frame_is_vacant_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_Spacecraft_update_does_not_set_any_curr_data_when_given_frame_is_all_empty_step_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_empty_steps = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_empty_steps):
        arg_frame.append('-')

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    cut.curr_data = []

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_empty_steps
    assert cut.curr_data == []
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_Spacecraft_update_does_puts_all_frame_data_into_curr_data_when_none_are_empty_step_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_full_steps = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_full_steps):
        arg_frame.append(MagicMock())

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    cut.curr_data = [MagicMock()] * num_fake_full_steps

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_full_steps
    assert cut.curr_data == arg_frame
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_Spacecraft_update_puts_frame_data_into_curr_data_at_same_list_location_unless_data_is_empty_step_then_leaves_curr_data_that_location_alone_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())
    location_fake_empty_steps = pytest.gen.sample(list(range(num_fake_total_steps)), pytest.gen.randint(1, num_fake_total_steps - 1)) # sample from a list of all numbers up to total then take from 1 to up to 1 less than total
    for i in location_fake_empty_steps:
        arg_frame[i] = '-'

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * num_fake_total_steps

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_total_steps
    for i in range(len(cut.curr_data)):
        if location_fake_empty_steps.count(i):
            assert cut.curr_data[i] == unchanged_data
        else:
            assert cut.curr_data[i] == arg_frame[i]
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_Spacecraft_update_puts_frame_data_into_curr_data_at_same_list_location_unless_data_is_empty_step_then_leaves_curr_data_that_location_alone_including_locations_in_curr_data_that_do_not_exist_in_frame_and_executes_suite_with_given_frame_and_sets_status_with_suite_status(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())
    location_fake_empty_steps = pytest.gen.sample(list(range(num_fake_total_steps)), pytest.gen.randint(1, num_fake_total_steps - 1)) # sample from a list of all numbers up to total then take from 1 to up to 1 less than total
    for i in location_fake_empty_steps:
        arg_frame[i] = '-'

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * (num_fake_total_steps + pytest.gen.randint(1, 10)) # arbitrary, from 1 to 10 extra items over frame

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    cut.update(arg_frame)

    # Assert
    assert len(arg_frame) == num_fake_total_steps
    assert len(cut.curr_data) > len(arg_frame)
    for i in range(len(cut.curr_data)):
        if i >= len(arg_frame) or location_fake_empty_steps.count(i):
            assert cut.curr_data[i] == unchanged_data
        else:
            assert cut.curr_data[i] == arg_frame[i]
    assert cut.test_suite.execute_suite.call_count == 1
    assert cut.test_suite.execute_suite.call_args_list[0].args == (arg_frame, )
    assert cut.test_suite.get_suite_status.call_count == 1
    assert cut.test_suite.get_suite_status.call_args_list[0].args == ()
    assert cut.status.set_status.call_count == 1
    assert cut.status.set_status.call_args_list[0].args == tuple(fake_suite_status)
    
def test_Spacecraft_update_raises_IndexError_when_frame_location_size_with_relevant_data_extends_beyond_curr_data_size(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    cut.curr_data = [unchanged_data] * (num_fake_total_steps - 1) # - 1 ensures less than with min of 1

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite')
    mocker.patch.object(cut.test_suite, 'get_suite_status', return_value=fake_suite_status)
    mocker.patch.object(cut.status, 'set_status')
    
    # Act
    with pytest.raises(IndexError) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match('list assignment index out of range')
    
def test_Spacecraft_update_does_not_raise_IndexError_when_frame_location_size_with_only_empty_steps_extends_beyond_curr_data_size(mocker):
    # Arrange
    arg_frame = []

    num_fake_total_steps = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10 (must have 2 to have at least one of each type)
    for i in range(num_fake_total_steps):
        arg_frame.append(MagicMock())

    cut = Spacecraft.__new__(Spacecraft)
    cut.test_suite = MagicMock()
    cut.status = MagicMock()
    unchanged_data = MagicMock()
    fake_curr_data_size = pytest.gen.randint(1, num_fake_total_steps - 1) # from 1 to 1 less than total
    cut.curr_data = [unchanged_data] * fake_curr_data_size

    for i in range(fake_curr_data_size, len(arg_frame)):
        arg_frame[i] = '-'

    fake_suite_status = []
    for i in range(pytest.gen.randint(1, 10)): # arbitrary, from 1 to 10 status items
        fake_suite_status.append(MagicMock())
    fake_non_IndexError_message = str(MagicMock())

    mocker.patch.object(cut.test_suite, 'execute_suite', side_effect=Exception(fake_non_IndexError_message)) # testing short circuit that is provable to not be the IndexError

    # Act
    with pytest.raises(Exception) as e_info:
        cut.update(arg_frame)

    # Assert
    assert e_info.match(fake_non_IndexError_message)

# get_headers

# get_current_faulting_mnemonics tests, return_value=fake_suite_status

# get_current_data tests

# get_current_time tests

# get_status tests

# get_bayesian_status tests

# get_batch_status_reports tests

# class TestSpacecraft(unittest.TestCase):

#     def setUp(self):
#         self.test_path = os.path.dirname(os.path.abspath(__file__))
#         self.SC = Spacecraft(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])

#     def test_init_empty_spacecraft(self):
#         SC = Spacecraft()
#         self.assertEqual(type(SC.status), Status)
#         self.assertEqual(SC.headers, [])
#         self.assertEqual(type(SC.test_suite), TelemetryTestSuite)
#         self.assertEqual(SC.curr_data, [])

#     def test_init_nonempty_spacecraft(self):
#         hdrs = ['TIME', 'A', 'B'] 
#         tests = [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]]

#         SC = Spacecraft(hdrs, tests)

#         self.assertEqual(type(SC.status), Status)
#         self.assertEqual(SC.headers, ['TIME', 'A', 'B'])
#         self.assertEqual(type(SC.test_suite), TelemetryTestSuite)
#         self.assertEqual(SC.curr_data, ['-', '-', '-'])

#     def test_update(self):
#         frame = [3, 1, 5]
#         self.SC.update(frame)
#         self.assertEqual(self.SC.get_current_data(), [3, 1, 5])

#         frame = [4, '-', 5]
#         self.SC.update(frame)
#         self.assertEqual(self.SC.get_current_data(), [4, 1, 5])

#     def test_get_current_time(self):
#         self.assertEqual(self.SC.get_current_time(), '-'), return_value=fake_suite_status

#     def tests_get_status(self):
#         self.assertEqual(self.SC.get_status(), '---')

#     def tests_get_bayesian_status(self):
#         self.assertEqual(self.SC.get_bayesian_status(), ('---', -1.0))



# if __name__ == '__main__':
#     unittest.main()
