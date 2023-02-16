""" Test Kalman Core Functionality """

import pytest
from mock import MagicMock
import src.data_driven_components.kalman.core as core
from src.data_driven_components.kalman.core import AIPlugIn

import importlib

# test init
def test_AIPlugIn__init__initializes_variables_to_expected_values_when_given_all_args_except_window_size(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()

    fake_var = MagicMock()
    class Fake_kalman():
        def __init__(self):
            self.test_var = fake_var

    mocker.patch('src.data_driven_components.kalman.core.Kalman', Fake_kalman)

    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    cut.__init__(arg_name, arg_headers)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == 3
    assert cut.agent.test_var == fake_var
  
def test_AIPlugIn__init__initializes_variables_to_expected_values_when_given_all_args(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_headers = MagicMock()
    arg_window_size = MagicMock()

    fake_var = MagicMock()
    class Fake_kalman():
        def __init__(self):
            self.test_var = fake_var

    mocker.patch('src.data_driven_components.kalman.core.Kalman', Fake_kalman)

    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    cut.__init__(arg_name, arg_headers, arg_window_size)

    # Assert
    assert cut.frames == []
    assert cut.component_name == arg_name
    assert cut.headers == arg_headers
    assert cut.window_size == arg_window_size
    assert cut.agent.test_var == fake_var
    
# test apiori training
def test_AIPlugIn_apiori_training_returns_none():
    # Arrange
    cut = AIPlugIn.__new__(AIPlugIn)

    # Act
    result = cut.apriori_training()  

    # Assert
    assert result == None

# test update
def test_AIPlugIn_update_does_not_mutate_frames_attribute_when_arg_frame_is_empty():
    # Arrange
    fake_frames = MagicMock()
    arg_frame = []

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.frames = fake_frames

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == fake_frames

def test_AIPlugIn_update_mutates_frames_attribute_as_expected_when_frames_is_empty_and_arg_frame_is_not_empty():
    # Arrange
    fake_frames = []
    len_arg_frame = pytest.gen.randint(1, 10) # arbitrary, random integer from 1 to 10
    arg_frame = [MagicMock()] * len_arg_frame

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.frames = fake_frames

    expected_result = []
    for data_pt in arg_frame:
        expected_result.append([data_pt])

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

def test_AIPlugIn_update_mutates_frames_attribute_as_expected_when_both_frames_and_arg_frame_are_not_empty_and_len_arg_frame_greater_than_len_frames():
    # Arrange
    len_fake_frames = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    
    len_arg_frame = pytest.gen.randint(6, 10) # arbitrary int greater than max len of fake_frames, from 6 to 10
    arg_frame = [MagicMock()] * len_arg_frame

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    len_dif = len_arg_frame - len_fake_frames
    expected_result = fake_frames.copy()

    for i in range(len_dif):
        expected_result.append([arg_frame[i]])

    for i in range(len_dif, len_arg_frame):
        expected_result[i].append(arg_frame[i])
        if len(expected_result[i]) > fake_window_size:
            expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

def test_AIPlugIn_update_mutates_frames_attribute_as_expected_when_both_frames_and_arg_frame_are_not_empty_and_len_arg_frame_less_than_len_frames():
    # Arrange
    len_fake_frames = pytest.gen.randint(6, 10) # arbitrary int greater than max len of arg_frame, from 6 to 10
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = pytest.gen.randint(1, 10) # arbitrary, random int from 1 to 10
    
    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = [MagicMock()] * len_arg_frame

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    expected_result = fake_frames.copy()
    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])
        if len(expected_result[i]) > fake_window_size:
            expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result


def test_AIPlugIn_update_pops_first_index_of_frames_data_points_when_window_size_is_exceeded():
    # Arrange
    len_fake_frames = pytest.gen.randint(6, 10) # arbitrary int greater than max len of arg_frame, from 6 to 10
                                                # choosing to keep len of fake_frames greater than arg_frame in order to guarantee 'popping'
    fake_frames = [[MagicMock()]] * len_fake_frames
    fake_window_size = 1 # arbitrary, chosen to guarantee 'popping'

    len_arg_frame = pytest.gen.randint(1, 5) # arbitrary, random int from 1 to 5
    arg_frame = [MagicMock()] * len_arg_frame

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.frames = fake_frames
    cut.window_size = fake_window_size

    expected_result = fake_frames.copy()

    for i in range(len_arg_frame):
        expected_result[i].append(arg_frame[i])
        expected_result[i].pop(0)

    # Act
    cut.update(arg_frame)

    # Assert
    assert cut.frames == expected_result

# test render diagnosis
def test_AIPlugIn_render_diagnosis_returns_value_returned_by_agent_frame_diagnose_function(mocker):
    # Arrange
    fake_agent = MagicMock()
    fake_frames = MagicMock()
    fake_headers = MagicMock()
    forced_frame_diagnose_return = MagicMock()

    mocker.patch.object(fake_agent, 'frame_diagnose', return_value=forced_frame_diagnose_return)

    cut = AIPlugIn.__new__(AIPlugIn)
    cut.agent = fake_agent
    cut.frames = fake_frames
    cut.headers = fake_headers
    
    # Act
    result = cut.render_diagnosis()

    # Assert
    assert result == forced_frame_diagnose_return