""" Test 42 Parser Functionality """
import pytest
from mock import MagicMock
import src.data_handling.parsers.forty_two_parser as forty_two_parser
from src.data_handling.parsers.forty_two_parser import FortyTwo

# tests for init
def test_forty_two_init_default_constructor_initializes_variables_to_empty_strings():
    # Arrange
    cut = FortyTwo.__new__(FortyTwo)

    # Act
    cut.__init__()

    # Assert
    assert cut.raw_data_file_path == ''
    assert cut.metadata_file_path == ''
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == ''

def test_forty_two_init_initializes_variables_correctly_when_dataFiles_arg_is_empty_string(mocker):
    # Arrange    
    arg_data_file_path = str(MagicMock())
    arg_metadata_file_path = str(MagicMock())
    arg_data_files = ''
    arg_config_files = str(MagicMock())

    cut = FortyTwo.__new__(FortyTwo)

    mocker.patch.object(cut, 'parse_sim_data')

    # Act
    cut.__init__(arg_data_file_path, arg_metadata_file_path, arg_data_files, arg_config_files)

    # Assert
    assert cut.raw_data_file_path == arg_data_file_path
    assert cut.metadata_file_path == arg_metadata_file_path
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == ''
    assert cut.parse_sim_data.call_count == 0

def test_forty_two_init_initializes_variables_correctly_when_configFiles_arg_is_empty_string(mocker):
    # Arrange    
    arg_data_file_path = str(MagicMock())
    arg_metadata_file_path = str(MagicMock())
    arg_data_files = str(MagicMock())
    arg_config_files = ''

    cut = FortyTwo.__new__(FortyTwo)
    
    mocker.patch.object(cut, 'parse_sim_data')

    # Act
    cut.__init__(arg_data_file_path, arg_metadata_file_path, arg_data_files, arg_config_files)

    # Assert
    assert cut.raw_data_file_path == arg_data_file_path
    assert cut.metadata_file_path == arg_metadata_file_path
    assert cut.all_headers == ''
    assert cut.sim_data == ''
    assert cut.binning_configs == ''
    assert cut.parse_sim_data.call_count == 0

def test_forty_two_init_initializes_values_correctly_when_given_non_empty_arguments_and_ss_breakdown_is_true(mocker):
    # Arrange
    fake_data_file_name = str(MagicMock())
    fake_headers = MagicMock()
    fake_sim_data = MagicMock()
    fake_list_len = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_list = [MagicMock()] * fake_list_len
    
    arg_data_file_path = str(MagicMock())
    arg_metadata_file_path = str(MagicMock())
    arg_data_files = str([fake_data_file_name])
    arg_config_files = str(MagicMock())

    forced_parse_sim_data_return_value = fake_headers, fake_sim_data
    forced_parse_config_data_return_value = { 'subsystem_assignments' : {fake_data_file_name:MagicMock()},
                                                'test_assignments' : {fake_data_file_name:MagicMock()},
                                                'description_assignments' : {fake_data_file_name:MagicMock()}}

    cut = FortyTwo.__new__(FortyTwo)

    mocker.patch('src.data_handling.parsers.forty_two_parser.str2lst', return_value=fake_list)
    mocker.patch.object(cut, 'parse_sim_data', return_value=forced_parse_sim_data_return_value)
    mocker.patch.object(cut, 'parse_config_data', return_value=forced_parse_config_data_return_value)
    
    # Act
    cut.__init__(arg_data_file_path, arg_metadata_file_path, arg_data_files, arg_config_files, True)

    # Assert
    assert cut.raw_data_file_path == arg_data_file_path
    assert cut.metadata_file_path == arg_metadata_file_path
    assert cut.all_headers == fake_headers
    assert cut.sim_data == fake_sim_data
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.parse_sim_data.call_count == 1
    assert cut.parse_sim_data.call_args_list[0].args == (fake_list[0],)
    assert cut.parse_config_data.call_count == 1
    assert cut.parse_config_data.call_args_list[0].args == (fake_list[0], True)

def test_forty_two_init_initializes_variables_correctly_when_given_arguments_and_ss_breakdown_is_false(mocker):
    # Arrange
    fake_data_file_name = str(MagicMock())
    fake_headers = MagicMock()
    fake_sim_data = MagicMock()
    fake_list_len = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_list = [MagicMock()] * fake_list_len

    arg_data_file_path = str(MagicMock())
    arg_metadata_file_path = str(MagicMock())
    arg_data_files = str([fake_data_file_name])
    arg_config_files = str(MagicMock())

    forced_parse_sim_data_return_value = fake_headers, fake_sim_data
    forced_parse_config_data_return_value = { 'subsystem_assignments' : {fake_data_file_name:MagicMock()},
                                                'test_assignments' : {fake_data_file_name:MagicMock()},
                                                'description_assignments' : {fake_data_file_name:MagicMock()}}
    
    cut = FortyTwo.__new__(FortyTwo)

    mocker.patch('src.data_handling.parsers.forty_two_parser.str2lst', return_value=fake_list)
    mocker.patch.object(cut, 'parse_sim_data', return_value=forced_parse_sim_data_return_value)
    mocker.patch.object(cut, 'parse_config_data', return_value=forced_parse_config_data_return_value)
    
    # Act
    cut.__init__(arg_data_file_path, arg_metadata_file_path, arg_data_files, arg_config_files, False)

    # Assert
    assert cut.raw_data_file_path == arg_data_file_path
    assert cut.metadata_file_path == arg_metadata_file_path
    assert cut.all_headers == fake_headers
    assert cut.sim_data == fake_sim_data
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.binning_configs == forced_parse_config_data_return_value
    assert cut.parse_sim_data.call_count == 1
    assert cut.parse_sim_data.call_args_list[0].args == (fake_list[0],)
    assert cut.parse_config_data.call_count == 1
    assert cut.parse_config_data.call_args_list[0].args == (fake_list[0], False)

# tests for parse sim data
def test_forty_two_parse_sim_data_raises_index_error_when_given_empty_data_file(mocker):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_file_path = MagicMock()
    fake_data_str = ''
    fake_headers = []

    cut = FortyTwo.__new__(FortyTwo)
    cut.raw_data_file_path = fake_file_path

    mocker.patch('src.data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(cut, 'parse_headers', return_value=fake_headers)

    # Act
    with pytest.raises(IndexError) as e_info:
        cut.parse_sim_data(arg_data_file)

    # Assert
    assert e_info.match('list index out of range')
    assert forty_two_parser.open.call_count == 1
    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.close.call_count == 1
    assert cut.parse_headers.call_count == 0

def test_forty_two_parse_sim_data_with_only_one_data_pt(mocker):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_file_path = MagicMock()
    data_pt = str(MagicMock())
    fake_data_str = data_pt + '[EOF]\n\n'
    data_pts = [data_pt]
    
    # fake headers and frames
    fake_headers = MagicMock()
    num_frame_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_frames = []
    for i in range(num_frame_data):
        fake_frames.append(MagicMock())

    cut = FortyTwo.__new__(FortyTwo)
    cut.raw_data_file_path = fake_file_path

    mocker.patch('src.data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(cut, 'parse_headers', return_value=fake_headers)
    mocker.patch.object(cut, 'parse_frame', return_value=fake_frames)

    # Act
    headers_result, data_result = cut.parse_sim_data(arg_data_file)

    # Assert
    assert headers_result == {arg_data_file : fake_headers}
    assert data_result == {fake_frames[0] : {arg_data_file : fake_frames}}
    assert forty_two_parser.open.call_count == 1
    assert forty_two_parser.open.call_args_list[0].args == (fake_file_path + arg_data_file, "r+")

    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.read.call_args_list[0].args == ()
    assert fake_txt_file.close.call_count == 1
    assert fake_txt_file.close.call_args_list[0].args == ()

    assert cut.parse_headers.call_count == 1
    assert cut.parse_headers.call_args_list[0].args == (data_pts[0],)
    assert cut.parse_frame.call_count == 1
    assert cut.parse_frame.call_args_list[0].args == (data_pts[0],)

def test_forty_two_parse_sim_data_with_more_than_one_data_pt(mocker):
    # Arrange
    arg_data_file = MagicMock()

    fake_txt_file = MagicMock()
    fake_file_path = MagicMock()
    
    # fake data pts and str
    num_data_pts = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    data_pts = []
    fake_data_str = ''
    for i in range(num_data_pts):
        data_pt = str(MagicMock())
        fake_data_str += data_pt + '[EOF]\n\n'
        data_pts.append(data_pt)
    
    # fake headers and frames
    fake_headers = MagicMock()
    num_frame_data = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_frames = []
    for i in range(num_frame_data):
        fake_frames.append(MagicMock())

    cut = FortyTwo.__new__(FortyTwo)
    cut.raw_data_file_path = fake_file_path

    mocker.patch('src.data_handling.parsers.forty_two_parser.open', return_value=fake_txt_file)
    mocker.patch.object(fake_txt_file, 'read',return_value=fake_data_str)
    mocker.patch.object(fake_txt_file, 'close')
    mocker.patch.object(cut, 'parse_headers', return_value=fake_headers)
    mocker.patch.object(cut, 'parse_frame', return_value=fake_frames)

    # Act
    headers_result, data_result = cut.parse_sim_data(arg_data_file)

    # Assert
    assert headers_result == {arg_data_file : fake_headers}
    assert data_result == {fake_frames[0] : {arg_data_file : fake_frames}}
    assert forty_two_parser.open.call_count == 1
    assert forty_two_parser.open.call_args_list[0].args == (fake_file_path + arg_data_file, "r+")
    
    assert fake_txt_file.read.call_count == 1
    assert fake_txt_file.read.call_args_list[0].args == ()
    assert fake_txt_file.close.call_count == 1
    assert fake_txt_file.close.call_args_list[0].args == ()
    
    assert cut.parse_headers.call_count == 1
    assert cut.parse_headers.call_args_list[0].args == (data_pts[0],)
    assert cut.parse_frame.call_count == num_data_pts
    for i in range(num_data_pts):
        assert cut.parse_frame.call_args_list[i].args == (data_pts[i],)

def test_forty_two_parse_headers_for_frame_with_data():
    # Arrange
    num_lines = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    fake_time = MagicMock()
    fake_headers = [MagicMock()] * num_lines

    expected_result = ['<MagicMock']
    arg_frame = str(fake_time)
    for fake_header in fake_headers:
        expected_result.append(str(fake_header))
        arg_frame += '\n' + str(fake_header) + ' = '
    cut = FortyTwo.__new__(FortyTwo)

    # Act
    result = cut.parse_headers(arg_frame)

    # Assert
    assert result == expected_result

# tests for parse frame
def test_forty_two_parse_frame_raises_error_because_of_frame_with_no_data():
    # Arrange
    cut = FortyTwo.__new__(FortyTwo)
    
    # Act
    with pytest.raises(IndexError) as e_info:
        cut.parse_frame('')

    # Assert
    assert e_info.match('list index out of range')


def test_forty_two_parse_frame_for_frame_with_data():
    # Arrange
    num_lines = pytest.gen.randint(1, 10) # arbitrary, from 1 to 10
    
    fake_time = MagicMock()
    fake_data = [MagicMock()] * num_lines

    expected_result = [str(fake_time).removeprefix('<MagicMock ')]

    arg_frame = str(fake_time)
    for fake_datum in fake_data:
        expected_result.append(str(fake_datum))
        arg_frame += '\n = ' + str(fake_datum)
    
    cut = FortyTwo.__new__(FortyTwo)

    # Act
    result = cut.parse_frame(arg_frame)

    # Assert
    assert result == expected_result

# tests for parse headers
def test_forty_two_parse_header_returns_list_with_a_single_empty_string_for_frame_with_no_data():
    # Arrange
    cut = FortyTwo.__new__(FortyTwo)
    
    # Act
    result = cut.parse_headers('')

    # Assert
    assert result == ['']

# tests for parse config data
def test_forty_two_parse_config_data_when_ss_breakdown_is_false_and_only_one_ss_assignment_returns_expected_result(mocker):
    # Arrange
    arg_config_file = MagicMock()

    fake_metadata_file_path = MagicMock()
    fake_filename = MagicMock()
    fake_subsystem_assignments = [MagicMock()]
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    cut = FortyTwo.__new__(FortyTwo)
    cut.metadata_file_path = fake_metadata_file_path

    forced_return_extract_configs = { 'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                                        'test_assignments' : {fake_filename:fake_tests},
                                        'description_assignments' : {fake_filename:fake_descs}}

    mocker.patch('src.data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    mocker.patch('src.data_handling.parsers.forty_two_parser.process_filepath', return_value=fake_filename)

    expected_result = { 'subsystem_assignments' : {fake_filename:[['MISSION']]},
                        'test_assignments' : {fake_filename:fake_tests},
                        'description_assignments' : {fake_filename:fake_descs}}

    # Act
    result = cut.parse_config_data(arg_config_file, False)
        
    # Assert
    assert result == expected_result
    assert forty_two_parser.extract_configs.call_count == 1
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_file_path, [arg_config_file])
    assert forty_two_parser.process_filepath.call_count == 2
    assert forty_two_parser.process_filepath.call_args_list[0].args == (arg_config_file,)
    assert forty_two_parser.process_filepath.call_args_list[1].args == (arg_config_file,)

def test_forty_two_parse_config_data_if_ss_breakdown_is_false_and_number_of_subsystem_assignments_greater_than_one(mocker):
    # Arrange
    arg_config_file = MagicMock()
    fake_metadata_file_path = MagicMock()

    fake_filename = MagicMock()
    num_ss_assignments = pytest.gen.randint(2, 10) # arbitrary, from 2 to 10
    fake_subsystem_assignments = [MagicMock()] * num_ss_assignments
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    cut = FortyTwo.__new__(FortyTwo)
    cut.metadata_file_path = fake_metadata_file_path

    forced_return_extract_configs = { 'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                                        'test_assignments' : {fake_filename:fake_tests},
                                        'description_assignments' : {fake_filename:fake_descs}}
    mocker.patch('src.data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    forced_return_process_filepath = fake_filename
    mocker.patch('src.data_handling.parsers.forty_two_parser.process_filepath', return_value=forced_return_process_filepath)

    expected_result = { 'subsystem_assignments' : {fake_filename:[['MISSION']] * num_ss_assignments},
                        'test_assignments' : {fake_filename:fake_tests},
                        'description_assignments' : {fake_filename:fake_descs}}

    # Act
    result = cut.parse_config_data(arg_config_file, False)
        
    # Assert
    assert result == expected_result
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_file_path, [arg_config_file])
    assert forty_two_parser.process_filepath.call_count == 2
    assert forty_two_parser.process_filepath.call_args_list[0].args == (arg_config_file,)
    assert forty_two_parser.process_filepath.call_args_list[1].args == (arg_config_file,)

def test_forty_two_parse_config_data_if_ss_breakdown_is_true(mocker):
    # Arrange
    arg_config_file = MagicMock()
    fake_metadata_file_path = MagicMock()

    fake_filename = MagicMock()
    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    cut = FortyTwo.__new__(FortyTwo)
    cut.metadata_file_path = fake_metadata_file_path

    forced_return_extract_configs = { 'subsystem_assignments' : {fake_filename:fake_subsystem_assignments},
                                        'test_assignments' : {fake_filename:fake_tests},
                                        'description_assignments' : {fake_filename:fake_descs}}
    mocker.patch('src.data_handling.parsers.forty_two_parser.extract_configs', return_value=forced_return_extract_configs)
    
    # Act
    result = cut.parse_config_data(arg_config_file, True)
        
    # Assert
    assert result == forced_return_extract_configs
    assert forty_two_parser.extract_configs.call_count == 1
    assert forty_two_parser.extract_configs.call_args_list[0].args == (fake_metadata_file_path, [arg_config_file])

# test for get_sim_data
def test_forty_two_get_sim_data_returns_tuple_of_all_headers_and_sim_data_and_binning_configs_without_modifying_values():
    # Arrange
    fake_all_headers = MagicMock()
    fake_sim_data = MagicMock()
    fake_binning_configs = MagicMock()
    
    cut = FortyTwo.__new__(FortyTwo)
    cut.all_headers = fake_all_headers
    cut.sim_data = fake_sim_data
    cut.binning_configs = fake_binning_configs

    # Act
    sim_data = cut.get_sim_data()

    # Assert
    assert sim_data == (fake_all_headers, fake_sim_data, fake_binning_configs)
