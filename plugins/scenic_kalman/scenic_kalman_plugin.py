# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import simdkalman
import numpy as np
import matplotlib.pyplot as plt
from onair.src.ai_components.ai_plugin_abstract.core import AIPlugIn

class Plugin(AIPlugIn):
    def __init__(self, name, headers, window_size=15):
        """
        :param headers: (int) length of time agent examines
        :param window_size: (int) size of time window to examine
        """
        super().__init__(name, headers)
        self.frames = []
        self.component_name = name
        self.headers = headers
        self.window_size = window_size
        self.residual_threshold = 100
        self.file = open('kalman_residuals.txt', 'w')
        self.step = 0

        self.kf = simdkalman.KalmanFilter(
        state_transition = [[1,1],[0,1]],      # matrix A
        process_noise = np.diag([0.1, 0.1]),   # Q
        observation_model = np.array([[1,0]]), # H
        observation_noise = 1.0)               # R

    #### START: Classes mandated by plugin architecture
    def update(self, frame):
        """
        :param frame: (list of floats) input sequence of len (input_dim)
        :return: None
        """
        for data_point_index in range(len(frame)):
            if len(self.frames) < len(frame): # If the frames variable is empty, append each data point in frame to it, each point wrapped as a list
                # This is done so the data can have each attribute grouped in one list before being passed to kalman
                # Ex: [[1:00, 1:01, 1:02, 1:03, 1:04, 1:05], [1, 2, 3, 4, 5]]
                self.frames.append([float(frame[data_point_index])])
            else:
                self.frames[data_point_index].append(float(frame[data_point_index]))
                if len(self.frames[data_point_index]) > self.window_size: # If after adding a point to the frame, that attribute is larger than the window_size, take out the first element
                    self.frames[data_point_index].pop(0)

    def render_reasoning(self):
        """
        System should return its diagnosis
        """
        broken_attributes = self._frame_diagnosis(self.frames, self.headers)
        return broken_attributes
    #### END: Classes mandated by plugin architecture

    # Takes in the kf being used, the data, how many prediction "steps" it will make, and an optional initial value
    # Gives a prediction values based on given parameters
    def _predict(self, subframe, forward_steps, initial_val = None):
        smoothed = self.kf.smooth(subframe, initial_value = initial_val)
        predicted =  self.kf.predict(subframe, forward_steps) # Make a prediction on the smoothed data
        return predicted

    # Get data, make predictions, and then find the errors for these predictions
    def _generate_residuals(self, frame):
        # predict last observation in frame based on all previous observations in frame
        # compute residual based on difference between last observation and KF-smoothed prediction
        # length of frame must be greater than 2 for valid initial and last value and data for KF to smooth
        if len(frame[0]) > 2:
            # generate initial values for frame, use first value for each attribute
            initial_val = np.zeros((len(frame), 2, 1))
            for i in range(len(frame)):
                initial_val[i] = np.array([[frame[i][0], 0]]).transpose()
            predicted = self._predict([data[1:-1] for data in frame], 1, initial_val)
            actual_next_obs = [data[-1] for data in frame]
            pred_mean = [pred for attr in predicted.observations.mean for pred in attr]
            residuals = np.abs(np.subtract(pred_mean, actual_next_obs))
        else:
            residuals = np.zeros((len(frame),)) # return residual of 0 for frames less than or equal to 2
        return residuals

    # Info: takes a frame of data and generates residuals based Kalman filter smoothing
    # Returns error if residual is greater than some threshold
    def _current_frame_get_error(self, frame):
        residuals = self._generate_residuals(frame)
        self.file.write(f"Step {self.step}: {residuals}\n\n")
        self.step = self.step + 1
        errors = residuals > self.residual_threshold
        return errors

    def _frame_diagnosis(self, frame, headers):
        kal_broken_attributes = []
        errors = self._current_frame_get_error(frame)
        for attribute_index, error in enumerate(errors):
            if error and not headers[attribute_index].upper() == 'TIME':
                kal_broken_attributes.append(headers[attribute_index])
        return kal_broken_attributes
