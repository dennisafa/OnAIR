""" Test Brain Functionality """
import os
import unittest

from src.systems.spacecraft import Spacecraft
from src.reasoning.brain import Brain

class TestBrain(unittest.TestCase):

    def setUp(self):
        self.test_path = os.path.dirname(os.path.abspath(__file__))
        SC = Spacecraft(['TIME', 'A', 'B'], [[['SYNC', 'TIME']], [['NOOP']], [['NOOP']]])
        self.B = Brain(SC)

    def test_init_nonempty_brain(self):
        self.assertEqual(type(self.B.spacecraft_rep), Spacecraft)
        self.assertEqual(self.B.mission_status, '---')
        self.assertEqual(self.B.bayesian_status, ('---', -1.0))

    def test_reason(self):
        frame = [1, 1, 1]
        self.B.reason(frame)
        self.assertEqual(self.B.spacecraft_rep.get_current_data(), [1,1,1])
        self.assertEqual(self.B.mission_status, 'GREEN')

    def test_diagnose(self):
        return
        

if __name__ == '__main__':
    unittest.main()
