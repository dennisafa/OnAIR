"""
Status Class
"""

class Status:
    def __init__(self, name='MISSION', stat='---', conf=-1.0):
        self.name =  name
        self.set_status(stat, conf)

    ##### GETTERS & SETTERS ##################################
    def set_status(self, stat, bayesianConf=-1.0):
        assert(-1.0 <= bayesianConf <= 1.0)
        assert(stat in ['---', 'RED', 'YELLOW', 'GREEN'])
        self.status = stat
        self.bayesian_conf = bayesianConf

    def get_status(self):
        return self.status

    def get_bayesian_status(self):
        return (self.status, self.bayesian_conf)

    def get_name(self):
        return self.name
