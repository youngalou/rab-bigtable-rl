import numpy as np
import time

class TimeLogger():
    def __init__(self, log_names, num_cycles=1):
        self.log_names = log_names
        self.num_loggers = len(log_names)
        self.num_cycles = num_cycles
        self.reset()

    def reset(self):
        self.loggers = np.zeros(self.num_loggers)
        self.start_time = time.time()
        
    def log(self, logger_id):
        current_time = time.time()
        self.loggers[logger_id] += current_time - self.start_time
        self.start_time = current_time
    
    def print_logs(self):
        print("-- Time Logs --")
        for log_name, logger in zip(self.log_names, self.loggers):
            print("| {}: {} |".format(log_name, logger/self.num_cycles))