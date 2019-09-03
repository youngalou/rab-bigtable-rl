import numpy as np
import time

class TimeLogger():
    def __init__(self, log_ids):
        self.log_ids = log_ids
        values = [(log_id, [0,0]) for log_id in self.log_ids]
        self.loggers = dict(values)
        self.reset()

    def reset(self):
        for log_id in self.log_ids:
            self.loggers[log_id] = [0,0]
        self.start_time = time.time()

    def set_start(self):
        self.start_time = time.time(0)
        
    def log(self, log_id):
        current_time = time.time()
        self.loggers[log_id][0] += current_time - self.start_time
        self.loggers[log_id][1] += 1
        self.start_time = current_time
    
    def print_avgtime_logs(self):
        print("-- Average Time Logs --")
        for log_id in  self.log_ids:
            total_time, num_cycles = self.loggers[log_id]
            display_time = total_time if num_cycles <= 0 else total_time / num_cycles
            print("| {}: {} |".format(log_id, display_time))
        
    def print_totaltime_logs(self):
        print("-- Total Time Logs --")
        for log_id in  self.log_ids:
            total_time = self.loggers[log_id][0]
            print("| {}: {} |".format(log_id, total_time))