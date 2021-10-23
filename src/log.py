import os
import uuid
import time
class Logger:
    def __init__(self, args, time_format="%a %b %d %H:%M:%S %Y"):
        self.args = args
        self.log_name = time.strftime("%Y%m%d-%H%M%S") + str(uuid.uuid4())[:8]
        args.time_format = time_format
        self.time_list = []
        self.time_format = time_format

    def info(self, message):
        time_now = time.strftime(self.time_format, time.localtime())
        message = "[info]%s:\t%s" % (time_now, message)
        self.time_list.append(message)
        print(message)
        self.save_log()

    def save_log(self):
        save_logs_file = os.path.join(self.args.log_path, "logs_%s.txt" % self.log_name)
        with open(save_logs_file, mode="w", encoding="utf-8") as wfp:
            for line in self.time_list:
                wfp.write(line + "\n")
