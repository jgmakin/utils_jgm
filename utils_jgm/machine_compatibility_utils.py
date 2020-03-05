import socket
import os
import subprocess
import platform
import json

'''
:Author: J.G. Makin (except where otherwise noted)
Created: 12/09/17
'''


class MachineCompatibilityUtils:

    def __init__(self):

        self.machine_name = socket.gethostname()
        cfg = os.path.join(
            os.path.expanduser('~'), '.config', 'master_jgm.json')
        if not os.path.isfile(cfg):
            cfg = os.path.join(os.path.dirname(__file__), 'utils_config.json')
        with open(cfg) as file:
            config_dict = json.load(file)
        self.path_dictionary = config_dict['paths']

        # number of GPUs
        try:
            self.num_GPUs = subprocess.check_output(
                ['nvidia-smi', '--list-gpus']).decode('utf-8').count('\n')
        except:
            self.num_GPUs = 0

        # number of CPUs
        if os.name == 'nt':
            _, num_CPUs_str = subprocess.getstatusoutput(
                'echo %NUMBER_OF_PROCESSORS%')
            self.num_CPUs = int(num_CPUs_str)
        elif platform.system() == 'Darwin':
            _, num_CPUs_str = subprocess.getstatusoutput(
                'sysctl -n hw.physicalcpu')
            self.num_CPUs = int(num_CPUs_str)
        else:
            _, num_CPUs_str = subprocess.getstatusoutput('nproc')
            self.num_CPUs = int(num_CPUs_str)

    def get_path(self, dir_type):
        return self.path_dictionary[dir_type]
