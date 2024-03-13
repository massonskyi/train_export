import os
import subprocess
from utils import Tools

FILENAME = os.getcwd() + "requirements.txt"

@Tools.timeit
def read_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


requirements = read_requirements(FILENAME)

for requirement in requirements:
    subprocess.check_call(['pip', 'install', requirement])

# end install rknn toolkit2

wheel_file = "rknn_toolkit2-1.6.0+81f21f4d-cp311-cp311-linux_x86_64.whl"

subprocess.run(["pip", "install", wheel_file])