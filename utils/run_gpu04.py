import os
import socket


def run_gpu04():
    if socket.gethostname() == 'mu01':
        print(os.system('ssh gpu04'))
    # os.system('conda activate torch')
    # print('conda activate torch')
