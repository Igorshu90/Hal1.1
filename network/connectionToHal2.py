# --------------------------------- connectionWithHal2  ---------------------------
# Parameters:   ip,port,msg
# Return:       answer from server Hal2
# ---------------------------------------------------------------------------------

import socket
import base64
import pickle
import pandas as pd


from datetime import time, timedelta, datetime


def tdelta_to_time(td):
    return (td + datetime.min).time()

def connectionWithHal2(host='87.71.40.55', port=8080, msg=''):
    while 1:
        try:

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
            sock.connect((host, port))
            msg = msg.encode()
            sock.sendall(msg)
            data = recvall(sock)
            data_decoded = base64.b64decode(data)
            data_unpickled = pickle.loads(data_decoded)


            sock.close()
            break
        except EOFError:
            sock.close()
    return data_unpickled

def recvall(sock):
    BUFF_SIZE = 1000000 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) == 0:
            # either 0 or end of data
            break
    return data


#df = pd.DataFrame(connectionWithHal2(msg='3'))
df = pd.DataFrame((connectionWithHal2(msg= '1')))
df.to_csv('/home/igor/Desktop/bla.csv')
print(df)