import numpy as np
from tensorflow.python.client import device_lib

def print_available_devices():
    local_device_protos = [(x.name, x.device_type, x.physical_device_desc)  for x in device_lib.list_local_devices()]
    for device_name, device_type, device_desc in local_device_protos:
        print("Device : {}\n\t type : {}\n\t desc :{}\n".format(device_name, device_type, device_desc))

def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)

def preprocess_HR(x):
    output = np.divide(x.astype(np.float32), 127.5)
    output -= np.ones_like(x,dtype=np.float32)
    return output

def deprocess_HR(x):
    x += np.ones_like(x)
    x *= 127.5
    return np.clip(x, 0, 255)

def deprocess_LR(x):
    x *= 255
    return np.clip(x, 0, 255)

def deprocess_HR_original(x):
    return np.clip((x+np.ones_like(x))*127.5, 0, 255)

def deprocess_LR_original(x):
    return np.clip(x*255, 0, 255)

def get_shape(size, data_format): 
    return (size, size, 3) if data_format=='channels_last' else (3, size, size)
