import numpy as np

def dist(obj1, obj2):
    return np.sqrt((obj2.x - obj1.x)**2 + (obj2.y - obj1.y)**2 + (obj2.z - obj1.z)**2)
    