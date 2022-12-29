import numpy as np

def rotx(a):
    ''' rotation about x axis dcm by a radians '''
    return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])

def roty(a):
    ''' rotation about y axis dcm by a radians '''
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])

def rotz(a):
    ''' rotation about z axis dcm by a radians '''
    return np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])

def dcm321(ax, ay, az):
    ''' 
    321 dcm 
        first rotate about z axis by az radians
        then rotate about y axis by ay radians
        finally rotate about x axis by ax radians
    '''
    return rotz(az) @ roty(ay) @ rotx(ax)


