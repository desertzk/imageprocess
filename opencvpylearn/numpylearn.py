import numpy as np


t1=np.arange(12).reshape((3,4)).astype("float")
print(type(t1))
t1[1,1:]=np.nan

print(t1)
