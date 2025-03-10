from Params import *

import subprocess
import time
import os,shutil

starttime = time.time()


plist = []

for F in Flist:
    p = subprocess.Popen(['nice','-n','19','python','Script.py','-F',str(F)])
    plist.append(p)

for p in plist:
    p.wait()

endtime = time.time()

print("total time:",endtime -starttime)
