import sys
from Record import *

filename = sys.argv[1]
feature = sys.argv[2]
records = Records(filename, feature)
records.dump(filename+'.'+feature+'/'+'instr')
