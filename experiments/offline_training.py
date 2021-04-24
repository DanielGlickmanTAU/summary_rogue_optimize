import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_rouge.py > res_xsum_psearch_50k_8beams.txt " +
          "' &")
