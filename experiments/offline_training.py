import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_rouge.py.py >res1.txt " +
          "' &")
