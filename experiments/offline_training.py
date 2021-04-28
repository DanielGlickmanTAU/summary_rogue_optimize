import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_model_loading2.py > res.txt " +
          "' &")
