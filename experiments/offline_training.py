import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_rouge_generation.py > res.txt " +
          "' &")
