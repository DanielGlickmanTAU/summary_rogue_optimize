import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_rouge_clone.py > res1_beam.txt " +
          "' &")
