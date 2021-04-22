import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_rouge_clone_cnn.py > res1_beam.txt " +
          "' &")
