import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " cnn_generation_with_xsum_pretrained.py> res.txt " +
          "' &")
