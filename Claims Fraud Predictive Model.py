import pandas as pd
from tkinter import *
from tkinter.filedialog import *

import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import kds
from sklearn import *

filepath = str(askopenfilename())

df = pd.read_excel(filepath)

#print(df.head(25))

