#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#linear algebra
import numpy as np
import pandas as pd

#helper modules
import prepare3
import acquire2
import visuals_telco

import matplotlib.ticker as mtick
import inflection

#visualization tools
import shap
import yellowbrick.classifier 
from yellowbrick.classifier import ROCAUC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


from itertools import cycle, islice
from matplotlib import cm

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix