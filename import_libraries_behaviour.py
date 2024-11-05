"""
This script contains import statements for the libraries to be used in the analysis of the behavioural context of forest elephant rumbles.

"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import umap.umap_ as umap
import sys
sys.path.insert(0, '..')
import seaborn as sns
import scipy
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from IPython.display import Image, Audio, display
import random
import plotly
from plotly.offline import iplot, plot
from plotly import graph_objs as go
from sklearn import metrics
import tensorflow_hub as hub
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score, silhouette_samples
from sklearn.cluster import AffinityPropagation
import itertools
import random
random.seed(10)


print("Main libraries successfully imported")

