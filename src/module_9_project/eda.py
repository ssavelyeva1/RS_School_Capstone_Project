import pandas as pd
from pandas_profiling import ProfileReport
from module_9_project.data import get_dataset

df = get_dataset()
profile = ProfileReport(df=df)
profile.to_file('pandas_profile_test.html')