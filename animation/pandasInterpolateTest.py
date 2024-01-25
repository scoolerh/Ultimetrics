import pandas as pd
import numpy as np

#testing
tester = pd.Series([0,1,2, -1 , 4, 5 ,6])
print(tester)
tester = tester.replace(-1, np.nan)
print(tester)
tester = tester.interpolate()
print(tester)
