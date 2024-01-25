import pandas as pd
import numpy as np

#testing
tester = pd.Series([0,1,2, -1 , 4, 5 ,6])
tester = pd.Series([(0,1), (2,3), (np.nan, np.nan), (4,5),(6,7)])
print(tester)
tester = tester.replace(-1, np.nan)
print(tester)
tester = tester.interpolate()
print(tester)

#note -- cannot interpolate coord-pairs, but maybe we can assume we can split them? shouldn't be too hard
