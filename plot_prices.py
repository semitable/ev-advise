import numpy as np
import pandas as pd
from datetime import timedelta, datetime, time

import matplotlib.pyplot as plt
from matplotlib import dates

df = pd.DataFrame(index=pd.DatetimeIndex(start=datetime(2002, 1, 1), periods=1440, freq='T'))
df['Europe'] = 0.16
df.loc[(df.index.time > time(hour=13, minute=00)) & (
	df.index.time < time(hour=16, minute=00)), 'Europe'] = 0.107

df.loc[(df.index.time > time(hour=20, minute=00)) & (
	df.index.time < time(hour=22, minute=00)), 'Europe'] = 0.107

df.loc[(df.index.time > time(hour=00, minute=00)) & (
	df.index.time < time(hour=5, minute=00)), 'Europe'] = 0.107

df['US'] = 0.202
df.loc[(df.index.time > time(hour=8, minute=30)) & (
	df.index.time < time(hour=21, minute=30)), 'US'] = 0.230
df.loc[(df.index.time > time(hour=12, minute=00)) & (
	df.index.time < time(hour=18, minute=00)), 'US'] = 0.253

min_price = 0.202

plt.plot(df.index, df['Europe'], 'b-')
plt.plot(df.index, df['US'], 'r-')

current_axes = plt.gca()
current_figure = plt.gcf()

hfmt = dates.DateFormatter('%I:%M %p', )
current_axes.xaxis.set_major_formatter(hfmt)

plt.ylabel('$/kWh')
# plt.yticks(range(0,24))
plt.ylim(0.0, 0.3)
plt.locator_params(nticks=4)

plt.xlabel('Time of Day')
plt.xticks(rotation='0')
