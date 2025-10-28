from pathlib import Path
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# path = Path('weather_data/sitka_weather_07-2021_simple.csv')
path = Path('sitka_weather_2021_full.csv')
lines = path.read_text().splitlines()

reader = csv.reader(lines)
header_row = next(reader)

"""
print(header_row)

for index, column_header in enumerate(header_row):
    print(index, column_header)
"""

# 提取日期和最高温度
dates, highs, lows = [], [], []
for row in reader:
    current_date = datetime.strptime(row[2], '%Y/%m/%d')
    # high = int(row[4])
    high = int(row[7])
    low = int(row[8])
    dates.append(current_date)
    highs.append(high)
    lows.append(low)

# print(highs)

plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(dates, highs, color='red')
ax.plot(dates, lows, color='blue')

# 设置绘图格式
# ax.set_title("Daily High Temperatures, July 2021", fontsize=24)
# ax.set_title("Daily High Temperatures, 2021", fontsize=24)
ax.set_title("Daily High and Low Temperatures, 2021", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Temperature (F)", fontsize=16)
ax.tick_params(labelsize=16)

plt.show()