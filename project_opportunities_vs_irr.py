# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:54:18 2019

@author: IST_1
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

irr = pd.read_excel('project_opportunities_vs_irr.xlsx')

irr = irr[0:21]
irr['Projected IRR (20 years)'] = irr['Projected IRR (20 years)']*100
btm = irr.loc[irr.app=='BTM']
masmart = irr.loc[irr.app=='MASMART']
nwa = irr.loc[irr.app=='NWA ']
ppa = irr.loc[irr.app=='PPA']
vder = irr.loc[irr.app=='VDER']
wholesale = irr.loc[irr.app=='WHOLESALE']

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot()
plt.scatter(btm['Projected IRR (20 years)'],btm['Project Size (kWh)'])
plt.scatter(vder['Projected IRR (20 years)'],vder['Project Size (kWh)'])
plt.scatter(masmart['Projected IRR (20 years)'],masmart['Project Size (kWh)'])
plt.scatter(ppa['Projected IRR (20 years)'],ppa['Project Size (kWh)'])
plt.scatter(wholesale['Projected IRR (20 years)'],wholesale['Project Size (kWh)'])
plt.scatter(nwa['Projected IRR (20 years)'],nwa['Project Size (kWh)'])

plt.grid(axis='y', alpha=0.5)
plt.legend(['BTM','VDER','MASMART','PPA','WHOLESALE','NWA'])
plt.title('Project Pipeline')
plt.xlabel('Projected IRR')
plt.ylabel('Project Size (kWh)')

ax.xaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))