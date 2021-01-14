'''
Defines paths for all the things.
'''

import os
import pathlib

parentd = str(pathlib.Path(__file__).parent.absolute())[:-8]
DATADIR = os.path.join(parentd,
        'data/')
RESULTSDIR = os.path.join(parentd, 'results/')
FIGSDIR = os.path.join(parentd, 'results/paperfigs/')
CHAINSDIR = os.path.join(parentd, 'results/chains/')
DIAGNOSTICDIR = os.path.join(parentd,
        'results/diagnostic_plots/')

alldirdict = {key:value for (key,value) in locals().items() if 'DIR' in key}

for i in alldirdict.values():
    if not os.path.exists(i):
        print("Creating needed directory:", i)
        os.makedirs(i)
