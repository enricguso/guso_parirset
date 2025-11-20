import os
import sys
import freesound
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import tqdm
api_key = os.getenv('FREESOUND_API_KEY', None)
from requests_ratelimiter import LimiterSession
import pandas as pd

if api_key is None:
    print("You need to set your API key as an environment variable")
    print("named FREESOUND_API_KEY")
    sys.exit(-1)

freesound_client = freesound.FreesoundClient()
freesound_client.set_token(api_key)

freesound_client.session = LimiterSession(per_minute=59)

words = ['crowd', 'applause', 'audience', 'cheering', 'chatter', 'protest']

licenses = ['http://creativecommons.org/licenses/by/3.0/',
 'http://creativecommons.org/publicdomain/zero/1.0/',
 'https://creativecommons.org/licenses/by-nc/4.0/',
 'https://creativecommons.org/licenses/by/4.0/']

urls_crowd = []
data = []

for word in words: 
    mindx = len(urls_crowd)
    results_pager = freesound_client.text_search(
        query=word,
        sort="rating_desc",
        fields="name,download,license,username"
    )

    while results_pager.next is not None:

        for i, sound in enumerate(results_pager):
            iidx = len(urls_crowd)
            if sound.download not in urls_crowd:
                if sound.license in licenses:
                    print(str(i+iidx-mindx)+'    '+sound.download)
                    urls_crowd.append(sound.download)
                    data.append({'rating_pos': i+iidx-mindx, 'word': word, 'name': sound.name, 'url': sound.download, 'license': sound.license, 'username': sound.username})
        try:
            results_pager = results_pager.next_page()
        except:
            print('Error advancing to next page, stopping here.')
            #print(results_pager.next)
            break
    df = pd.DataFrame(data)
    df.to_csv('fsd_crowd_sounds2.csv', index=False)
    
print('Total number of crowd sounds: '+str(len(urls_crowd)))
print('DONE!')
