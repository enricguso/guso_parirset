from requests_oauthlib import OAuth2Session
import pandas as pd
from requests_ratelimiter import LimiterSession
import tqdm
import freesound
import os

client_id = 
client_secret = 

# do the OAuth dance
oauth = OAuth2Session(client_id)

authorization_url, state = oauth.authorization_url(
    "https://freesound.org/apiv2/oauth2/authorize/"
)
print(f"Please go to {authorization_url} and authorize access.")

authorization_code = input("Please enter the authorization code:")
oauth_token = oauth.fetch_token(
    "https://freesound.org/apiv2/oauth2/access_token/",
    authorization_code,
    client_secret=client_secret,
)

client = freesound.FreesoundClient()
client.set_token(oauth_token["access_token"], "oauth")
client.session = LimiterSession(per_minute=59)

df = pd.read_csv('fsd_crowd_sounds.csv')


all_words = df.word.unique().tolist()
for all_word in all_words:
    if not os.path.exists('downloads/'+all_word):
        os.makedirs('downloads/'+all_word)
df.rating_pos = df.rating_pos // 2
downloaded = [False]*len(df)
filenames = [None] * len(df)
df['filenames'] = filenames
df['downloaded'] = downloaded


# RUN1 STARTED AT 0
for i in tqdm.tqdm(range(len(df))): 
    if not df.loc[i, 'downloaded']:
        try:
            g = oauth.get(df.iloc[i].url)
        except:
            print('error querying the API')
            print(df.iloc[i].url)
            print(' ')
            continue
        try:
            df.loc[i, 'name'] = df.loc[i, 'name'].replace(" ", "_").replace("/", "_")
            if '.' in str(df.loc[i, 'name']):
                filename = df.loc[i, 'name']
            else:
                filename = g.headers['Content-Disposition'].split('filename="')[1:][0][:-1].replace(" ", "_").replace("/", "_")
        except:
            query_page = 'https://freesound.org/apiv2/sounds/'+str(df.loc[i,'url'].split('/')[5])+'/'
            fitype = oauth.get(query_page).json()['type']
            filename = df.loc[i, 'name'] + '.' + fitype
        try:
            with open('downloads/'+df.iloc[i].word+'/'+'{:07d}'.format(df.iloc[i].rating_pos)+'_'+filename, "wb") as f:
                for chunk in g.iter_content():
                    if chunk:
                        f.write(chunk)
            df.loc[i, 'downloaded']= True
            df.loc[i, 'filenames'] = filename
        except:
            print('error storing file')
            print('downloads/'+df.iloc[i].word+'/'+'{:07d}'.format(df.iloc[i].rating_pos)+'_'+filename)
            continue
    else:
        print('exists already')

df.to_csv('fsd_crowd_sounds.csv', index=False)