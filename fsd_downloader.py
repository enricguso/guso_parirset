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


words = ['crowd', 'applause', 'audience', 'cheering', 'chatter', 'protest']

licenses = ['http://creativecommons.org/licenses/by/3.0/',
 'http://creativecommons.org/publicdomain/zero/1.0/',
 'https://creativecommons.org/licenses/by-nc/4.0/',
 'https://creativecommons.org/licenses/by/4.0/']


data = []
ids_list = []
i = 0
if os.path.exists('fsd_crowd_sounds.xlsx'):
    print('Loading existing data...')
    df_existing = pd.read_excel('fsd_crowd_sounds.xlsx')
    for i in range(len(df_existing)):
        data.append({
                    'id': df_existing.loc[i, 'id'],
                    'type': df_existing.loc[i, 'type'],
                    'query': df_existing.loc[i, 'query'], 
                    'name': df_existing.loc[i, 'name'],
                    'url': df_existing.loc[i, 'url'], 
                    'license': df_existing.loc[i, 'license'],  
                    'username': df_existing.loc[i, 'username']
                    })
    i +=1 
    print(f'Starting from index {i}')
    ids_list = df_existing['id'].tolist()

for word in words: 

    os.makedirs('downloads/'+word, exist_ok=True)

    try:
        results_pager = client.search(
            query=word,
            sort="rating_desc",
            fields="id,type,name,download,license,username,duration"
        )
    except:
        print(f'Error searching for word {word}, skipping it.')
        continue    

    while results_pager.next is not None:
        for sound in tqdm.tqdm(results_pager):
            if sound.duration < 480.0:
                if sound.id not in ids_list:
                    if sound.license in licenses:
                        try:
                            filename = sound.name.replace(" ", "_").replace("/", "_")
                            if filename[-3:] not in ['wav', 'aif', 'iff', 'ogg', 'mp3', 'm4a', 'lac']:
                                filename = filename + '.' + sound.type
                            g = oauth.get(sound.download)

                            with open('downloads/'+ word +'/'+'{:07d}'.format(i)+'_'+filename, "wb") as f:
                                for chunk in g.iter_content():
                                    if chunk:
                                        f.write(chunk)
                            data.append({
                                        'id': sound.id,
                                        'type': sound.type,
                                        'query': word, 
                                        'name': filename, 
                                        'url': sound.download, 
                                        'license': sound.license, 
                                        'username': sound.username})
                            df = pd.DataFrame(data)
                            df.to_excel('fsd_crowd_sounds.xlsx')#, index=False)
                            print(f'Downloaded {i+1} sounds so far...')
                            i += 1
                            ids_list.append(sound.id)
                        except:
                            print('Error downloading sound with id '+str(sound.id))
        try:
            results_pager = results_pager.next_page()
        except:
            print('Error advancing to next page, going to next word.')
            break

    df = pd.DataFrame(data)
    df.to_excel('fsd_crowd_sounds.xlsx')#, index=False)

print('Total number of crowd sounds: '+str(len(ids_list)))
print('DONE!')
