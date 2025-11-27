from requests_oauthlib import OAuth2Session
import pandas as pd
from requests_ratelimiter import LimiterSession
import tqdm
import freesound
import os
import time

client_id = 'put your client id here'
client_secret = 'put your client secret here'


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


words = ['crowd', 'applause', 'audience', 'cheering', 'chatter', 'protest'] # the queries to search for

# only download sounds with these licenses
licenses = ['http://creativecommons.org/licenses/by/3.0/',
 'http://creativecommons.org/publicdomain/zero/1.0/',
 'https://creativecommons.org/licenses/by-nc/4.0/',
 'https://creativecommons.org/licenses/by/4.0/',
 'http://creativecommons.org/licenses/by-nc/3.0/',
 'http://creativecommons.org/licenses/sampling+/1.0/']


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
    print(word)
    os.makedirs('downloads/'+word, exist_ok=True)

    try:
        results_pager = client.search(
            query=word,
            sort="rating_desc",
            fields="id,type,name,download,license,username,duration"
        )
    except:
        print('error paging')
        time.sleep(60)
        continue



    while results_pager.next is not None:
        for sound in tqdm.tqdm(results_pager):
            try:
                if sound.duration < 480.0:
                    if sound.id not in ids_list:
                        if sound.license in licenses:
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
                        else:
                            print(f'Sound id {sound.id} has license {sound.license}, skipping...')
                    else:
                        print(f'Sound id {sound.id} already downloaded, skipping...')
                else:
                    print(f'Sound id {sound.id} is too long ({sound.duration} seconds), skipping...')
            except:
                time.sleep(60)
                print('error')
                continue
        try:
            results_pager = results_pager.next_page()
        except:
            print('error advancing page, sleeping 60 seconds...')
            time.sleep(60)
            print('error')
            continue


    df = pd.DataFrame(data)
    df.to_excel('fsd_crowd_sounds.xlsx')#, index=False)

print('Total number of crowd sounds: '+str(len(ids_list)))
print('DONE!')
