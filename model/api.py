def api_call(url: str):
    """Returns the result of an API call.
       This can override issues in api_call2"""

    import urllib.request
    import json
    import pandas as pd
    import io

    #url = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
   
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers={'User-Agent':user_agent,} 

    request=urllib.request.Request(url,None,headers) #The assembled request
    response = urllib.request.urlopen(request)
    data = response.read() # The data u need
    rawData = pd.read_csv(io.StringIO(data.decode('utf-8')))
    
    return rawData