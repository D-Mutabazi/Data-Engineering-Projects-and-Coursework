import requests
import json
import pandas as pd
# read data from api
url = "https://movie-database-alternative.p.rapidapi.com/"

querystring = {"s":"Avengers Endgame","r":"json","page":"1"}

headers = {
	"x-rapidapi-key": "93e68e6474msh76e0475701fb48dp172559jsn63ac4bbda213",
	"x-rapidapi-host": "movie-database-alternative.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

if response.status_code == 200:
    json_data = response.json() #parse response
    # extract  data
    data = json_data.get('Search', [])

formattted_data = json.dumps(data, indent=4)
# process the data store in db

print(formattted_data)