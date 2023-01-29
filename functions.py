# Import libraries to API calls
import requests
import json

def getCall(url, apiKey):
    headers = {"Authorization": f"Bearer {apiKey}", "Accept": "application/vnd.api+json"}
    response = requests.get(url, headers=headers)
    return response.json()

def getMatchTelemetryID(platform, match_id, apiKey):
    url = f"https://api.pubg.com/shards/{platform}/matches/{match_id}"
    headers = {"Authorization": f"Bearer {apiKey}", "Accept": "application/vnd.api+json"}
    response = requests.get(url, headers=headers)

    return response.json()["data"]["relationships"]["assets"]["data"][0]["id"]

