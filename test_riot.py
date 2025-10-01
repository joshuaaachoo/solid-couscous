from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
import os
print("Loaded key:", os.getenv("RIOT_API_KEY"))
print(os.listdir())           # Should show your .env in the list
print(os.getcwd())            # Should match your project/test script dir
print(os.getenv("RIOT_API_KEY"))
import requests

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
print("Using Riot Key:", RIOT_API_KEY)
url = "https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/radpoles/chill"
headers = {"X-Riot-Token": RIOT_API_KEY}
resp = requests.get(url, headers=headers)
print(resp.status_code, resp.text)
