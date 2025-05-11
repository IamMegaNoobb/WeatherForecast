import requests
from datetime import date
from django.shortcuts import render
from django.http import HttpResponse

tmd_token = ''
date = date.today()

# get computer location using IP address
def get_location(request):
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data.get("loc")  # format: "latitude,longitude"
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error: {http_err}"}
    return loc

# Create your views here.
def index(request):
    return render(request, "main/index.html", {"tmd_api": tmd_api(request)})

def tmd_api(request, date=date):
    lat, lon = get_location(request).split(",")
    url = "https://data.tmd.go.th/nwpapi/v1/forecast/location/daily/at"
    querystring = {"lat":f"{lat}", "lon":f"{lon}", "fields":"tc_max,rh,cond", "date":f"{date}", "duration":"1"}
    headers = {
        'accept': "application/json",
        'authorization': tmd_token,
        }
    response = requests.request("GET", url, headers=headers, params=querystring)
    return response
