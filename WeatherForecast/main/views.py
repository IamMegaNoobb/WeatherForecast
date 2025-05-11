import requests
from datetime import date
from django.shortcuts import render
from django.http import HttpResponse

tmd_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjZjYWQ4OWQyMjUxZGQ3NzMxYjk3M2M2ZTk4NTE2NDE4ZTliMmQ0ZDE1NzQ5ZTZkZDA1NzEyZDNjNTRhOTQ5YzRhYTk4NmY5ZmQyYTczZTk4In0.eyJhdWQiOiIyIiwianRpIjoiNmNhZDg5ZDIyNTFkZDc3MzFiOTczYzZlOTg1MTY0MThlOWIyZDRkMTU3NDllNmRkMDU3MTJkM2M1NGE5NDljNGFhOTg2ZjlmZDJhNzNlOTgiLCJpYXQiOjE3NDY5Nzk3MDUsIm5iZiI6MTc0Njk3OTcwNSwiZXhwIjoxNzc4NTE1NzA1LCJzdWIiOiIzOTIwIiwic2NvcGVzIjpbXX0.MkERo0YMdc5dn0unXgoJ-aXLxoH-b754TNKSejBkqh9pa4os58SLiNMk3rOLmoTR0FJ0qdKRu7KB4WW4d5PD5SZ6ewRGOSXmeW8yNy_igdO1El0_caUL8Zz3yQTQh2IlUd03D-QZOQ9w6hYJkjZYA8qLrESCLfSBAV95UmOLhL_AIomL_hvBQt4od4G4T2sBFvibOplR-lGtl4MKOXnwSqG8pdgvbTihtF2-MSsWx7s0em3OfCVGTS9g22mvTA2rya2uuQ1a_bcZDmrodA_K893BAoArSxezQLkdGyHU440KWLCZnGH9Ro8AKAFV77zTbOwzadyyVVM6IdztqPHtjV7bLa-lGOzFTssJfpJCwhE9jO-cuTkoviTU3FOtInGWVsMmLfDqDE2wMqfzswLj_5cR2dthcI11SlgaYayU_jeMW2OqnRiXf4dAOiMGTsbvmKMSRX8sJ5u5QB-2pdikmZy9k2HyKeG-Zl36bztxjP1fmPru8v7VBwEDLzl_lOqZz7jiDrbdnNCfrELTdB0K5T9oX0rFFAt_JsADD7xcAbMnCO5Cx1Wl6DGSkxfoYYnqboc4O5Ngu5LZsdvUD0-B_YNo5SyMCOQ4ojnTMOi_kEJYbGPUpVF21PFeqk0zK9qQyp-1xvtEBEOtmcQaf0VH91iMW4bkMa1w7XC2sWTjcwM'
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
