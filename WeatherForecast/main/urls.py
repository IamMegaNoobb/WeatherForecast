from django.urls import path
from . import main

urlpatterns = [
    path("", main.index, name="index")
]
