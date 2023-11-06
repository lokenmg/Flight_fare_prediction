import requests
body = {
    "pH":3.5,
    "Temprature":70,
    "Taste":0,
    "Odor":1,
    "Fat" :0,
    "Turbidity":1,
    "Colour":246
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}
