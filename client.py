import requests
body = {
    "duration": 4.17,
    "days_left": 1,
    "price": 5
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}
