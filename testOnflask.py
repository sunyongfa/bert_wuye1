import requests, json

contents = {'content': "好像有人要跳楼"}
headers = {'content-type': 'application/json'}
r = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(contents), headers=headers)
print(r.headers)
print(r.json())

