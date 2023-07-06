import requests

file = open('../25.m4a', 'rb')

files = {
    'music': file
}

data = {
    "user_id": "ekffuf",
    "declaration": "01012345678"
}

res = requests.post('http://127.0.0.1:5000/file/', files=files, data=data)
