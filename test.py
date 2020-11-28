import requests

url = 'http://109.184.147.246:8080/api/getsemsear/wow'
response = requests.get(url)

print(response.json())