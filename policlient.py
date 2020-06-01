"""
  Simple client for testing the serving application.

"""

import json
import requests

input = {"signature_name": "serving_default",
         "instances": [ "The text", "dummy" ] }
data = json.dumps(input)

print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/politeness:predict', data=data, headers=headers)
print('json_response: ', json_response.text)
if json_response.ok:
    predictions = json.loads(json_response.text)['predictions']
    print('predictions:', predictions)
else:
    print('response status: ', json_response.status_code)
