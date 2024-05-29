import requests
import base64

url = "http://localhost:55699/v1/end2end/text-translate-custom"
save_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/test_return.wav"

data = {"source_text": "我終於做好了", "source_lang": "zh", "target_lang": "fr", "name": "evonne"}
response = requests.post(url, data=data)

if response.status_code == 200:
    # 将返回的文件内容保存到本地
    data = response.json()
    print(data['source_text'])
    print(data['target_text'])
    if data['file'] is not None:
        print(len(data['file']))
        file_bytes = base64.b64decode(data['file'])
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
    else:
        print("No file returned")

else:
    print(f"Failed to upload file: {response.status_code}, {response.text}")