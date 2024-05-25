import requests
import base64

url = "http://localhost:55699/v1/end2end/speech-translate-custom"
file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240521/20240521132110.wav"
file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240522/20240522112311.wav"
file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240522/20240522170117.wav"
# file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240522/20240522112516.wav"
file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240522/20240522163804.wav"
save_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/test_return.wav"

with open(file_path, "rb") as file:
    data = {"source_lang": "zh", "target_lang": "pl", "name": "evonne"}
    files = {"file": file}
    response = requests.post(url, files=files, data=data)

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