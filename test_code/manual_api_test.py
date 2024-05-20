import requests

url = "http://localhost:55699/v1/end2end/speech-translate"
file_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/uploaded_audio/20240520181309.wav"
save_path = "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/test_return.wav"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

if response.status_code == 200:
    # 将返回的文件内容保存到本地
    with open(save_path, "wb") as save_file:
        save_file.write(response.content)
    print(f"File saved successfully to {save_path}")
else:
    print(f"Failed to upload file: {response.status_code}, {response.text}")