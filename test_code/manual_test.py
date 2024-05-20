import sys
sys.path.append("/mnt/disk1/chris/uaicraft_workspace/translate-everywhere")

import time

from core.end2end import End2End, AzureEnd2End

model = AzureEnd2End()
test_file = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/jupyter_test/test.wav'
start = time.perf_counter()
final_text = model.end2end_flow('zh', 'ja', test_file)
end = time.perf_counter()
print(f'TOTAL Time taken: {end - start}')
