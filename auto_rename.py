import os
from os import listdir
from os.path import isfile, isdir, join
import sys

# 指定要列出所有檔案的目錄
mypath = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/plate"

# 取得所有檔案與子目錄名稱
files = listdir(mypath)
count = 0
# 以迴圈處理
for f in files:
  if ".jpg" in f:
      count += 1
      # 產生檔案的絕對路徑
      fullpath = join(mypath, f)
      # 判斷 fullpath 是檔案還是目錄
      if isfile(fullpath):
        print("檔案：", f)
        os.rename(fullpath, join(mypath, str(count) + '.jpg'))
      elif isdir(fullpath):
        print("目錄：", f)
