import os
from os import listdir
from os.path import isfile, isdir, join
import sys

# # 指定要列出所有檔案的目錄
# mypath = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/plate"
tessdata_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/open-source/tesseract-ocr/tessdata'
plus = '--psm 7'
# # 取得所有檔案與子目錄名稱
# files = listdir(mypath)
#
# for f in files:
#     fullpath = join(mypath, f)
#     os.system('export TESSDATA_PREFIX=%s \n' % tessdata_path + 'tesseract %s %s %s' % (plus, fullpath, join(mypath, f.split('.')[0])))

# -------------- single file ------------------
input = 'ssss2.bmp'
output = 'ssss2'
os.system('export TESSDATA_PREFIX=%s \n' % tessdata_path +\
          'tesseract %s /media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/plate/%s /media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/plate/%s' % (plus, input, output))
