from distutils.core import setup
import py2exe
setup(
    options = {'py2exe':{'bundle_files':1, 'compressed':True}},
    zipfile = None,
    data_files=['adb.exe', 'AdbWinUsbApi.dll', 'AdbWinApi.dll', 'config.json']
    )
