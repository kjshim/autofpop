from numpy import *
import subprocess
import struct
import logging
import re
from autodoraconfig import C


DEVRE = re.compile('/dev/input/event[\d]+')

def StartServer():
    logging.debug(subprocess.check_output(["adb", "start-server" ],stderr=subprocess.STDOUT))

def GetTempDirectory():
    result = subprocess.check_output(["adb", "shell", "echo", "$EXTERNAL_STORAGE" ],stderr=subprocess.STDOUT).strip()
    logging.debug("TMPDIR:%s"%result)
    return result


def DetectScreenDevice():
    result = subprocess.check_output(["adb", "shell", "getevent", "-p" ],stderr=subprocess.STDOUT)

    logging.debug(result)
    result = result.lower()
    v = [ l for l in result.split("add device") if l.find('0035') > 0 and l.find('0036') > 0 and (l.find('screen') > 0 or l.find('touch') >0 )]
    logging.debug("Possible screen device: " + str(v))
    try:
        devname = DEVRE.findall(v[0])[0]
        print devname
        return devname
    except Exception,e:
        logging.exception(e)



def Init():
    StartServer()
    if not C['SCREEN_DEVICE']:
        C['SCREEN_DEVICE'] = DetectScreenDevice()

    if not C['TMP_DIR']:
        C['TMP_DIR'] = GetTempDirectory()

    logging.debug("Init andlib:" + str(C))

def SwipeLine(poslist):
    events = []
    events.append((3,57,7973))
    for x, y in poslist:
        events.append((3,53,x))
        events.append((3,54,y))
        events.append((0,0,0))

    events.append((3,57,4294967295))
    events.append((0,0,0))
    f = open("my.sh", "w")
    for v in events:
        f.write("sendevent " + C['SCREEN_DEVICE'] + " %d %d %d\n"%v)
    f.close()

    logging.debug(subprocess.check_output(["adb", "push", "my.sh", "%s/my.sh" % C['TMP_DIR'] ],stderr=subprocess.STDOUT))
    logging.debug(subprocess.check_output(["adb", "shell", "sh", "%s/my.sh" %C['TMP_DIR']]))


def ReadRawImage(fname):
    arr = fromfile(fname, dtype=uint8)
    w, h, _ = struct.unpack('III', arr[:12])
    im = arr[12:].reshape((h,w,4))
    return im

def GetScreen(fname='img.raw'):
    logging.debug(subprocess.check_output(["adb", "shell", "/system/bin/screencap %s/%s"%(C['TMP_DIR'], fname) ]))
    logging.debug(subprocess.check_output(["adb", "pull", "%s/%s"%(C['TMP_DIR'], fname), fname ],stderr=subprocess.STDOUT))
    logging.debug(subprocess.check_output(["adb", "shell", "rm %s/%s" % (C['TMP_DIR'],fname) ]))
    im = ReadRawImage(fname)
    return im

def GetDeviceId():
    output = subprocess.check_output(["adb", "devices"]).strip().split("\r\n")[1:]
    if len(output) < 1:
        return None
    else:
        return output[0]

