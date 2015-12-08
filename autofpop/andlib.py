from numpy import *
import subprocess
import struct
import logging
import re

C = {}
DEVRE = re.compile('/dev/input/event[\d]+')

def StartServer():
    logging.debug(subprocess.check_output(["adb", "-d", "start-server" ],stderr=subprocess.STDOUT))

def GetTempDirectory():
    result = subprocess.check_output(["adb", "-d", "shell", "echo", "$EXTERNAL_STORAGE" ],stderr=subprocess.STDOUT).strip()
    logging.debug("TMPDIR:%s"%result)
    return result


def DetectScreenDevice():
    result = subprocess.check_output(["adb", "-d", "shell", "getevent", "-p" ],stderr=subprocess.STDOUT)

    logging.debug(result)
    result = result.lower()
    v = [ l for l in result.split("add device") if l.find('0035') > 0 and l.find('0036') > 0 and (l.find('screen') > 0 or l.find('touch') >0 )]
    logging.debug("Possible screen device: " + str(v))

def Init():
    StartServer()
    C['SCREEN_DEVICE'] = DetectScreenDevice()
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

    logging.debug(subprocess.check_output(["adb", "-d", "push", "my.sh", "%s/my.sh" % C['TMP_DIR'] ],stderr=subprocess.STDOUT))
    logging.debug(subprocess.check_output(["adb", "-d", "shell", "sh", "%s/my.sh" %C['TMP_DIR']]))


def ReadRawImage(fname):
    arr = fromfile(fname, dtype=uint8)
    w, h, _ = struct.unpack('III', arr[:12])
    im = arr[12:].reshape((h,w,4))
    return im

def GetScreen(fname='img.raw'):
    logging.debug(subprocess.check_output(["adb", "-d", "shell", "/system/bin/screencap %s/%s"%(C['TMP_DIR'], fname) ]))
    logging.debug(subprocess.check_output(["adb", "-d", "pull", "%s/%s"%(C['TMP_DIR'], fname), "tmp/" + fname ],stderr=subprocess.STDOUT))
    logging.debug(subprocess.check_output(["adb", "-d", "shell", "rm %s/%s" % (C['TMP_DIR'],fname) ]))
    im = ReadRawImage(fname)
    return im

def GetDeviceId():
    output = subprocess.check_output(["adb", "-d", "devices"]).strip().split("\r\n")[1:]
    if len(output) < 1:
        return None
    else:
        return output[0]

