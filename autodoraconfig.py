import json
import os
import logging

CONFIG_FILENAME = 'config.json'

C = {
    "SCREEN_DEVICE":'',
    "TMP_DIR":'',
}

if os.path.exists(CONFIG_FILENAME):
    v = json.load(open(CONFIG_FILENAME,'r'))
    logging.debug("Config read: " + str(v))
    C.update(v)
