import os
from os.path import expanduser
import socket
def parseSchema(schemaFileName):
    schemaDict = {}
    colIndex = 0  #colIndex starts from 0
    with open(schemaFileName) as f:
        for line in f:
            colName = line.split("=")[0]
            schemaDict[colName]=colIndex
            colIndex=colIndex+1 # no need to keep data type recorded in the schema Dict. It is column as key and colIndex as value
    return schemaDict

def getConfig(relativePath):
    homeDir = expanduser("~")
    if socket.gethostname() == "en4119510l" or socket.gethostname() == "en4119509l" or socket.gethostname() == "en4119508l" or socket.gethostname() == "en4119507l":
        homeDir = "/hdd2/vamsiCodeData"
    # absPath = homeDir+"/"+relativePath
    baseDir = "/data/lxhdata/IncrementalPredictionSQL"
    if "output/" in relativePath:
        absPath = os.path.join(baseDir, relativePath)
    else:
        absPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), relativePath)
    return absPath

def parseConfigFile(fileName):
    configDict = {}
    with open(fileName) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            print("Config parameter: "+line.strip())

            (key, val) = line.strip().split("=")
            configDict[key] = val.strip()
    return configDict