from __future__ import division
import sys, operator
import os
from typing import Any

import numpy as np

import QueryRecommender as QR
from bitmap import BitMap
import TupleIntent as ti
import ParseConfigFile as parseConfig
import argparse
from ParseConfigFile import getConfig
import ReverseEnggQueries
import CreateSQLFromIntentVec
import ReverseEnggQueries_selOpConst
import CreateSQLFromIntentVec_selOpConst

def readFromConcurrentFile(concSessFile):
    # Note that query IDs start in the file from 1 but in the outputIntent, query ID starts from 0: so Decrement by 1
    curQueryDict = {}
    try:
        with open(concSessFile) as f:
            for line in f:
                tokens = line.strip().split(";")
                sessQueryID = tokens[0]
                sessID = int(sessQueryID.split(", ")[0].split(" ")[1])
                queryID = int(sessQueryID.split(", ")[1].split(" ")[1]) - 1
                curQuery = tokens[1]
                sessQueryID = "Session:"+str(sessID)+";"+"Query:"+str(queryID)
                assert sessQueryID not in curQueryDict
                curQueryDict[sessQueryID] = curQuery
    except:
        print("cannot read line !!")
        sys.exit(0)
    return curQueryDict

def readFromOutputEvalFile(outputEvalQualityFileName):
    outputEvalDict = {}
    with open(outputEvalQualityFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            outputEvalDict[tokens[0]+";"+tokens[1]] = ";".join(tokens[2:])
    return outputEvalDict
def is_2d_ndarray(obj: Any) -> bool:
    """判断是否为二维 NumPy 数组"""
    return isinstance(obj, np.ndarray) and obj.ndim == 2
def procPredictedIntents(configDict, schemaDicts, curQueryDict, outputEvalDict, outputIntentFileName, outputSQLLog):
    QR.deleteIfExists(outputSQLLog)
    assert configDict['INCLUDE_SEL_OP_CONST'] == 'True' or configDict['INCLUDE_SEL_OP_CONST'] == 'False'
    if configDict['INCLUDE_SEL_OP_CONST'] == 'False':
        createSqlLib = CreateSQLFromIntentVec
    else:
        createSqlLib = CreateSQLFromIntentVec_selOpConst
    assert configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY' or configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE'
    with open(outputIntentFileName) as f:
        for line in f:
            tokens = line.strip().split(";")
            #assert len(tokens) == 4 + int(configDict['TOP_K'])
            sessQueryID = tokens[0]+";"+tokens[1]
            outputSQLStr = []
            outputSQLStr.append(outputEvalDict[sessQueryID]+";"+sessQueryID) # prints the metrics first
            outputSQLStr.append("Current Query: "+curQueryDict[sessQueryID])
            nextQueryID = "Query:"+str(int(tokens[1].split(":")[1]) + 1)
            outputSQLStr.append("Next Query: "+curQueryDict[tokens[0]+";"+nextQueryID])
            actualIntent = BitMap.fromstring(tokens[3].split(":")[1])
            if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
                actualIntentObjs = createSqlLib.regenerateSQL(None, actualIntent, schemaDicts)
                if is_2d_ndarray(actualIntentObjs):
                    for actualIntentObjRow in actualIntentObjs:
                        for actualIntentObj in actualIntentObjRow:
                            outputSQLStr.append("Actual SQL Ops:\n" + createSqlLib.createSQLString(actualIntentObj))
                # outputSQLStr += "Actual SQL Ops:\n" + createSqlLib.createSQLString(actualIntentObj)
            elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
                actualIntentObj = createSqlLib.regenerateSQLTable(None, actualIntent, None, schemaDicts, configDict)
                outputSQLStr.append("Actual SQL Ops:\n" + createSqlLib.createSQLStringForTable(actualIntentObj))
            for i in range(4, len(tokens)):
                predictedIntent = BitMap.fromstring(tokens[i].split(":")[1])
                relIndex = i - 4
                if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'QUERY':
                    predictedIntentObjs = createSqlLib.regenerateSQL(None, predictedIntent, schemaDicts)
                    if is_2d_ndarray(predictedIntentObjs):
                        for predictedIntentObjRow in predictedIntentObjs:
                            for predictedIntentObj in predictedIntentObjRow:
                                outputSQLStr.append("Predicted SQL Ops " + str(
                                    relIndex) + ":\n" + createSqlLib.createSQLString(predictedIntentObj))
                elif configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
                    predictedIntentObj = createSqlLib.regenerateSQLTable(None, predictedIntent, None, schemaDicts, configDict)
                    outputSQLStr.append("Predicted SQL Ops " + str(
                        relIndex) + ":\n" + createSqlLib.createSQLStringForTable(predictedIntentObj))
            ti.appendToFile(outputSQLLog, outputSQLStr)
    print(outputSQLStr)
    return

def createSQLLogsFromConfigDict(configDict, args):
    # accThres = float(configDict['ACCURACY_THRESHOLD'])
    accThres = 0.95
    if args.intent is not None:
        outputIntentFileName = args.intent
    elif configDict['ALGORITHM'] == 'RNN':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + \
                               configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"] + "_" + \
                               configDict['INTENT_REP'] + "_" + \
                               configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                               configDict['EPISODE_IN_QUERIES']
    elif configDict['ALGORITHM'] == 'CF':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + \
                           configDict['CF_COSINESIM_MF'] + "_" + \
                           configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                               'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    elif configDict['ALGORITHM'] == 'SVD':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
        configDict[
            'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    elif configDict['ALGORITHM'] == 'QLEARNING':
        outputIntentFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict['INTENT_REP'] + "_" + \
        configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
            'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict['EPISODE_IN_QUERIES']
    if args.eval is not None:
        outputEvalQualityFileName = args.eval
    elif configDict['ALGORITHM'] == 'RNN':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                        'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                    configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'CF':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + configDict[
        'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                    'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'SVD':
        outputEvalQualityFileName = getConfig(configDict['OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + \
                                    configDict[
                                        'ALGORITHM'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                        'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                    configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    elif configDict['ALGORITHM'] == 'QLEARNING':
        outputEvalQualityFileName = os.path.join(getConfig(configDict['OUTPUT_DIR']),
                                                 "OutputEvalQualityShortTermIntent_" + configDict[
                                        'ALGORITHM'] + "_" + configDict['QL_BOOLEAN_NUMERIC_REWARD'] + "_" + configDict[
                                        'INTENT_REP'] + "_" + configDict[
                                        'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                    configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres))
    if args.conc is not None:
        concSessFile = args.conc
    else:
        concSessFile = getConfig(configDict['CONCURRENT_QUERY_SESSIONS'])
    if args.output is not None:
        outputSQLLog = args.output
    else:
        outputSQLLog = getConfig(configDict['OUTPUT_DIR']) + "/outputSQLLog"
    curQueryDict = readFromConcurrentFile(concSessFile)
    outputEvalDict = readFromOutputEvalFile(outputEvalQualityFileName)
    assert configDict['INCLUDE_SEL_OP_CONST'] == 'True' or configDict['INCLUDE_SEL_OP_CONST'] == 'False'
    if configDict['INCLUDE_SEL_OP_CONST'] == 'False':
        schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    else:
        schemaDicts = ReverseEnggQueries_selOpConst.readSchemaDicts(configDict)
    procPredictedIntents(configDict, schemaDicts, curQueryDict, outputEvalDict, outputIntentFileName, outputSQLLog)
    print("Saved SQL logs to " + outputSQLLog)
    return

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    configFile="/home/xhh/db_workspace/IncrementalPredictionSQL/config/window/APM_Window_Novel_RNN_singularity_configFile.txt"
    intentFile = "/home/xhh/毕业论文/实验结果/RNN-simple/Result/OutputFileShortTermIntent_RNN_LSTM_FRAGMENT_BIT_TOP_K_3train_214256"
    evalFile = "/home/xhh/毕业论文/实验结果/RNN-simple/Result/OutputEvalQualityShortTermIntent_RNN_LSTM_FRAGMENT_BIT_TOP_K_3_EPISODE_IN_QUERIES_256_ACCURACY_THRESHOLD_0.95"
    concFile = "/home/xhh/db_workspace/IncrementalPredictionSQL/input/window/5minute_1tab/ApmConcurrentSessionstrain_214"
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    # parser.add_argument("-intent", help="intent output file", type=str, required=False)
    # parser.add_argument("-eval", help="eval quality file", type=str, required=False)
    # parser.add_argument("-conc", help="concurrent session file", type=str, required=False)
    # parser.add_argument("-output", help="output sql log file", type=str, required=False)
    args = parser.parse_args()
    args.config = configFile
    args.intent = intentFile
    args.eval = evalFile
    args.conc = concFile
    args.output = None
    configDict = parseConfig.parseConfigFile(args.config)
    createSQLLogsFromConfigDict(configDict, args)