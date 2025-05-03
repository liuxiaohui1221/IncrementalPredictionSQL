import argparse

import numpy as np
from bitmap import BitMap
from typing import Any
import CreateSQLFromIntentVec
import ParseConfigFile as parseConfig
import ReverseEnggQueries
def is_2d_ndarray(obj: Any) -> bool:
    """判断是否为二维 NumPy 数组"""
    return isinstance(obj, np.ndarray) and obj.ndim == 2
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
    # args.intent = intentFile
    # args.eval = evalFile
    # args.conc = concFile
    args.output = None
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    truncActual = "10000100000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011111111111111111100000000000001000000000000000000010000000000000100111000000000000000000000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001001110000000000000000000000100000000000000000000000000000000000000000100000000000000000000000000000000000000001000000000000000000000000000000000000000010000000000000111111000000010000"
    actualIntent = BitMap.fromstring(truncActual)
    createSqlLib = CreateSQLFromIntentVec
    actualIntentObjs = createSqlLib.regenerateSQL(None, actualIntent, schemaDicts)
    if is_2d_ndarray(actualIntentObjs):
        print("ActualIntentObjs:", actualIntentObjs)
        for actualIntentObjRow in actualIntentObjs:
            for actualIntentObj in actualIntentObjRow:
                outputSQLStr = "Actual SQL Ops:\n" + createSqlLib.createSQLString(actualIntentObj)
                print(outputSQLStr)