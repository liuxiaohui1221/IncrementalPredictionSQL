import argparse
import os
import glob
from collections import defaultdict

from bitmap import BitMap
from holoviews.operation import threshold

import CreateSQLFromIntentVec
import ReverseEnggQueries
import apm.evaluate_window as eval
import numpy as np
import tensorflow as tf
import ParseConfigFile as parseConfig
from ReverseEnggQueries import employWeightThreshold
from apm.CreateSQLLogs import is_2d_ndarray
from apm.LSTM_RNN_Incremental_window_selOpCons import createCharListFromIntent


class LatestModelPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.current_model_path = None
        self.current_model_mtime = None

    def _get_latest_model_info(self):
        """获取最新模型信息和修改时间"""
        model_files = glob.glob(os.path.join(self.model_dir, "*.h5"))
        if not model_files:
            return None, None

        # 获取带时间戳的文件列表
        files_with_mtime = [(f, os.path.getmtime(f)) for f in model_files]
        # 按修改时间排序，获取最新文件
        latest_file, latest_mtime = max(files_with_mtime, key=lambda x: x[1])
        return latest_file, latest_mtime

    def _load_model(self, model_path):
        """加载指定模型"""
        try:
            # 如果需要自定义层，在此处添加custom_objects参数
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def update_model(self):
        """检查并更新到最新模型"""
        latest_path, latest_mtime = self._get_latest_model_info()

        if not latest_path:
            raise ValueError("指定目录中没有找到H5模型文件")

        # 检查是否需要更新模型
        if not self.model or (latest_mtime > self.current_model_mtime):
            print(f"发现新模型: {latest_path}")
            new_model = self._load_model(latest_path)

            # 原子操作更新模型引用
            self.model = new_model
            self.current_model_path = latest_path
            self.current_model_mtime = latest_mtime
            print("模型更新成功")

    def predict(self, input_data):
        """使用最新模型进行预测"""
        # 预测前检查模型更新
        self.update_model()

        if not self.model:
            raise RuntimeError("模型尚未加载")

        # 执行预测
        return self.model.predict(input_data)
import os

def get_prefix_files(directory: str, prefix: str) -> list[str]:
    """获取目录中文件名以指定前缀开头的文件路径列表"""
    files = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):  # 排除同名目录
                files.append(filepath)
    return files
def read_lines(filepath: str) -> list[str]:
    """按行读取文件内容（含末尾换行符）"""
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)
    except Exception as e:
        print(f"读取文件 {filepath} 出错: {e}")
    return lines
def process_prefix_files(directory: str, prefix: str) -> dict[str, list[str]]:
    """处理目录下所有前缀匹配文件，返回 {文件路径: 行内容列表}"""
    sessionStreamDict = {}
    sessionStreamSqlDict = {}
    for filepath in get_prefix_files(directory, prefix):
        lines = read_lines(filepath)
        for line in lines:
            (sessID, queryID, curQueryIntent, sql) = retrieveSessIDQueryIDIntent(line, configDict)
            sessionStreamDict[str(sessID) + "," + str(queryID)] = curQueryIntent
            sessionStreamSqlDict[str(sessID) + "," + str(queryID)] = sql
    return sessionStreamDict,sessionStreamSqlDict
def retrieveSessIDQueryIDIntent(line, configDict):
    tokens = line.strip().split(";")
    sessQueryName = tokens[0]
    sessID = int(sessQueryName.split(", ")[0].split(" ")[1])
    queryID = int(sessQueryName.split(", ")[1].split(" ")[1]) - 1  # coz queryID starts from 1 instead of 0
    strQueryIntent=tokens[2:]
    # print("before queryintent list length:", len(strQueryIntent[0]))
    curQueryIntent = ';'.join(strQueryIntent) # actual query intent
    # print("before queryintent length:", len(curQueryIntent))
    if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        curStrip = curQueryIntent.strip()
        curQueryIntent = BitMap.fromstring(curStrip)
        # print("curqueryintent length: ", curQueryIntent.size())
    else:
        curQueryIntent = eval.normalizeWeightedVector(curQueryIntent)
    return (sessID, queryID, curQueryIntent, tokens[1])

def inverseVectorToSQL(queryIntent, schemaDicts,isIncludeSelOpConst=False,isPredictQueryOrTable='QUERY'):
    if isinstance(queryIntent, str):
        queryIntent = BitMap.fromstring(queryIntent)
    if isIncludeSelOpConst == False:
        createSqlLib = CreateSQLFromIntentVec
    topTableQuerys = defaultdict(list)
    if isPredictQueryOrTable == 'QUERY':
        actualIntentObjs = createSqlLib.regenerateSQL(None, queryIntent, schemaDicts)
        if is_2d_ndarray(actualIntentObjs):
            for actualIntentObjRow in actualIntentObjs:
                for actualIntentObj in actualIntentObjRow:
                    topTableQuerys[",".join(actualIntentObj.tables)].append(createSqlLib.createSQLString(
                        actualIntentObj))
    return topTableQuerys
# 使用示例
def predictAndInverseVectorToSQL(queryIntent, predictor, configDict, threshold):
    if isinstance(queryIntent, str):
        queryIntent = BitMap.fromstring(queryIntent)
    testX = []
    intentStrList = createCharListFromIntent(queryIntent, configDict)
    testX.append(intentStrList)
    # print "Appended charList sessID: "+str(sessID)+", queryID: "+str(queryID)
    # modify testX to be compatible with the RNN prediction
    testX = np.array(testX)
    testX = testX.reshape(1, testX.shape[0], testX.shape[1])
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)  # -- required for predicting completely new queries
    # 进行预测
    try:
        predictedY = predictor.predict(testX)
        predictedY = predictedY[0][predictedY.shape[1] - 1]

        topKCandidateVector = employWeightThreshold(predictedY, schemaDicts, threshold)
        outputSQLDict = inverseVectorToSQL(topKCandidateVector, schemaDicts)
    except Exception as e:
        print(f"预测失败: {str(e)}")
    return outputSQLDict


if __name__ == "__main__":
    configFile = "/home/xhh/db_workspace/IncrementalPredictionSQL/config/window/APM_Window_Novel_RNN_singularity_configFile.txt"
    intentFile = "/home/xhh/毕业论文/实验结果/RNN-simple/Result/OutputFileShortTermIntent_RNN_LSTM_FRAGMENT_BIT_TOP_K_3train_214256"
    evalFile = "/home/xhh/毕业论文/实验结果/RNN-simple/Result/OutputEvalQualityShortTermIntent_RNN_LSTM_FRAGMENT_BIT_TOP_K_3_EPISODE_IN_QUERIES_256_ACCURACY_THRESHOLD_0.95"
    concFile = "/home/xhh/db_workspace/IncrementalPredictionSQL/input/window/5minute_1tab/ApmConcurrentSessionstrain_214"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.config = configFile
    # args.intent = intentFile
    # args.eval = evalFile
    # args.conc = concFile
    args.output = None
    configDict = parseConfig.parseConfigFile(args.config)
    # 初始化预测器（指定模型目录）
    predictor = LatestModelPredictor("/home/xhh/db_workspace/IncrementalPredictionSQL/apm/models/")
    threshold = float(configDict["ACCURACY_THRESHOL"])
    # todo示例输入数据（需要根据实际模型调整形状和数据类型）
    directory = "/home/xhh/db_workspace/IncrementalPredictionSQL/input/window/"
    prefix = "test_batchsize_1000_"
    sessionStreamDict,sessionStreamSqlDict = process_prefix_files(directory, prefix)
    # Session 0, Query 0; OrigQuery:SELECT sum(err) AS err_RESP, sum(fail) AS fail_RESP, sum(frustrated) AS frustrated_RESP, sum(tolerated) AS slow_RESP, count() AS total_RESP FROM dwm_request_cluster WHERE (appsysid = '9ba9403b-b000-4a4e-9d85-8e831bbf9d06') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600.000, 3))->SELECT count() AS total_RESP FROM pmone_0d5de51f17.dwm_request WHERE (appsysid = '89c9d48c-482c-4f44-8120-48c2e414dd7c') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600., 3)) AND (status IN ('1', '10', '17', '18', '2', '20', '21', '22', '24', '25', '26', '33', '34', '36', '37', '38', '4', '40', '41', '42', '49', '5', '50', '52', '53', '54', '56', '57', '58', '6', '64', '65', '66', '68', '69', '70', '72', '73', '74', '8', '80', '81', '82', '84', '85', '86', '88', '89', '9', '90'))->SELECT count() AS total_RESP FROM dwm_request_cluster WHERE (appsysid = '6fea236d-3bc6-4acc-8f87-af1f531da761') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600.000, 3)) AND ((status = '1') OR (status = '10') OR (status = '17') OR (status = '18') OR (status = '2') OR (status = '20') OR (status = '21') OR (status = '22') OR (status = '24') OR (status = '25') OR (status = '26') OR (status = '33') OR (status = '34') OR (status = '36') OR (status = '37') OR (status = '38') OR (status = '4') OR (status = '40') OR (status = '41') OR (status = '42') OR (status = '49') OR (status = '5') OR (status = '50') OR (status = '52') OR (status = '53') OR (status = '54') OR (status = '56') OR (status = '57') OR (status = '58') OR (status = '6') OR (status = '64') OR (status = '65') OR (status = '66') OR (status = '68') OR (status = '69') OR (status = '70') OR (status = '72') OR (status = '73') OR (status = '74') OR (status = '8') OR (status = '80') OR (status = '81') OR (status = '82') OR (status = '84') OR (status = '85') OR (status = '86') OR (status = '88') OR (status = '89') OR (status = '9') OR (status = '90'));
    # curSessIntent = "10000100000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011110000000000000000001000000000000000000010000000000000100011000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000110000000000000000000000100000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111100000000000000"
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    for sessAndQueryID, queryIntent in sessionStreamDict.items():
        sessId = sessAndQueryID.split(',')[0]
        queryId = sessAndQueryID.split(',')[1]
        print("Current window SQL：\n",sessionStreamSqlDict[sessAndQueryID])
        print("Current window SQL ops:\n",inverseVectorToSQL(queryIntent, schemaDicts))
        outputPredictNextSQLStr = predictAndInverseVectorToSQL(queryIntent, predictor, configDict, threshold)
        # nextIntent = predictor.predict(queryIntent)
        nextKey = sessId + ',' + str(int(queryId) + 1)
        if nextKey in sessionStreamSqlDict:
            nextActualQueryIntent = sessionStreamDict[nextKey]
            outputActualNextSQLFragmentStr = inverseVectorToSQL(nextActualQueryIntent, schemaDicts)

            outputActualNextSQLStr = sessionStreamSqlDict[nextKey]
            #根据->分割sql
            outputActualNextSQLs = outputActualNextSQLStr.split("->")
        print("Predict next window SQL：",outputPredictNextSQLStr)
        print("Actual next window SQL:\n")
        print(outputActualNextSQLFragmentStr)
        for sql in outputActualNextSQLs:
            print(sql)
        print("===========================================================")
