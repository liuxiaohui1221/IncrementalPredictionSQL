import argparse
import os
import glob
from collections import defaultdict

from bitmap import BitMap
from holoviews.operation import threshold
from pandas import DataFrame

import CreateSQLFromIntentVec
import ReverseEnggQueries
import apm.evaluate_window as eval
import numpy as np
import tensorflow as tf
import ParseConfigFile as parseConfig
from ReverseEnggQueries import employWeightThreshold
from apm.CreateSQLLogs import is_2d_ndarray
from apm.LSTM_RNN_Incremental_window_selOpCons import createCharListFromIntent
import os

from apm.LatestModelPredictor import LatestModelPredictor, predictAndInverseVectorToSQL


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
def computeBitFMeasure(actualQueryIntent, topKQueryIntent, schemaDicts):
    # 断言actualQueryIntent和topKQueryIntent的大小相等
    assert actualQueryIntent.size() == topKQueryIntent.size()
    TP=0
    FP=0
    TN=0
    FN=0
    hit=-1
    # 遍历actualQueryIntent和topKQueryIntent的每一个元素

    for pos in range(actualQueryIntent.size()):
        # 统计是否命中
        if actualQueryIntent.test(pos):
            if not topKQueryIntent.test(pos):
                hit=0  # 未命中
        # 如果actualQueryIntent和topKQueryIntent的元素都为真，则TP+1
        if actualQueryIntent.test(pos) and topKQueryIntent.test(pos):
            TP+=1
        # 如果actualQueryIntent和topKQueryIntent的元素都为假，则TN+1
        elif not actualQueryIntent.test(pos) and not topKQueryIntent.test(pos):
            TN+=1
        # 如果actualQueryIntent的元素为真，topKQueryIntent的元素为假，则FN+1
        elif actualQueryIntent.test(pos) and not topKQueryIntent.test(pos):
            FN+=1
        # 如果actualQueryIntent的元素为假，topKQueryIntent的元素为真，则FP+1
        elif not actualQueryIntent.test(pos) and topKQueryIntent.test(pos):
            FP+=1
    if hit==-1:
        hit=1 # 命中
    # 如果TP和FP都为0，则precision为0.0
    if TP == 0 and FP == 0:
        precision = 0.0
    else:
        # 否则，precision为TP/(TP+FP)
        precision = float(TP)/float(TP+FP)
    # 如果TP和FN都为0，则recall为0.0
    if TP == 0 and FN == 0:
        recall = 0.0
    else:
        # 否则，recall为TP/(TP+FN)
        recall = float(TP)/float(TP+FN)
    # 如果precision和recall都为0，则FMeasure为0.0
    if precision == 0.0 and recall == 0.0:
        FMeasure = 0.0
    else:
        # 否则，FMeasure为2*precision*recall/(precision+recall)
        FMeasure = 2 * precision * recall / (precision + recall)
    # accuracy为(TP+TN)/(TP+FP+TN+FN)
    accuracy = float(TP+TN)/float(TP+FP+TN+FN)
    # 返回precision、recall、FMeasure和accuracy
    return (precision, recall, FMeasure, accuracy,hit)

def saveEvalMetrics(precision, recall, FMeasure, outputExcel):
    df = DataFrame(
        {'precision': precision,
         'recall': recall, 'FMeasure': FMeasure, 'hitMeasure': accuracy})
    df.to_excel(outputExcel, sheet_name='sheet1', index=False)

from collections import Counter
def is_subset(list1, subList):
    # 判断 list1 是否完全包含 subList 的所有元素（包括重复次数）
    return Counter(subList) <= Counter(list1)
if __name__ == "__main__":
    configFile = ("/home/xhh/db_workspace/IncrementalPredictionSQL/config/window"
                  "/APM_Window_Novel_RNN_singularity_configFile.txt")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.config = configFile
    args.output = None
    configDict = parseConfig.parseConfigFile(args.config)
    # 初始化预测器（指定模型目录）
    predictor = LatestModelPredictor("/home/xhh/db_workspace/IncrementalPredictionSQL/apm/models/")
    # threshold = float(configDict["ACCURACY_THRESHOLD"])
    threshold = 0.5
    # todo示例输入数据（需要根据实际模型调整形状和数据类型）
    directory = "/home/xhh/db_workspace/IncrementalPredictionSQL/input/window/"
    prefix = "test_batchsize_1000_"
    sessionStreamDict,sessionStreamSqlDict = process_prefix_files(directory, prefix)
    # Session 0, Query 0; OrigQuery:SELECT sum(err) AS err_RESP, sum(fail) AS fail_RESP, sum(frustrated) AS frustrated_RESP, sum(tolerated) AS slow_RESP, count() AS total_RESP FROM dwm_request_cluster WHERE (appsysid = '9ba9403b-b000-4a4e-9d85-8e831bbf9d06') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600.000, 3))->SELECT count() AS total_RESP FROM pmone_0d5de51f17.dwm_request WHERE (appsysid = '89c9d48c-482c-4f44-8120-48c2e414dd7c') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600., 3)) AND (status IN ('1', '10', '17', '18', '2', '20', '21', '22', '24', '25', '26', '33', '34', '36', '37', '38', '4', '40', '41', '42', '49', '5', '50', '52', '53', '54', '56', '57', '58', '6', '64', '65', '66', '68', '69', '70', '72', '73', '74', '8', '80', '81', '82', '84', '85', '86', '88', '89', '9', '90'))->SELECT count() AS total_RESP FROM dwm_request_cluster WHERE (appsysid = '6fea236d-3bc6-4acc-8f87-af1f531da761') AND (ts <= toDateTime64(1684489499.999, 3)) AND (ts >= toDateTime64(1684425600.000, 3)) AND ((status = '1') OR (status = '10') OR (status = '17') OR (status = '18') OR (status = '2') OR (status = '20') OR (status = '21') OR (status = '22') OR (status = '24') OR (status = '25') OR (status = '26') OR (status = '33') OR (status = '34') OR (status = '36') OR (status = '37') OR (status = '38') OR (status = '4') OR (status = '40') OR (status = '41') OR (status = '42') OR (status = '49') OR (status = '5') OR (status = '50') OR (status = '52') OR (status = '53') OR (status = '54') OR (status = '56') OR (status = '57') OR (status = '58') OR (status = '6') OR (status = '64') OR (status = '65') OR (status = '66') OR (status = '68') OR (status = '69') OR (status = '70') OR (status = '72') OR (status = '73') OR (status = '74') OR (status = '8') OR (status = '80') OR (status = '81') OR (status = '82') OR (status = '84') OR (status = '85') OR (status = '86') OR (status = '88') OR (status = '89') OR (status = '9') OR (status = '90'));
    # curSessIntent = "10000100000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011110000000000000000001000000000000000000010000000000000100011000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000110000000000000000000000100000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111100000000000000"
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    total = 0
    hitCount = 0
    precisions=[]
    recalls=[]
    FMeasures=[]
    hitMeasures={}
    outputEvalExcel = "/home/xhh/db_workspace/IncrementalPredictionSQL/output/window/APM_Window_Novel_RNN_singularity_eval.xlsx"
    for sessAndQueryID, queryIntent in sessionStreamDict.items():
        sessId = sessAndQueryID.split(',')[0]
        queryId = sessAndQueryID.split(',')[1]
        curentSQLDict = inverseVectorToSQL(queryIntent, schemaDicts)
        curQueryTime = curentSQLDict[schemaDicts.tableOrderDict[0]][0].get("queryTime")
        outputSQLDict,predictNextQueryWindowIntent = predictAndInverseVectorToSQL(queryIntent, predictor, configDict, threshold)
        # nextIntent = predictor.predict(queryIntent)
        nextKey = sessId + ',' + str(int(queryId) + 1)
        if nextKey in sessionStreamSqlDict:
            nextActualQueryWindowIntent = sessionStreamDict[nextKey]
            outputActualSQLDict= inverseVectorToSQL(nextActualQueryWindowIntent, schemaDicts)
            (precision, recall, FMeasure, accuracy, hit) = computeBitFMeasure(nextActualQueryWindowIntent,
                                                                              predictNextQueryWindowIntent,schemaDicts)
            precisions.append(precision)
            recalls.append(recall)
            FMeasures.append(FMeasure)
            # 统计各个部分命中率：查询时间命中率，分组维度命中率，聚合粒度命中率，查询时间范围命中率
            for actualSQLs in outputActualSQLDict.values():
                for actualSQL in actualSQLs:
                    total += 1
                    groupByHit = 0
                    queryTimeHit = 0
                    queryGranularityHit = 0
                    timeRangeHit = 0
                    projColsHit = 0
                    selColsHit = 0
                    for predictSQLs in outputSQLDict.values():
                        for predictSQL in predictSQLs:
                            if actualSQL.get("selCols") is not None:
                                selCols = actualSQL["selCols"]
                            else:
                                selCols = []
                            if actualSQL.get("projCols") is not None:
                                projCols = actualSQL["projCols"]
                            else:
                                projCols = []
                            if actualSQL.get("groupByCols") is not None:
                                groupByCols = actualSQL["groupByCols"]
                            else:
                                groupByCols = []
                            if actualSQL.get("queryTime") is not None:
                                queryTime = actualSQL["queryTime"]
                            else:
                                queryTime = ""
                            if actualSQL.get("queryGranularity") is not None:
                                queryGran = int(actualSQL["queryGranularity"][:-1])
                            else:
                                queryGran = 0
                            if actualSQL.get("timeRange") is not None:
                                timeRange = int(actualSQL["timeRange"][:-1])
                            else:
                                timeRange = 0
                            if predictSQL.get("queryGranularity") is not None:
                                predictGran = int(predictSQL["queryGranularity"][:-1])
                            else:
                                predictGran = 0
                            if predictSQL.get("timeRange") is not None:
                                predictTimeRange = int(predictSQL["timeRange"][:-1])
                            else:
                                predictTimeRange = 0

                            #各个片段预测下个窗口与实际下个窗口片段兼容性检查是否命中
                            if is_subset(selCols, predictSQL.get("selCols")):
                                selColsHit = 1
                            if is_subset(predictSQL.get("projCols"),projCols):
                                projColsHit = 1
                            if predictGran <= queryGran:
                                queryGranularityHit = 1
                            if is_subset(predictSQL.get("groupByCols"),groupByCols):
                                groupByHit = 1
                            # if predictSQL.get("queryTime","-1") == queryTime:
                            #     queryTimeHit = 1
                            if curQueryTime == queryTime:
                                queryTimeHit = 1
                            if predictTimeRange >= timeRange:
                                timeRangeHit = 1

                    hitMeasures["groupByCols"] = hitMeasures.get("groupByCols", 0) + groupByHit
                    hitMeasures["queryTime"] = hitMeasures.get("queryTime", 0) + queryTimeHit
                    hitMeasures["timeRange"] = hitMeasures.get("timeRange", 0) + timeRangeHit
                    hitMeasures["queryGranularity"] = hitMeasures.get("queryGranularity", 0) + queryGranularityHit
                    hitMeasures["projCols"] = hitMeasures.get("projCols", 0) + projColsHit
                    hitMeasures["selCols"] = hitMeasures.get("selCols", 0) + selColsHit
                    # 预查询模板命中率（维度命中，时间范围命中，聚合粒度命中）
                    if groupByHit == 1 and timeRangeHit == 1 and queryGranularityHit == 1:
                        hitMeasures["template"] = hitMeasures.get("template", 0) + 1
            print("eval: ", precision, recall, FMeasure, accuracy, hit)
    saveEvalMetrics(precisions, recalls, FMeasures, outputEvalExcel)
    for key, value in hitMeasures.items():
        hitMeasures[key] = value / (total)
    print("threshold: ",threshold,", hitMeasures: ", hitMeasures, " total: ", total)