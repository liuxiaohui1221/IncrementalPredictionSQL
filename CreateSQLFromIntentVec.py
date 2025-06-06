from __future__ import division

from copy import copy, deepcopy

import numpy as np
from bitmap import BitMap
import ParseConfigFile as parseConfig
import argparse
import ReverseEnggQueries
import socket

class SQLForBitMapIntent:
    # 初始化SQLForBitMapIntent类
    def __init__(self, schemaDicts, intentBitVec, newSetBitPosList):
        # 存储schema字典
        self.schemaDicts = schemaDicts
        # 存储意图位向量
        self.intentBitVec = intentBitVec
        # 存储新设置的位位置列表
        self.newSetBitPosList = newSetBitPosList
        # 存储查询类型
        # self.queryType = None
        # 查询事件时间离散特征：周期，小时
        self.queryTime = None
        # 存储表名
        self.tables = []
        # 存储投影列
        self.projCols = []
        # 存储平均列
        self.avgCols = []
        # 存储最小列
        self.minCols = []
        # 存储最大列
        self.maxCols = []
        # 存储求和列
        self.sumCols = []
        # 存储计数列
        self.countCols = []
        # 存储选择列
        self.selCols = []
        # 存储分组列
        self.groupByCols = []
        # 存储排序列
        self.orderByCols = []
        # 存储筛选列
        self.havingCols = []
        # 存储限制
        self.limit = None
        # 存储连接谓词
        self.joinPreds = []
        # 存储查询条件时间偏移量
        self.timeOffset = None
        # 存储查询条件时间范围
        self.timeRange = None
        # 存储查询聚合粒度
        self.queryGran = None


class SQLForIntentStr:
    def __init__(self, schemaDicts, intentVec):
        self.schemaDicts = schemaDicts
        self.intentVec = intentVec
        # following populates the required sub-bitmaps
        # self.queryTypeBitMap = self.intentVec[schemaDicts.queryTypeStartBitIndex:schemaDicts.queryTypeStartBitIndex + schemaDicts.queryTypeBitMapSize]
        self.queryTimeBitMap = self.intentVec[schemaDicts.queryTimeStartBitIndex:schemaDicts.queryTimeStartBitIndex + schemaDicts.queryTimeBitMapSize]
        self.tableBitMap = self.intentVec[schemaDicts.tableStartBitIndex:schemaDicts.tableStartBitIndex + schemaDicts.tableBitMapSize]
        self.projectionBitMap = self.intentVec[schemaDicts.projectionStartBitIndex:schemaDicts.projectionStartBitIndex + schemaDicts.allColumnsSize]
        self.avgBitMap = self.intentVec[schemaDicts.avgStartBitIndex:schemaDicts.avgStartBitIndex + schemaDicts.allColumnsSize]
        self.minBitMap = self.intentVec[schemaDicts.minStartBitIndex:schemaDicts.minStartBitIndex + schemaDicts.allColumnsSize]
        self.maxBitMap = self.intentVec[schemaDicts.maxStartBitIndex:schemaDicts.maxStartBitIndex + schemaDicts.allColumnsSize]
        self.sumBitMap = self.intentVec[schemaDicts.sumStartBitIndex:schemaDicts.sumStartBitIndex + schemaDicts.allColumnsSize]
        self.countBitMap = self.intentVec[schemaDicts.countStartBitIndex:schemaDicts.countStartBitIndex + schemaDicts.allColumnsSize]
        self.selectionBitMap = self.intentVec[schemaDicts.selectionStartBitIndex:schemaDicts.selectionStartBitIndex + schemaDicts.allColumnsSize]
        self.groupByBitMap = self.intentVec[schemaDicts.groupByStartBitIndex:schemaDicts.groupByStartBitIndex + schemaDicts.allColumnsSize]
        self.orderByBitMap = self.intentVec[schemaDicts.orderByStartBitIndex:schemaDicts.orderByStartBitIndex + schemaDicts.allColumnsSize]
        self.havingBitMap = self.intentVec[schemaDicts.havingStartBitIndex:schemaDicts.havingStartBitIndex + schemaDicts.allColumnsSize]
        self.limitBitMap = self.intentVec[schemaDicts.limitStartBitIndex:schemaDicts.limitStartBitIndex + schemaDicts.limitBitMapSize]
        self.joinPredicatesBitMap = self.intentVec[schemaDicts.joinPredicatesStartBitIndex:schemaDicts.joinPredicatesStartBitIndex + schemaDicts.joinPredicatesBitMapSize]
        self.timeOffsetBitMap = self.intentVec[schemaDicts.timeOffsetGranStartBitIndex:schemaDicts.timeOffsetGranStartBitIndex + schemaDicts.granBitMapSize]
        self.timeRangeBitMap = self.intentVec[schemaDicts.timeRangeGranStartBitIndex:schemaDicts.timeRangeGranStartBitIndex + schemaDicts.granBitMapSize]
        self.queryGranBitMap = self.intentVec[schemaDicts.queryGranStartBitIndex:schemaDicts.queryGranStartBitIndex + schemaDicts.granBitMapSize]
        self.queryType = None
        self.queryTime = None
        self.tables = []
        self.projCols = []
        self.avgCols = []
        self.minCols = []
        self.maxCols = []
        self.sumCols = []
        self.countCols = []
        self.selCols = []
        self.groupByCols = []
        self.orderByCols = []
        self.havingCols = []
        self.limit = None
        self.joinPreds = []
        self.timeOffset = None
        self.timeRange = None
        self.queryGran = None


def readIntentObjectsFromFile(intentFileName):
    intentObjDict = {}
    with open(intentFileName) as f:
        for line in f:
            tokens = line.strip().split(":")
            if len(tokens) == 2:
                intentObjDict[tokens[0]] =tokens[1]
    return intentObjDict

def initIntentStrObj(schemaDicts, intentVec):
    intentObj = SQLForIntentStr(schemaDicts, intentVec)
    return intentObj

def assertSize(intentObj):
    # estAllColumnsSize = estGroupByBitMapSize = estOrderByBitMapSize = estProjectionBitMapSize = estHavingBitMapSize =
    # estMinBitMapSize = estMaxBitMapSize = estAvgBitMapSize = estSumBitMapSize = estCountBitMapSize
    print("estAllOpSize: "+str(intentObj.schemaDicts.allOpSize)+", len(intentVec): "+str(len(intentObj.intentVec)))
    return

def assertIntentOpObjects(intentObj, intentObjDict):
    assertSize(intentObj)
    sameObjs = 1
    if(intentObj.queryTypeBitMap != intentObjDict['queryTypeBitMap'] or intentObj.tableBitMap != intentObjDict['TableBitMap']
       or intentObj.groupByBitMap != intentObjDict['GroupByBitMap'] or intentObj.orderByBitMap != intentObjDict['OrderByBitMap']
       or intentObj.projectionBitMap != intentObjDict['ProjectionBitMap'] or intentObj.havingBitMap != intentObjDict['HavingBitMap']
       or intentObj.joinPredicatesBitMap != intentObjDict['JoinPredicatesBitMap'] or intentObj.limitBitMap != intentObjDict['LimitBitMap']
       or intentObj.minBitMap != intentObjDict['MINBitMap'] or intentObj.maxBitMap != intentObjDict['MAXBitMap']
       or intentObj.avgBitMap != intentObjDict['AVGBitMap'] or intentObj.sumBitMap != intentObjDict['SUMBitMap']
       or intentObj.countBitMap != intentObjDict['COUNTBitMap'] or intentObj.selectionBitMap != intentObjDict['SelectionBitMap']):
        sameObjs = 0
    print("Assertion outcome: "+str(sameObjs))
    return sameObjs

def populateQueryTypeStr(intentObj):
    assert len(intentObj.queryTypeBitMap) == 4 # select, update, insert, delete
    if int(intentObj.queryTypeBitMap[0]) == 1:
        intentObj.queryType = "select"
    elif int(intentObj.queryTypeBitMap[1]) == 1:
        intentObj.queryType = "update"
    elif int(intentObj.queryTypeBitMap[2]) == 1:
        intentObj.queryType = "insert"
    elif int(intentObj.queryTypeBitMap[3]) == 1:
        intentObj.queryType = "delete"
    return intentObj

def searchForTable(tableDict, index):
    for table in tableDict:
        val = tableDict[table]
        if int(val) == index:
            return table
    return None

def populateTablesStr(intentObj):
    assert len(intentObj.tables) == 0
    for i in range(len(intentObj.tableBitMap)):
        if int(intentObj.tableBitMap[i]) == 1:
            tableName = intentObj.schemaDicts.tableOrderDict[i]
            assert tableName is not None
            intentObj.tables.append(tableName)
    return intentObj

def populateLimitStr(intentObj):
    assert len(intentObj.limitBitMap)==1
    if int(intentObj.limitBitMap[0])==1:
        intentObj.limit = "Limit"
    return intentObj

def populateColsForOpStr(opCols, opBitMap, intentObj):
    assert len(opCols) == 0
    bitMapIndex=0
    for tableIndex in range(len(intentObj.schemaDicts.tableOrderDict)):
        tableName = intentObj.schemaDicts.tableOrderDict[tableIndex]
        colList = intentObj.schemaDicts.colDict[tableName]
        for col in colList:
            if int(opBitMap[bitMapIndex]) == 1:
                opCols.append(tableName+"."+col)
            bitMapIndex+=1
    return opCols

def populateJoinPredsStr(intentObj):
    assert len(intentObj.joinPreds) == 0
    for tablePairIndex in intentObj.schemaDicts.joinPredBitPosDict:
        startEndBitPos = intentObj.schemaDicts.joinPredBitPosDict[tablePairIndex]
        startBitPos = startEndBitPos[0]
        endBitPos = startEndBitPos[1]
        for i in range(startBitPos, endBitPos+1):
            if int(intentObj.joinPredicatesBitMap[i])==1:
                joinColPair = intentObj.schemaDicts.joinPredDict[tablePairIndex][i-startBitPos]
                joinStrToAppend = tablePairIndex.split(",")[0] + "." + joinColPair.split(",")[0]+ "," + tablePairIndex.split(",")[1] + "." + joinColPair.split(",")[1]
                intentObj.joinPreds.append(joinStrToAppend)
    return intentObj

def createSQLFromIntentString(intentObj):
    # intentObj = populateQueryTypeStr(intentObj)
    intentObj = populateQueryTimeStr(intentObj)
    intentObj = populateTablesStr(intentObj)
    intentObj.projCols = populateColsForOpStr(intentObj.projCols, intentObj.projectionBitMap, intentObj)
    intentObj.avgCols = populateColsForOpStr(intentObj.avgCols, intentObj.projectionBitMap, intentObj)
    intentObj.minCols = populateColsForOpStr(intentObj.minCols, intentObj.minBitMap, intentObj)
    intentObj.maxCols = populateColsForOpStr(intentObj.maxCols, intentObj.maxBitMap, intentObj)
    intentObj.sumCols = populateColsForOpStr(intentObj.sumCols, intentObj.sumBitMap, intentObj)
    intentObj.countCols = populateColsForOpStr(intentObj.countCols, intentObj.countBitMap, intentObj)
    intentObj.selCols = populateColsForOpStr(intentObj.selCols, intentObj.selectionBitMap, intentObj)
    intentObj.groupByCols = populateColsForOpStr(intentObj.groupByCols, intentObj.groupByBitMap, intentObj)
    intentObj.orderByCols = populateColsForOpStr(intentObj.orderByCols, intentObj.orderByBitMap, intentObj)
    intentObj.havingCols = populateColsForOpStr(intentObj.havingCols, intentObj.havingBitMap, intentObj)
    intentObj = populateLimitStr(intentObj)
    intentObj = populateJoinPredsStr(intentObj)
    return intentObj

def createSQLStringForTable(intentObj):
    actualSQLStr = str(None)
    if len(intentObj.tables) > 0:
        actualSQLStr = "Tables: "+str(intentObj.tables) + "\n"
    return actualSQLStr

def createSQLString(intentObj):
    actualSQLStr = "----Query ops----:\n"
    sqlOpsDict = {}
    if intentObj.queryTime is not None:
        actualSQLStr += "Query Time: "+str(intentObj.queryTime)+"\n"
        sqlOpsDict["queryTime"] = intentObj.queryTime
    if intentObj.timeOffset is not None:
        actualSQLStr += "Time Offset: "+str(intentObj.timeOffset)+"\n"
        sqlOpsDict["timeOffset"] = intentObj.timeOffset
    if intentObj.timeRange is not None:
        actualSQLStr += "Time Range: "+str(intentObj.timeRange)+"\n"
        sqlOpsDict["timeRange"] = intentObj.timeRange
    if intentObj.queryGran is not None:
        actualSQLStr += "Query Granularity: "+str(intentObj.queryGran)+"\n"
        sqlOpsDict["queryGranularity"] = intentObj.queryGran
    if intentObj.tables is not None and len(intentObj.tables) > 0:
        actualSQLStr += "Tables: "+str(intentObj.tables) + "\n"
        sqlOpsDict["tables"] = intentObj.tables
    if intentObj.projCols is not None and len(intentObj.projCols) > 0:
        actualSQLStr += "Projected Columns: " + str(intentObj.projCols) + "\n"
        sqlOpsDict["projCols"] = intentObj.projCols
    if intentObj.avgCols is not None and len(intentObj.avgCols) > 0:
        actualSQLStr += "AVG Columns: " + str(intentObj.avgCols) + "\n"
        sqlOpsDict["avgCols"] = intentObj.avgCols
    if intentObj.minCols is not None and len(intentObj.minCols) > 0:
        actualSQLStr += "MIN Columns: " + str(intentObj.minCols) + "\n"
        sqlOpsDict["minCols"] = intentObj.minCols
    if intentObj.maxCols is not None and len(intentObj.maxCols) > 0:
        actualSQLStr += "MAX Columns: " + str(intentObj.maxCols) + "\n"
        sqlOpsDict["maxCols"] = intentObj.maxCols
    if intentObj.sumCols is not None and len(intentObj.sumCols) > 0:
        actualSQLStr += "SUM Columns: " + str(intentObj.sumCols) + "\n"
        sqlOpsDict["sumCols"] = intentObj.sumCols
    if intentObj.countCols is not None and len(intentObj.countCols) > 0:
        actualSQLStr += "COUNT Columns: " + str(intentObj.countCols) + "\n"
        sqlOpsDict["countCols"] = intentObj.countCols
    if intentObj.selCols is not None and len(intentObj.selCols) > 0:
        actualSQLStr += "SEL Columns: " + str(intentObj.selCols) + "\n"
        sqlOpsDict["selCols"] = intentObj.selCols
    if intentObj.groupByCols is not None and len(intentObj.groupByCols) > 0:
        actualSQLStr += "GROUP BY Columns: " + str(intentObj.groupByCols) + "\n"
        sqlOpsDict["groupByCols"] = intentObj.groupByCols
    if intentObj.orderByCols is not None and len(intentObj.orderByCols) > 0:
        actualSQLStr += "ORDER BY Columns: " + str(intentObj.orderByCols) + "\n"
        sqlOpsDict["orderByCols"] = intentObj.orderByCols
    if intentObj.havingCols is not None and len(intentObj.havingCols) > 0:
        actualSQLStr += "HAVING Columns: " + str(intentObj.havingCols) + "\n"
        sqlOpsDict["havingCols"] = intentObj.havingCols
    if intentObj.limit is not None:
        actualSQLStr += "Limit: "+str(intentObj.limit)+"\n"
        sqlOpsDict["limit"] = intentObj.limit
    if intentObj.joinPreds is not None and len(intentObj.joinPreds) > 0:
        actualSQLStr += "JOIN PRED ColPairs: " + str(intentObj.joinPreds) + "\n"
        sqlOpsDict["joinPreds"] = intentObj.joinPreds
    return sqlOpsDict


def printSQLOps(intentObj):
    print("Query Type: "+str(intentObj.queryType))
    print("Tables: "+str(intentObj.tables))
    print("Projected Columns: "+str(intentObj.projCols))
    print("AVG Columns: "+str(intentObj.avgCols))
    print("MIN Columns: "+str(intentObj.minCols))
    print("MAX Columns: " + str(intentObj.maxCols))
    print("SUM Columns: " + str(intentObj.sumCols))
    print("COUNT Columns: " + str(intentObj.countCols))
    print("SEL Columns: " + str(intentObj.selCols))
    print("GROUP BY Columns: " + str(intentObj.groupByCols))
    print("ORDER BY Columns: " + str(intentObj.orderByCols))
    print("HAVING Columns: " + str(intentObj.havingCols))
    print("Limit: " + str(intentObj.limit))
    print("JOIN PRED ColPairs: "+ str(intentObj.joinPreds))

def checkBitMapWorking(intentObjDict):
    bitStr = "0100000001110001000"
    b = BitMap.fromstring(bitStr)
    if(b.size()>len(bitStr)):
        startPos = b.size() % 8
        bitStr = b.tostring()[startPos:b.size()]
    print(bitStr)
    print(b.nonzero())
    print(b.test(3))
    print(b.size())
    print(len(bitStr))
    print("Length of intentObjDict['intentVector']: " + str(len(intentObjDict['intentVector'])))
    bitmap = BitMap.fromstring(intentObjDict['intentVector'])
    print("Length of bitmap.tostring(): " + str(len(bitmap.tostring())))

def createSQLFromIntentStrSanityCheck(schemaDicts, intentObjDict):
    intentObj = initIntentStrObj(schemaDicts, intentObjDict['intentVector'])
    assertIntentOpObjects(intentObj, intentObjDict)
    createSQLFromIntentString(intentObj)
    # printSQLOps(intentObj)

def checkOpToPopulate(newSetBitPos, intentObj):
    if newSetBitPos >= intentObj.schemaDicts.joinPredicatesStartBitIndex:
        return "join"
    elif newSetBitPos >= intentObj.schemaDicts.limitStartBitIndex:
        return "limit"
    elif newSetBitPos >= intentObj.schemaDicts.havingStartBitIndex:
        return "having"
    elif newSetBitPos >= intentObj.schemaDicts.orderByStartBitIndex:
        return "orderby"
    elif newSetBitPos >= intentObj.schemaDicts.groupByStartBitIndex:
        return "groupby"
    elif newSetBitPos >= intentObj.schemaDicts.selectionStartBitIndex:
        return "select"
    elif newSetBitPos >= intentObj.schemaDicts.countStartBitIndex:
        return "count"
    elif newSetBitPos >= intentObj.schemaDicts.sumStartBitIndex:
        return "sum"
    elif newSetBitPos >= intentObj.schemaDicts.maxStartBitIndex:
        return "max"
    elif newSetBitPos >= intentObj.schemaDicts.minStartBitIndex:
        return "min"
    elif newSetBitPos >= intentObj.schemaDicts.avgStartBitIndex:
        return "avg"
    elif newSetBitPos >= intentObj.schemaDicts.projectionStartBitIndex:
        return "project"
    elif newSetBitPos >= intentObj.schemaDicts.tableStartBitIndex:
        return "table"
    elif newSetBitPos >= intentObj.schemaDicts.queryTypeStartBitIndex:
        return "querytype"
    else:
        print("not possible !!")


def populateQueryTimeStr(intentObj):
    assert len(intentObj.queryTimeBitMap) == intentObj.schemaDicts.queryTimeBitMapSize
    week=-1
    hour=-1
    for i in range(len(intentObj.queryTimeBitMap)):
        if int(intentObj.queryTimeBitMap[i]) == 1:
            if(i<7):
                week = i+1
            else:
                hour = i-7
    intentObj.queryTime = str(week) + "w" + str(hour) + "h"
    return intentObj


def populateOps(intentObj, opsToPopulate):
    for opToPopulate in opsToPopulate:
        if opToPopulate == "querytype":
            intentObj = populateQueryTypeStr(intentObj)
        elif opToPopulate == "querytime":
            intentObj = populateQueryTimeStr(intentObj)
        elif opToPopulate == "table":
            intentObj = populateTablesStr(intentObj)
        elif opToPopulate == "project":
            intentObj.projCols = populateColsForOpStr(intentObj.projCols, intentObj.projectionBitMap, intentObj)
        elif opToPopulate == "avg":
            intentObj.avgCols = populateColsForOpStr(intentObj.avgCols, intentObj.projectionBitMap, intentObj)
        elif opToPopulate == "min":
            intentObj.minCols = populateColsForOpStr(intentObj.minCols, intentObj.minBitMap, intentObj)
        elif opToPopulate == "max":
            intentObj.maxCols = populateColsForOpStr(intentObj.maxCols, intentObj.maxBitMap, intentObj)
        elif opToPopulate == "sum":
            intentObj.sumCols = populateColsForOpStr(intentObj.sumCols, intentObj.sumBitMap, intentObj)
        elif opToPopulate == "count":
            intentObj.countCols = populateColsForOpStr(intentObj.countCols, intentObj.countBitMap, intentObj)
        elif opToPopulate == "select":
            intentObj.selCols = populateColsForOpStr(intentObj.selCols, intentObj.selectionBitMap, intentObj)
        elif opToPopulate == "groupby":
            intentObj.groupByCols = populateColsForOpStr(intentObj.groupByCols, intentObj.groupByBitMap, intentObj)
        elif opToPopulate == "orderby":
            intentObj.orderByCols = populateColsForOpStr(intentObj.orderByCols, intentObj.orderByBitMap, intentObj)
        elif opToPopulate == "having":
            intentObj.havingCols = populateColsForOpStr(intentObj.havingCols, intentObj.havingBitMap, intentObj)
        elif opToPopulate == "limit":
            intentObj = populateLimitStr(intentObj)
        elif opToPopulate == "join":
            intentObj = populateJoinPredsStr(intentObj)
    return intentObj

def createSQLFromIntentStringBitPos(intentObj, newSetBitPosList):
    opsToPopulate = set()
    for newSetBitPos in newSetBitPosList:
        opToPopulate = checkOpToPopulate(newSetBitPos, intentObj)
        opsToPopulate.add(opToPopulate)
    populateOps(intentObj, opsToPopulate)
    #printSQLOps(intentObj)

def populateSQLOpFromType(intentObj, sqlOp, opType):
    assert (opType == "querytype" or opType == "querytime" or opType == "table" or opType == "project" or opType ==
            "avg" or opType == "min" \
           or opType == "max" or opType == "sum" or opType == "count" or opType == "select" or opType == "groupby" \
           or opType == "orderby" or opType == "having" or opType == "limit" or opType == "join" or opType ==
            "timeoffset" or opType == "timerange" or opType == "querygran")
    if opType == "querytype":
        intentObj.queryType = sqlOp
    elif opType == "querytime":
        if intentObj.queryTime == None:
            intentObj.queryTime = sqlOp
        else:
            intentObj.queryTime = intentObj.queryTime +","+ sqlOp
    elif opType == "table":
        intentObj.tables.append(sqlOp)
    elif opType == "project":
        intentObj.projCols.append(sqlOp)
    elif opType == "avg":
        intentObj.avgCols.append(sqlOp)
    elif opType == "min":
        intentObj.minCols.append(sqlOp)
    elif opType == "max":
        intentObj.maxCols.append(sqlOp)
    elif opType == "sum":
        intentObj.sumCols.append(sqlOp)
    elif opType == "count":
        intentObj.countCols.append(sqlOp)
    elif opType == "select":
        intentObj.selCols.append(sqlOp)
    elif opType == "groupby":
        intentObj.groupByCols.append(sqlOp)
    elif opType == "orderby":
        intentObj.orderByCols.append(sqlOp)
    elif opType == "having":
        intentObj.havingCols.append(sqlOp)
    elif opType == "limit":
        intentObj.limit = "Limit"
    elif opType == "join":
        intentObj.joinPreds.append(sqlOp)
    elif opType == "timeoffset":
        intentObj.timeOffset=sqlOp
    elif opType == "timerange":
        if intentObj.timeRange==None:
            intentObj.timeRange=sqlOp
    elif opType == "querygran":
        if intentObj.queryGran==None:
            intentObj.queryGran=sqlOp
    else:
        print("OpError !!")
    return intentObj

def createSQLTableFromIntentBits(intentObj):
    # 遍历intentObj中的newSetBitPosList
    for setBitIndex in intentObj.newSetBitPosList:
        # 计算tempIndex
        tempIndex = setBitIndex + intentObj.schemaDicts.tableStartBitIndex
        # 如果tempIndex在forwardMapBitsToOps中
        if tempIndex in intentObj.schemaDicts.forwardMapBitsToOps:
            # 获取setSQLOp
            setSQLOp = intentObj.schemaDicts.forwardMapBitsToOps[tempIndex]
            # 将setSQLOp按";"分割
            opTokens = setSQLOp.split(";")
            # 获取sqlOp
            sqlOp = opTokens[0]
            # 断言sqlOp等于tableOrderDict中的setBitIndex
            assert sqlOp == intentObj.schemaDicts.tableOrderDict[setBitIndex]
            # 获取opType
            opType = opTokens[1]
            # 根据opType填充intentObj中的SQLOp
            intentObj = populateSQLOpFromType(intentObj, sqlOp, opType)
    # 打印SQLOps
    #printSQLOps(intentObj)
    # 返回intentObj
    return intentObj

def createSQLFromIntentBits(intentObj):
    # 遍历intentObj中的newSetBitPosList
    topKTables = intentObj.schemaDicts.tableBitMapSize
    topNQuerys = intentObj.schemaDicts.topQueryN
    intentObjArrays = [[None for _ in range(topNQuerys)] for _ in range(topKTables)]
    for i in range(topKTables):
        for j in range(topNQuerys):
            intentObjArrays[i][j] = deepcopy(intentObj)  # 初始化每个查询
            (intentObjArrays[i][j]).tables.append(intentObj.schemaDicts.tableOrderDict[i])
    for setBitIndex in intentObj.newSetBitPosList:
        # 如果setBitIndex在intentObj的schemaDicts.forwardMapBitsToOps中
        if setBitIndex in intentObj.schemaDicts.forwardMapBitsToOps:
            # 获取setSQLOp
            setSQLOp = intentObj.schemaDicts.forwardMapBitsToOps[setBitIndex]
            # 将setSQLOp按照";"分割，得到sqlOp和opType
            opTokens = setSQLOp.split(";")
            sqlOp = opTokens[0]
            opType = opTokens[1]
            if intentObj.schemaDicts.embeddingType=="table" and len(opTokens)==3:
                #todo 第三位为索引位：tableIndex,queryIndex
                opTableAndQueryIndexs = opTokens[2].split(',')
                tableIndex=int(opTableAndQueryIndexs[0])
                queryIndex=int(opTableAndQueryIndexs[1])
                intentObjArrays[tableIndex][queryIndex] = populateSQLOpFromType(
                    intentObjArrays[tableIndex][queryIndex], sqlOp, opType)
            else:
                # 根据opType调用populateSQLOpFromType函数，得到新的intentObj
                intentObj = populateSQLOpFromType(intentObj, sqlOp, opType)
    # printSQLOps(intentObj)
    # 返回新的intentObj
    return intentObjArrays

def setBit(opDimBit, intentObj):
    revBitPos = intentObj.schemaDicts.allOpSize - 1 - opDimBit
    assert intentObj.intentBitVec.test(revBitPos) == False
    intentObj.intentBitVec.flip(revBitPos)
    return intentObj

def unsetBit(opDimBit, intentObj):
    revBitPos = intentObj.schemaDicts.allOpSize - 1 - opDimBit
    assert intentObj.intentBitVec.test(revBitPos) == True
    intentObj.intentBitVec.flip(revBitPos)
    return intentObj

def fixColumnTableViolations(intentObj, opString, precOrRecallFavor):
    if opString == "project":
        colsToFix = intentObj.projCols
    elif opString == "avg":
        colsToFix = intentObj.avgCols
    elif opString == "min":
        colsToFix = intentObj.minCols
    elif opString == "max":
        colsToFix = intentObj.maxCols
    elif opString == "sum":
        colsToFix = intentObj.sumCols
    elif opString == "count":
        colsToFix = intentObj.countCols
    elif opString == "select": # this refers to the selection predicate or where clause should not be confused with the select keyword for queryType
        colsToFix = intentObj.selCols
    elif opString == "groupby":
        colsToFix = intentObj.groupByCols
    elif opString == "orderby":
        colsToFix = intentObj.orderByCols
    elif opString == "having":
        colsToFix = intentObj.havingCols
    copyCols = list(colsToFix)
    for projAttr in copyCols:
        tableName = projAttr.split(".")[0]
        if tableName not in intentObj.tables: #and intentObj.queryType == "select": # fix projections for select queries as inserts, updates and deletes do not project attributes for now
            if precOrRecallFavor == "precision":
                # drop the column to preserve precision
                colsToFix.remove(projAttr)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[projAttr+";"+opString]
                intentObj = unsetBit(opDimBit, intentObj)
            elif precOrRecallFavor == "recall":
                # add the table name to increase recall
                intentObj.tables.append(tableName)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[tableName+";table"]
                intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixJoinViolations(intentObj, precOrRecallFavor):
    copyJoinPreds = list(intentObj.joinPreds)
    for joinPred in copyJoinPreds:
        colPair = []
        leftTabCol = joinPred.split(",")[0]
        rightTabCol = joinPred.split(",")[1]
        colPair.append(leftTabCol)
        colPair.append(rightTabCol)
        for joinCol in colPair:
            tableName = joinCol.split(".")[0]
            if tableName not in intentObj.tables:  # and intentObj.queryType == "select": # fix projections for select queries as inserts, updates and deletes do not project attributes for now
                if precOrRecallFavor == "precision":
                    # drop the join predicate to preserve precision
                    if joinPred in intentObj.joinPreds: # because the same predicate may be dropped twice for the two cols
                        intentObj.joinPreds.remove(joinPred)
                        opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[joinPred+";join"]
                        intentObj = unsetBit(opDimBit, intentObj)
                elif precOrRecallFavor == "recall":
                    # add the table name to increase recall
                    intentObj.tables.append(tableName)
                    opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[tableName+";table"]
                    intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixGroupByViolation1(intentObj, precOrRecallFavor):
    # Violation 1: attribute in projection and not in min,max,sum,avg,count(aggr), (while there is at least one diff aggr column) but not in group by
    if len(intentObj.avgCols) == 0 and len(intentObj.minCols) == 0 and len(intentObj.maxCols) == 0 and \
                    len(intentObj.sumCols) == 0 and len(intentObj.countCols) == 0:
        return intentObj
    for projCol in intentObj.projCols:
        if projCol not in intentObj.avgCols and projCol not in intentObj.minCols and projCol not in intentObj.maxCols \
                and projCol not in intentObj.sumCols and projCol not in intentObj.countCols \
                and projCol not in intentObj.groupByCols:
            if precOrRecallFavor == "precision":
                # drop the projection column
                intentObj.projCols.remove(projCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[projCol+";project"]
                intentObj = unsetBit(opDimBit, intentObj)
            elif precOrRecallFavor == "recall":
                # add the projection column to the group by columnList
                intentObj.groupByCols.append(projCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[projCol+";groupby"]
                intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixGroupByViolation2(intentObj, precOrRecallFavor):
    # Violation 2: An attribute is in the group by list but not in the projection list while aggr list is empty - aggr cols are also included in the projection list
    # since we cannot predict which aggr op a column belongs to or if at all it is just projected and doesnt belong to any aggregate, the fact
    # that it is in a group by list should make it  part of the projection list
    for grpByCol in intentObj.groupByCols:
        if grpByCol not in intentObj.projCols and len(intentObj.avgCols) == 0 and len(intentObj.minCols) == 0 and len(intentObj.maxCols) == 0 and \
                        len(intentObj.sumCols) == 0 and len(intentObj.countCols) == 0:
            if precOrRecallFavor == "precision":
                # drop the group by column
                intentObj.groupByCols.remove(grpByCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[grpByCol+";groupby"]
                intentObj = unsetBit(opDimBit, intentObj)
            elif precOrRecallFavor == "recall":
                # add the group by column to the projection column list
                intentObj.projCols.append(grpByCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[grpByCol+";project"]
                intentObj = setBit(opDimBit, intentObj)
    return intentObj


def fixGroupByViolations(intentObj, precOrRecallFavor):
    # only column column violations are fixed as column table violations are already fixed
    fixGroupByViolation1(intentObj, precOrRecallFavor)
    fixGroupByViolation2(intentObj, precOrRecallFavor)
    return intentObj

def fixHavingViolations(intentObj, precOrRecallFavor):
    # if the attribute is present in having clause, the aggregate column should not be empty; if it is, add the having column to the projection list
    # and to an aggregate operator picked randomly, to increase recall. For precision, simply drop the having column
    for havingCol in intentObj.havingCols:
        if len(intentObj.avgCols) == 0 and len(intentObj.minCols) == 0 and len(intentObj.maxCols) == 0 and \
                        len(intentObj.sumCols) == 0 and len(intentObj.countCols) == 0:
            if precOrRecallFavor == "precision":
                # drop the having column
                intentObj.havingCols.remove(havingCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[havingCol+";having"]
                intentObj = unsetBit(opDimBit, intentObj)
            elif precOrRecallFavor == "recall":
                # add the having col to projection list if absent
                if havingCol not in intentObj.projCols:
                    intentObj.projCols.append(havingCol)
                    opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[havingCol + ";project"]
                    intentObj = setBit(opDimBit, intentObj)
                # add the having col to the avgCols -- may not be avg, but reqd for valid SQL
                intentObj.avgCols.append(havingCol)
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[havingCol + ";avg"]
                intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixNullQueryTypeWithPrevEffect(intentObj, curIntentObj):
    if intentObj.queryType is None:
        if curIntentObj is not None:
            # fixNullQueryTypeDefaultOtherQuery(intentObj)
            intentObj.queryType = "select" # borrow the querytype of the current query
            opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[intentObj.queryType+";querytype"]
        else:
            #select by default
            intentObj.queryType = "select"
            opDimBit = 0 # 0 for select
        intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixNullTableViolationsWithPrevEffect(intentObj, curIntentObj):
    if len(intentObj.tables) == 0:
        if curIntentObj is not None:
            intentObj.tables = list(curIntentObj.tables)
            for tableName in intentObj.tables:
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[tableName+";table"]
                intentObj = setBit(opDimBit, intentObj)
        else:
            # by default select the first table from the schema order
            tableStartBit = intentObj.schemaDicts.tableStartBitIndex
            intentObj = setBit(tableStartBit, intentObj)
            tableName = intentObj.schemaDicts.forwardMapBitsToOps[tableStartBit].split(";")[0] # coz the value has ;table in the end
            intentObj.tables.append(tableName)
    return intentObj

def fixNullProjColViolationsWithPrevEffect(intentObj, curIntentObj):
    if intentObj.queryType == "select" and len(intentObj.projCols) == 0:
        intersectTables = []
        if curIntentObj is not None:
            intersectTables = list(set(intentObj.tables).intersection(set(curIntentObj.tables)))
        if len(intersectTables) > 0 and len(curIntentObj.projCols) > 0:
            for curProjCol in curIntentObj.projCols:
                curTableName = curProjCol.split(".")[0]
                if curTableName in intersectTables:
                    intentObj.projCols.append(curProjCol)
                    opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[curProjCol+";project"]
                    setBit(opDimBit, intentObj)
        else:
            # by default project the first column from the first table
            if len(intentObj.tables) > 0:
                tableName = intentObj.tables[0]
                colName = intentObj.schemaDicts.colDict[tableName][0]
                opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[tableName+"."+colName+";project"]
                intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixNullQueryTypeDefaultOtherQuery(intentObj):
    if intentObj.queryType is None:
        #select by default
        intentObj.queryType = "select"
        opDimBit = 0 # 0 for select
        intentObj = setBit(opDimBit, intentObj)
    return intentObj

def fixNullTableViolationsDefaultOtherQuery(intentObj, curIntentObj):
    if len(intentObj.tables) == 0:
        # by default select the first table from the schema order
        tableStartBit = intentObj.schemaDicts.tableStartBitIndex
        intentObj = setBit(tableStartBit, intentObj)
        tableName = intentObj.schemaDicts.forwardMapBitsToOps[tableStartBit].split(";")[0] # coz the value has ;table in the end
        intentObj.tables.append(tableName)
    return intentObj

def fixNullProjColViolationsDefaultOtherQuery(intentObj, curIntentObj):
    if intentObj.queryType == "select" and len(intentObj.projCols) == 0:
        # by default project the first column from the first table
        if len(intentObj.tables) > 0:
            tableName = intentObj.tables[0]
            colName = intentObj.schemaDicts.colDict[tableName][0]
            opDimBit = intentObj.schemaDicts.backwardMapOpsToBits[tableName+"."+colName+";project"]
            intentObj = setBit(opDimBit, intentObj)
    return intentObj


def fixSQLViolations(intentObj, precOrRecallFavor, curIntentObj):
    assert precOrRecallFavor == "precision" or precOrRecallFavor == "recall"
    # fixNullQueryTypeWithPrevEffect(intentObj, curIntentObj)
    # no need to fix tables -- they are fixed automatically while fixing other operators
    fixColumnTableViolations(intentObj, "project", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "avg", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "min", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "max", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "sum", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "count", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "select", precOrRecallFavor) # this is also a column-table violation fix
    fixColumnTableViolations(intentObj, "groupby", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "orderby", precOrRecallFavor)
    fixColumnTableViolations(intentObj, "having", precOrRecallFavor)
    # no need to fix limit as it is associated with a constant and cannot cause violations
    fixJoinViolations(intentObj, precOrRecallFavor)
    # out of order by, group by and having -- order by and having can have columns different from projected columns but group by cannot
    fixGroupByViolations(intentObj, precOrRecallFavor)
    fixHavingViolations(intentObj, precOrRecallFavor)
    fixNullTableViolationsWithPrevEffect(intentObj, curIntentObj)
    fixNullProjColViolationsWithPrevEffect(intentObj, curIntentObj)
    return intentObj

def convertOldToNewSetBitsTable(schemaDicts, setBitPosList):
    newSetBitPosList = []
    for bitPos in setBitPosList:
        newBitPos = schemaDicts.tableBitMapSize - 1 - bitPos  # because 1s appear in reverse, no need to prune the extra padded bits
        if newBitPos >= 0: # sometimes padded bits may exceed the weight threshold and may get set
            newSetBitPosList.append(newBitPos)
    return newSetBitPosList

def regenerateSQLTableOrig(topKCandidateVector, curIntentBitVec, schemaDicts, configDict):
    setBitPosList = topKCandidateVector.nonzero() #返回非0元素索引
    #print setBitPosList
    #print schemaDicts.allOpSize
    newSetBitPosList = convertOldToNewSetBitsTable(schemaDicts, setBitPosList)
    if len(newSetBitPosList) == 0 and configDict['RNN_DEFAULT_CUR_QUERY'] == 'True' and curIntentBitVec is not None:
        setBitPosList = curIntentBitVec.nonzero()  # default to current query
        newSetBitPosList = convertOldToNewSetBitsTable(schemaDicts, setBitPosList)
    #print newSetBitPosList
    intentObj = SQLForBitMapIntent(schemaDicts, topKCandidateVector, newSetBitPosList)
    intentObj = createSQLTableFromIntentBits(intentObj)
    return intentObj

def regenerateSQLTable(dummyParam, topKCandidateVector, curIntentBitVec, schemaDicts, configDict):
    #dummyParam to be consistent with the five param usage in CreateSQLFromIntentVec_selOpConst.py
    return regenerateSQLTableOrig(topKCandidateVector, curIntentBitVec, schemaDicts, configDict)

def regenerateSQL(dummyParam, topKCandidateVector, schemaDicts):
    #dummyParam to be consistent with the three param usage in CreateSQLFromIntentVec_selOpConst.py
    return regenerateSQLOrig(topKCandidateVector, schemaDicts)

def regenerateSQLOrig(topKCandidateVector, schemaDicts):
    # 获取topKCandidateVector中非零元素的索引
    setBitPosList = topKCandidateVector.nonzero()
    # print("setBitPosList:",setBitPosList)
    #print schemaDicts.allOpSize
    newSetBitPosList = []
    # 遍历setBitPosList中的每个元素
    for bitPos in setBitPosList:
        newBitPos = schemaDicts.allOpSize - 1 - bitPos # because 1s appear in reverse, no need to prune the extra
        # padded bits
        newSetBitPosList.append(newBitPos)
    #print newSetBitPosList
    intentObj = SQLForBitMapIntent(schemaDicts, topKCandidateVector, newSetBitPosList)
    intentObjs = createSQLFromIntentBits(intentObj)
    return np.array(intentObjs)
def round_to_eight_diff(n: int) -> int:
    """计算正整数向上取整到最近的8的倍数后的差值
    Args:
        n: 需要处理的正整数 (n >= 1)
    Returns:
        原始值与向上取整结果的差值
    Examples:
        >>> round_to_eight_diff(5)
        3    # 5 → 8 (差3)
        >>> round_to_eight_diff(15)
        1    # 15 → 16 (差1)
        >>> round_to_eight_diff(8)
        0    # 8 → 8 (差0)
    """
    remainder = n % 8
    return 0 if remainder == 0 else 8 - remainder


def createSQLFromIntentBitMapSanityCheck(schemaDicts, intentObjDict):
    checkBitMapWorking(intentObjDict)
    intentBitMap = BitMap.fromstring(intentObjDict['intentVector'])
    intentStr = intentBitMap.tostring()
    if intentBitMap.size() > schemaDicts.allOpSize:
        startBit = intentBitMap.size() - schemaDicts.allOpSize
        intentStr = intentStr[startBit:intentBitMap.size()]
    assert intentStr == intentObjDict['intentVector']
    #intentObj = initIntentStrObj(schemaDicts, intentStr)
    #assertIntentOpObjects(intentObj, intentObjDict)
    #createSQLFromIntentStringBitPos(intentObj, newSetBitPosList)
    intentObj = regenerateSQL(None, intentBitMap, schemaDicts)
    return intentObj

if __name__ == "__main__":
    #configDict = parseConfig.parseConfigFile("configFile.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="Config parameters file", type=str, required=True)
    args = parser.parse_args()
    configDict = parseConfig.parseConfigFile(args.config)
    schemaDicts = ReverseEnggQueries.readSchemaDicts(configDict)
    tempVectorFileName = "/Users/postgres/data/MINC/InputOutput/tempVector"
    if socket.gethostname() == "en4119510l" or socket.gethostname() == "en4119509l" or socket.gethostname() == "en4119508l" or socket.gethostname() == "en4119507l":
        tempVectorFileName = "/hdd2/vamsiCodeData/data/MINC/InputOutput/tempVector"
    intentObjDict = readIntentObjectsFromFile(tempVectorFileName)
    #intentObjDict = readIntentObjectsFromFile()
    #createSQLFromIntentStrSanityCheck(schemaDicts, intentObjDict)
    intentObj = createSQLFromIntentBitMapSanityCheck(schemaDicts, intentObjDict)
    # printSQLOps(intentObj)
    #createSQLFromIntent(schemaDicts, intentObjDict['intentVector'])
