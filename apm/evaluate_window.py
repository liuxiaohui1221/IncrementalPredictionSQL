from __future__ import division
import sys
import os
import time
import QueryExecution as QExec
from bitmap import BitMap
import CFCosineSim
import TupleIntent as ti
import ParseConfigFile as parseConfig
import ParseResultsToExcel
import pickle
from ParseConfigFile import getConfig

def fetchIntentFileFromConfigDict(configDict):
    if configDict['INTENT_REP'] == 'TUPLE':
        intentSessionFile = getConfig(configDict['TUPLEINTENTSESSIONS'])
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        if configDict['RNN_PREDICT_QUERY_OR_TABLE'] == 'TABLE':
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_TABLE_INTENT_SESSIONS'])
        else:
            intentSessionFile = getConfig(configDict['BIT_FRAGMENT_INTENT_SESSIONS'])
    elif configDict['INTENT_REP'] == 'FRAGMENT' and configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        intentSessionFile = getConfig(configDict['WEIGHTED_FRAGMENT_INTENT_SESSIONS'])
    elif configDict['INTENT_REP'] == 'QUERY':
        intentSessionFile = getConfig(configDict['QUERY_INTENT_SESSIONS'])
    else:
        print("ConfigDict['INTENT_REP'] must either be TUPLE or FRAGMENT or QUERY !!")
        sys.exit(0)
    return intentSessionFile

def updateSessionDict(line, configDict, sessionStreamDict):
    # curQueryIntent query的one-hot编码后的bitmap类型
    (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
    if str(sessID)+","+str(queryID) in sessionStreamDict:
        print(str(sessID)+","+str(queryID)+ " already exists !!")
        sys.exit(0)
    sessionStreamDict[str(sessID)+","+str(queryID)] = curQueryIntent
    return (sessID, queryID, curQueryIntent, sessionStreamDict)

def updateSessionLineDict(line, configDict, sessionLineDict, newSessionLengthDict):
    (sessID, queryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
    #if (sessID == 36 or sessID == 30) and queryID > 212:
        #print "hi: in QR"
    if str(sessID)+","+str(queryID) in sessionLineDict:
        print(str(sessID)+","+str(queryID)+ " already exists !!")
        sys.exit(0)
    sessionLineDict[str(sessID)+","+str(queryID)] = line.strip()
    if sessID not in newSessionLengthDict:
        newSessionLengthDict[sessID] = 1
    elif sessID in newSessionLengthDict:
        newSessionLengthDict[sessID] = newSessionLengthDict[sessID]+1
    return (sessionLineDict, newSessionLengthDict)

def findNextQueryIntent(intentSessionFile, sessID, queryID, configDict, lines):
    #with open(intentSessionFile) as f:
    for line in lines:
        (curSessID, curQueryID, curQueryIntent) = retrieveSessIDQueryIDIntent(line, configDict)
        if curSessID == sessID and curQueryID == queryID:
            #f.close()
            return curQueryIntent
    print("Error: Could not find the nextQueryIntent !!")
    sys.exit(0)

def normalizeWeightedVector(curQueryIntent):
    tokens = curQueryIntent.split(";")
    total = 0.0
    for token in tokens:
        total = total+float(token)
    normalizedVector = []
    for token in tokens:
        normalizedVector.append(str(float(token)/total))
    res = ';'.join(normalizedVector)
    return res

def retrieveQueryAndIntent(line, configDict):
    tokens = line.strip().split(";")
    sqlQuery = tokens[1].replace("OrigQuery:","").strip()
    curQueryIntent = ';'.join(tokens[2:])
    if ";" not in curQueryIntent and configDict['BIT_OR_WEIGHTED'] == 'BIT':
        curQueryIntent = BitMap.fromstring(curQueryIntent)
    else:
        curQueryIntent = normalizeWeightedVector(curQueryIntent)
    return (sqlQuery, curQueryIntent)

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
        curQueryIntent = normalizeWeightedVector(curQueryIntent)
    return (sessID, queryID, curQueryIntent)

def computeWeightedVectorFromList(predictedY):
    topKPredictedIntents = []
    topKPredictedIntent = ';'.join(str(x) for x in predictedY)
    topKPredictedIntents.append(topKPredictedIntent)
    return topKPredictedIntents

def computePredictedOutputStrRNN(sessID, queryID, topKPredictedIntents, actualQueryIntent, numEpisodes, configDict):
    output_str = "Session:" + str(sessID) + ";Query:" + str(queryID) + ";#Episodes:" + str(
        numEpisodes) + ";ActualQueryIntent:"
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        output_str += actualQueryIntent.tostring()
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        if ";" in actualQueryIntent:
            actualQueryIntent.replace(";", ",")
        output_str += actualQueryIntent
    for k in range(len(topKPredictedIntents)):
        output_str += ";TOP_" + str(
            k) + "_PREDICTED_INTENT:"  # no assertion on topKSessQueryIndices and no appending of them to the output string
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            output_str += topKPredictedIntents[k].tostring()
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            output_str += topKPredictedIntents[k].replace(";", ",")
    return output_str

def appendPredictedRNNIntentToFile(sessID, queryID, topKPredictedIntents, actualQueryIntent, numEpisodes, outputIntentFileName, configDict, foldID):
    startAppendTime = time.time()
    output_str=computePredictedOutputStrRNN(sessID, queryID, topKPredictedIntents, actualQueryIntent, numEpisodes, configDict)
    ti.appendToFile(outputIntentFileName, output_str)
    if configDict['SINGULARITY_OR_KFOLD']=='KFOLD':
        print("FoldID: "+str(foldID)+", Predicted " + str(len(topKPredictedIntents)) + " query intent vectors for "
                                                                                       "Session " + str(sessID) + ", "
                                                                                                                  "Query " + str(queryID))
    elif configDict['SINGULARITY_OR_KFOLD']=='SINGULARITY':
        print ("Predicted " + str(len(topKPredictedIntents)) + " query intent vectors for Session " + str(sessID) +
               ", Query " + str(queryID))
    elapsedAppendTime = float(time.time() - startAppendTime)
    return elapsedAppendTime

def appendPredictedIntentsToFile(topKSessQueryIndices, topKPredictedIntents, sessID, queryID, actualQueryIntent, numEpisodes, configDict, outputIntentFileName, foldID):
    startAppendTime = time.time()
    output_str = "Session:"+str(sessID)+";Query:"+str(queryID)+";#Episodes:"+str(numEpisodes)+";ActualQueryIntent:"
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        output_str += actualQueryIntent.tostring()
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        if ";" in actualQueryIntent:
            actualQueryIntent.replace(";",",")
        output_str += actualQueryIntent
    assert len(topKSessQueryIndices) == len(topKPredictedIntents)
    for k in range(len(topKPredictedIntents)):
        output_str += ";TOP_" +str(k)+"_PREDICTED_INTENT_"+str(topKSessQueryIndices[k])+":"
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            output_str += topKPredictedIntents[k].tostring()
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            output_str += topKPredictedIntents[k].replace(";",",")
    ti.appendToFile(outputIntentFileName, output_str)
    if configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        print("FoldID: "+str(foldID)+" Predicted " + str(len(topKPredictedIntents)) + " query intent vectors for "
                                                                                      "Session " + str(
            sessID) + ", Query " + str(queryID))
    #elif configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        #print "Predicted "+str(len(topKPredictedIntents))+" query intent vectors for Session "+str(sessID)+", Query "+str(queryID)
    elapsedAppendTime = float(time.time()-startAppendTime)
    return elapsedAppendTime

def deleteIfExists(fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass
    return

def updateResponseTime(episodeResponseTimeDictName, episodeResponseTime, numEpisodes, startEpisode, elapsedAppendTime):
    episodeResponseTime[numEpisodes] = float(time.time()-startEpisode) - elapsedAppendTime # we exclude the time consumed by appending predicted intents to the output intent file
    print("Episode Response Time: "+str(episodeResponseTime[numEpisodes]))
    elapsedAppendTime = 0.0
    writeToPickleFile(episodeResponseTimeDictName, episodeResponseTime)
    startEpisode = time.time()
    return (episodeResponseTimeDictName, episodeResponseTime, startEpisode, elapsedAppendTime)

def createQueryExecIntentCreationTimes(configDict,split_i):
    # assert (configDict['DATASET'].lower()== 'apm' or configDict['DATASET'] == 'MINC' or configDict['DATASET'] ==
    #         'BusTracker')
    numQueries = 0
    episodeQueryExecutionTime = {}
    episodeIntentCreationTime = {}
    numEpisodes = 0
    tempExecTimeEpisode = 0.0
    tempIntentTimeEpisode = 0.0
    with open(getConfig(configDict['CONCURRENT_QUERY_SESSIONS']+split_i),encoding='utf-8') as f:
        for line in f:
            sessQueries = line.split(";")
            sessQueryName = sessQueries[0]
            sessQuery = sessQueries[1].strip()
            queryVocabulary = {}
            (queryVocabulary, resObj, queryExecutionTime, intentCreationTime) = QExec.executeQueryWithIntent(sessQuery,
                                                                                                             configDict,
                                                                                                             queryVocabulary)
            tempExecTimeEpisode += float(queryExecutionTime)
            tempIntentTimeEpisode += float(intentCreationTime)
            print("Executed and obtained intent for " + sessQueryName)
            numQueries += 1
            if numQueries % int(configDict['EPISODE_IN_QUERIES']) == 0:
                numEpisodes += 1
                episodeQueryExecutionTime[numEpisodes] = tempExecTimeEpisode
                episodeIntentCreationTime[numEpisodes] = tempIntentTimeEpisode
                tempExecTimeEpisode = 0.0
                tempIntentTimeEpisode = 0.0
        if (tempExecTimeEpisode > 0 or tempIntentTimeEpisode > 0):
            numEpisodes += 1
            episodeQueryExecutionTime[numEpisodes] = tempExecTimeEpisode
            episodeIntentCreationTime[numEpisodes] = tempIntentTimeEpisode
            tempExecTimeEpisode = 0.0
            tempIntentTimeEpisode = 0.0
    return (episodeQueryExecutionTime, episodeIntentCreationTime)

def writeKFoldTrainTestTimesToPickleFiles(avgTrainTime, avgTestTime, algoName, configDict):
    trainFN = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/avgKFoldTrainTime_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K']
    testFN = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/avgKFoldTestTime_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K']
    writeToPickleFile(trainFN, avgTrainTime)
    writeToPickleFile(testFN, avgTestTime)
    return (trainFN, testFN)

def readFromPickleFile(fileName):
    with open(fileName, 'rb') as handle:
        readObj = pickle.load(handle)
    return readObj

def computeBitFMeasure(actualQueryIntent, topKQueryIntent):
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

def writeToPickleFile(fileName, writeObj):
    with open(fileName, 'wb') as handle:
        pickle.dump(writeObj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def computeWeightedFMeasure(actualQueryIntent, topKQueryIntent, delimiter, configDict):
    groundTruthDims = actualQueryIntent.split(delimiter)
    predictedDims = topKQueryIntent.split(delimiter)
    assert groundTruthDims.size() == predictedDims.size()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pos in range(groundTruthDims.size()):
        if groundTruthDims[pos] == '1' and predictedDims[pos]  == '1':
            TP += 1
        elif groundTruthDims[pos] == '0' and predictedDims[pos]  == '0':
            TN += 1
        elif groundTruthDims[pos] == '1' and predictedDims[pos]  == '0':
            FN += 1
        elif groundTruthDims[pos] == '0' and predictedDims[pos]  == '1':
            FP += 1
    if TP == 0 and FP == 0:
        precision = 0.0
    else:
        precision = float(TP) / float(TP + FP)
    if TP == 0 and FN == 0:
        recall = 0.0
    else:
        recall = float(TP) / float(TP + FN)
    if precision == 0.0 and recall == 0.0:
        FMeasure = 0.0
    else:
        FMeasure = 2 * precision * recall / (precision + recall)
    accuracy = float(TP + TN) / float(TP + FP + TN + FN)
    return (precision, recall, FMeasure, accuracy)

def computeQueRIEFMeasureForEachEpisode(line, configDict):
    tokens = line.strip().split(";")
    sessID = tokens[0].split(":")[1]
    queryID = tokens[1].split(":")[1]
    numEpisodes = tokens[2].split(":")[1]
    precisionAtMaxFMeasure = 0.0
    recallAtMaxFMeasure = 0.0
    maxFMeasure = 0.0
    accuracyAtMaxFMeasure = 0.0
    hitMeasure = 0
    maxFIndex = -1
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        actualQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        actualQueryIntent = tokens[3].split(":")[1]
    for i in range(4, len(tokens)):#从4位置开始取出预测的topK个query intent
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
            (precision, recall, FMeasure, accuracy, hit) = computeBitFMeasure(actualQueryIntent, topKQueryIntent)
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            topKQueryIntent = tokens[i].split(":")[1]
            (precision, recall, FMeasure, accuracy) = computeWeightedFMeasure(actualQueryIntent, topKQueryIntent, ",",
                                                                    configDict)
        if FMeasure > maxFMeasure:
            maxFMeasure = FMeasure
            precisionAtMaxFMeasure = precision
            recallAtMaxFMeasure = recall
            accuracyAtMaxFMeasure = accuracy
            hitMeasure = hit
            maxFIndex = i-4 # gives the topKIndex
        #if precision > maxPrecision:
        #if recall > maxRecall:
        #if accuracy > maxAccuracy:
    # print "float(len(tokens)-4 ="+str(len(tokens)-4)+", precision = "+str(precision/float(len(tokens)-4))
    return (sessID, queryID, numEpisodes, accuracyAtMaxFMeasure, precisionAtMaxFMeasure, recallAtMaxFMeasure, maxFMeasure, maxFIndex, hitMeasure)

def computeCosineSimFMeasureForEachEpisode(line, configDict):
    tokens = line.strip().split(";")
    sessID = tokens[0].split(":")[1]
    queryID = tokens[1].split(":")[1]
    numEpisodes = tokens[2].split(":")[1]
    precision = 0.0
    recall = 0.0
    maxCosineSim = 0.0
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    if configDict['BIT_OR_WEIGHTED'] == 'BIT':
        actualQueryIntent = BitMap.fromstring(tokens[3].split(":")[1])
    elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
        actualQueryIntent = tokens[3].split(":")[1]
    for i in range(4, len(tokens)):
        if configDict['BIT_OR_WEIGHTED'] == 'BIT':
            topKQueryIntent = BitMap.fromstring(tokens[i].split(":")[1])
            cosineSim = CFCosineSim.computeBitCosineSimilarity(actualQueryIntent, topKQueryIntent)
        elif configDict['BIT_OR_WEIGHTED'] == 'WEIGHTED':
            topKQueryIntent = tokens[i].split(":")[1]
            cosineSim = CFCosineSim.computeWeightedCosineSimilarity(actualQueryIntent, topKQueryIntent, ",",
                                                                    configDict)
        if cosineSim >= float(accThres):
            recall = 1.0
            precision += 1.0
        if cosineSim > maxCosineSim:
            maxCosineSim = cosineSim
    # print "float(len(tokens)-4 ="+str(len(tokens)-4)+", precision = "+str(precision/float(len(tokens)-4))
    precision /= float(len(tokens) - 4)
    if precision == 0 or recall == 0:
        FMeasure = 0
    else:
        FMeasure = 2 * precision * recall / (precision + recall)
    return (sessID, queryID, numEpisodes, maxCosineSim, precision, recall, FMeasure)

def computeAccuracyForEachEpisode(line, configDict):
    assert configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'COSINESIM' or configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'QUERIE'
    maxFIndex = -1
    if configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'COSINESIM':
        (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure) = computeCosineSimFMeasureForEachEpisode(line, configDict)
    elif configDict['COSINESIM_OR_QUERIE_FMEASURE'] == 'QUERIE':
        (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure, maxFIndex, hitMeasure) = computeQueRIEFMeasureForEachEpisode(line, configDict)
    return (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure, maxFIndex,hitMeasure)

def appendToDict(avgDict, key, value):
    if key not in avgDict:
        avgDict[key] = []
    avgDict[key].append(value)
    return avgDict

def computeAvgFoldAccuracy(kFoldOutputIntentFiles, configDict):
    algoName = None
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict['RNN_BACKPROP_LSTM_GRU']
    avgMaxAccuracy = {}
    avgPrecision = {}
    avgRecall = {}
    avgFMeasure = {}
    accThres = configDict['ACCURACY_THRESHOLD']
    for foldOutputIntentFile in kFoldOutputIntentFiles:
        with open(foldOutputIntentFile,encoding='utf-8') as f:
            for line in f:
                (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure, maxFIndex,hitMeasure) = computeAccuracyForEachEpisode(line, configDict)
                avgMaxAccuracy = appendToDict(avgMaxAccuracy, numEpisodes, accuracy)
                avgPrecision = appendToDict(avgPrecision, numEpisodes, precision)
                avgRecall = appendToDict(avgRecall, numEpisodes, recall)
                avgFMeasure = appendToDict(avgFMeasure, numEpisodes, FMeasure)
    outputEvalQualityFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputEvalQualityShortTermIntent_" + algoName + "_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    try:
        os.remove(outputEvalQualityFileName)
    except OSError:
        pass
    assert len(avgMaxAccuracy) == len(avgPrecision) and len(avgPrecision) == len(avgRecall) and len(avgRecall) == len(avgFMeasure)
    episodeIndex = 0
    for key in avgMaxAccuracy:
        episodeIndex+=1
        outputAccuracy = float(sum(avgMaxAccuracy[key])) / float(len(avgMaxAccuracy[key]))
        outputPrecision = float(sum(avgPrecision[key])) / float(len(avgPrecision[key]))
        outputRecall = float(sum(avgRecall[key])) / float(len(avgRecall[key]))
        outputFMeasure = float(sum(avgFMeasure[key])) / float(len(avgFMeasure[key]))
        outputEvalQualityStr = "#Episodes:" + str(
            episodeIndex) + ";Precision:" + str(outputPrecision) + ";Recall:" + str(outputRecall) + ";FMeasure:" + str(outputFMeasure)+ ";Accuracy:" + str(outputAccuracy)
        ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)
    return outputEvalQualityFileName

def evaluateQualityPredictions(outputIntentFileName, configDict, accThres, algoName):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputDir = None
    if configDict['SINGULARITY_OR_KFOLD'] =='SINGULARITY':
        outputDir = getConfig(configDict['OUTPUT_DIR'])
    elif configDict['SINGULARITY_OR_KFOLD'] =='KFOLD':
        outputDir = getConfig(configDict['KFOLD_OUTPUT_DIR'])
    outputEvalQualityFileName = outputDir + "/OutputEvalQualityShortTermIntent_" + algoName+"_"+configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                                configDict['EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
    try:
        os.remove(outputEvalQualityFileName)
    except OSError:
        pass
    batch_count=0
    hit_count=0
    with open(outputIntentFileName,encoding='utf-8') as f:
        for line in f:
            # 评估性能
            (sessID, queryID, numEpisodes, accuracy, precision, recall, FMeasure, maxFIndex, hitMeasure) = computeAccuracyForEachEpisode(line,
                                                                                                        configDict)
            outputEvalQualityStr = "Session:" + str(sessID) + ";Query:" + str(queryID) + ";#Episodes:" + str(
                numEpisodes) + ";Precision:" + str(precision) + ";Recall:" + str(recall) + ";FMeasure:" + str(FMeasure) +";Accuracy:" + str(
                accuracy)+";MaxFIndex:"+str(maxFIndex)
            batch_count+=1
            hit_count+=hitMeasure
            ti.appendToFile(outputEvalQualityFileName, outputEvalQualityStr)
    print("Hit Rate:", hit_count/batch_count, "Batch Count:", batch_count)
    ti.appendToFile(outputEvalQualityFileName+"_hit", "Hit Rate:"+str(hit_count/batch_count)+";Batch Count:"+str(batch_count))
def avgKFoldTimeAndQualityPlots(kFoldOutputIntentFiles,kFoldEpisodeResponseTimeDicts, avgTrainTimeFN, avgTestTimeFN, algoName, configDict):
    avgTrainTime = readFromPickleFile(avgTrainTimeFN)
    avgTestTime = readFromPickleFile(avgTestTimeFN)
    (outputEvalQualityFileName, avgKFoldTimeDictName) = plotAllFoldQualityTime(kFoldOutputIntentFiles,
                                                                                  kFoldEpisodeResponseTimeDicts,
                                                                                  algoName, configDict)
    outputExcelQuality = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputExcelQuality_" + algoName + "_" + configDict['INTENT_REP'] + "_" + configDict[
                             'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                             'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                             'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(
        configDict['ACCURACY_THRESHOLD']) +"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+ ".xlsx"
    ParseResultsToExcel.parseQualityFileWithoutEpisodeRep(outputEvalQualityFileName, outputExcelQuality, configDict)

    outputExcelTimeEval = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputExcelTime_" + algoName + "_" + configDict[
                              'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                              'TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                          configDict['EPISODE_IN_QUERIES'] +"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+ ".xlsx"
    outputExcelKFoldTimeEval = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputExcelKFoldTime_" + algoName + "_" + configDict[
                                   'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + \
                               configDict['EPISODE_IN_QUERIES'] +"_"+configDict['RNN_INCREMENTAL_OR_FULL_TRAIN']+ ".xlsx"
    # compute avg train time across kfolds and append it to the list
    avgTrainTime.append(float(sum(avgTrainTime)) / float(len(avgTrainTime)))
    # compute avg test time across kfolds and append it to the list
    avgTestTime.append(float(sum(avgTestTime)) / float(len(avgTestTime)))
    avgKFoldTimeDict = readFromPickleFile(avgKFoldTimeDictName)
    ParseResultsToExcel.parseKFoldTimeDict(avgKFoldTimeDict, avgTrainTime, avgTestTime, outputExcelTimeEval,
                                           outputExcelKFoldTimeEval)
    return


def computeAvgFoldTime(kFoldEpisodeResponseTimeDicts, algoName, configDict):
    avgKFoldTimeDictName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/AvgFoldTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + ".pickle"
    avgKFoldTimeDict = {}
    for kFoldEpisodeTimeDict in kFoldEpisodeResponseTimeDicts:
        episodeResponseTime = readFromPickleFile(kFoldEpisodeTimeDict)
        #print "Ep: "+str(episodeResponseTime.keys())
        for episodes in range(1,len(episodeResponseTime)+1):
            if episodes not in avgKFoldTimeDict:
                avgKFoldTimeDict[episodes] = []
            avgKFoldTimeDict[episodes].append(episodeResponseTime[episodes])
    for episodes in range(1,len(avgKFoldTimeDict)+1):
        avgKFoldTimeDict[episodes] = float(sum(avgKFoldTimeDict[episodes]))/float(len(avgKFoldTimeDict[episodes]))
    writeToPickleFile(avgKFoldTimeDictName, avgKFoldTimeDict)
    return avgKFoldTimeDictName

def plotAllFoldQualityTime(kFoldOutputIntentFiles, kFoldEpisodeResponseTimeDicts, algoName, configDict):
    outputEvalQualityFileName = computeAvgFoldAccuracy(kFoldOutputIntentFiles, configDict)
    avgKFoldTimeDict = computeAvgFoldTime(kFoldEpisodeResponseTimeDicts, algoName, configDict)
    return (outputEvalQualityFileName, avgKFoldTimeDict)

def evaluateTimePredictions(episodeResponseTimeDictName, configDict, algoName,split_i):
    assert configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY' or configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD'
    outputDir = None
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = getConfig(configDict['OUTPUT_DIR'])
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = getConfig(configDict['KFOLD_OUTPUT_DIR'])
    outputEvalTimeFileName = outputDir + "/OutputEvalTimeShortTermIntent_" + algoName+"_"+\
                             configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict['TOP_K'] + split_i + \
                             configDict['EPISODE_IN_QUERIES']
    print("evaluateTimePredictions OutputEvalTimeFileName: " + outputEvalTimeFileName)
    try:
        os.remove(outputEvalTimeFileName)
    except OSError:
        pass
    # Simulate or borrow query execution and intent creation to record their times #
    # the following should be configDict['OUTPUT_DIR] and not outputDir because it gets intent creation and queryExec times from the existing pickle files in the outer directory for kfold exp"
    intentCreationTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/IntentCreationTimeDict_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + split_i + configDict[
                                     'EPISODE_IN_QUERIES'] + ".pickle"
    queryExecutionTimeDictName = getConfig(configDict['OUTPUT_DIR']) + "/QueryExecutionTimeDict_" + configDict[
        'INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + split_i + configDict[
                                     'EPISODE_IN_QUERIES'] + ".pickle"

    if os.path.exists(intentCreationTimeDictName) and os.path.exists(queryExecutionTimeDictName):
        episodeQueryExecutionTime = readFromPickleFile(queryExecutionTimeDictName)
        episodeIntentCreationTime = readFromPickleFile(intentCreationTimeDictName)
    else:
        (episodeQueryExecutionTime, episodeIntentCreationTime) = createQueryExecIntentCreationTimes(configDict,split_i)
        writeToPickleFile(queryExecutionTimeDictName, episodeQueryExecutionTime)
        writeToPickleFile(intentCreationTimeDictName, episodeIntentCreationTime)

    episodeResponseTime = readFromPickleFile(episodeResponseTimeDictName)

    print("len(episodeQueryExecutionTime) = " + str(
        len(episodeQueryExecutionTime)) + ", len(episodeIntentCreationTime) = " + str(
        len(episodeIntentCreationTime)) + ", len(episodeResponseTime) = " + str(len(episodeResponseTime)))
    minLen = min(len(episodeQueryExecutionTime), len(episodeIntentCreationTime), len(episodeResponseTime))
  #  assert len(episodeQueryExecutionTime) == len(episodeResponseTime) and len(episodeIntentCreationTime) == len(episodeResponseTime)
    outputEvalTimeStr=None
    for episodes in episodeResponseTime:
        if episodes >= minLen:
            continue
        totalResponseTime = float(episodeIntentCreationTime[episodes]) + float(
            episodeQueryExecutionTime[episodes]) + float(episodeResponseTime[episodes])
        outputEvalTimeStr = "#Episodes:" + str(episodes) + ";QueryExecutionTime(secs):" + str(
            episodeQueryExecutionTime[episodes]) + ";IntentCreationTime(secs):" + str(
            episodeIntentCreationTime[episodes]) + ";IntentPredictionTime(secs):" + str(
            episodeResponseTime[episodes]) + ";TotalResponseTime(secs):" + str(totalResponseTime)
        ti.appendToFile(outputEvalTimeFileName, outputEvalTimeStr)
    print("appended to file:",outputEvalTimeFileName,outputEvalTimeStr)
    return outputEvalTimeFileName

def evaluatePredictions(outputIntentFileName, episodeResponseTimeDictName, configDict,split_i):
    evaluateQualityPredictions(outputIntentFileName, configDict, configDict['ACCURACY_THRESHOLD'], configDict['ALGORITHM'])
    evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'],split_i)
    print("--Completed Quality and Time Evaluation--")
    return

if __name__ == "__main__":
    configDict = parseConfig.parseConfigFile("configFile.txt")
    accThres = float(configDict['ACCURACY_THRESHOLD'])
    algoName = None
    outputDir=None
    outputEvalQualityFileName = None
    if configDict['ALGORITHM'] == 'CF':
        algoName = configDict['ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF']
    elif configDict['ALGORITHM'] == 'RNN':
        algoName = configDict['ALGORITHM'] + "_" + configDict["RNN_BACKPROP_LSTM_GRU"]
    if configDict['SINGULARITY_OR_KFOLD'] == 'SINGULARITY':
        outputDir = getConfig(configDict['OUTPUT_DIR'])
        outputIntentFileName = outputDir + "/OutputFileShortTermIntent_" + configDict[
            'ALGORITHM'] + "_" + configDict['CF_COSINESIM_MF'] + "_" + configDict['INTENT_REP'] + "_" + configDict[
                                   'BIT_OR_WEIGHTED'] + "_TOP_K_" + configDict[
                                   'TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                   'EPISODE_IN_QUERIES'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        evaluateQualityPredictions(outputIntentFileName, configDict, accThres, configDict['ALGORITHM'])
        episodeResponseTimeDictName = outputDir + "/ResponseTimeDict_" + algoName + "_" + \
                                      configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                      configDict['TOP_K'] + "_EPISODE_IN_QUERIES_" + configDict[
                                          'EPISODE_IN_QUERIES'] + ".pickle"
        evaluateTimePredictions(episodeResponseTimeDictName, configDict, configDict['ALGORITHM'])
    elif configDict['SINGULARITY_OR_KFOLD'] == 'KFOLD':
        outputDir = getConfig(configDict['KFOLD_OUTPUT_DIR'])
        outputIntentFileName = configDict[
                                   'KFOLD_OUTPUT_DIR'] + "/OutputFileShortTermIntent_" + algoName + "_" + \
                               configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                               configDict['TOP_K'] + "_ACCURACY_THRESHOLD_" + str(accThres)
        kFoldOutputIntentFiles = []
        kFoldEpisodeResponseTimeDicts = []
        for foldID in range(int(configDict['KFOLD'])):
            outputIntentFileName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/OutputFileShortTermIntent_" + algoName + "_" + \
                                   configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                   configDict['TOP_K'] + "_FOLD_" + str(foldID)
            episodeResponseTimeDictName = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/ResponseTimeDict_" + algoName + "_" + \
                                          configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                                          configDict['TOP_K'] + "_FOLD_" + str(foldID) + ".pickle"
            kFoldOutputIntentFiles.append(outputIntentFileName)
            kFoldEpisodeResponseTimeDicts.append(episodeResponseTimeDictName)
        avgTrainTimeFN = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/avgKFoldTrainTime_" + algoName + "_" + \
                  configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                  configDict['TOP_K']
        avgTestTimeFN = getConfig(configDict['KFOLD_OUTPUT_DIR']) + "/avgKFoldTestTime_" + algoName + "_" + \
                 configDict['INTENT_REP'] + "_" + configDict['BIT_OR_WEIGHTED'] + "_TOP_K_" + \
                 configDict['TOP_K']
        avgKFoldTimeAndQualityPlots(kFoldOutputIntentFiles, kFoldEpisodeResponseTimeDicts, avgTrainTimeFN,
                                       avgTestTimeFN, algoName, configDict)
