#!/bin/sh 
python analyzeLogs_runQueries.py -config configDir/MINC_Novel_RNN_trainTest_sustenance_0.8_configFile.txt -log ../data/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/NovelRNN-LAST-3/Sustenance_NovelRNN_outputSQLLog
# nohup sh scripts/runNovelRNNSustenance_0.8_Last3_runQueries.sh > ../runNovelRNNSustenance_0.8_Last3_runQueries.out 2> ../runNovelRNNSustenance_0.8_Last3_runQueries.err &
