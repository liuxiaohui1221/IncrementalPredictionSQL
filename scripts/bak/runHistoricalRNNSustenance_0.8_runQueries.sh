#!/bin/sh
python analyzeLogs_runQueries.py -config configDir/MINC_Historical_RNN_trainTest_sustenance_0.8_configFile.txt -log ../data/MINC/InputOutput/ClusterRuns/NovelTables-114607-Constants/sustenance/HistoricalRNN/Sustenance_HistoricalRNN_outputSQLLog
# nohup sh scripts/runHistoricalRNNSustenance_0.8_runQueries.sh > ../runHistoricalRNNSustenance_0.8_runQueries.out 2> ../runHistoricalRNNSustenance_0.8_runQueries.err &
