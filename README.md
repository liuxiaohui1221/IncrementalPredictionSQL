# About this repo:

This is a part of our work towards our paper https://dl.acm.org/doi/10.1145/3442338

This codebase was written in Python 2.7 and requires the SQL fragment bit-vectors to be pre-created and fed as input to each of the algorithms here.

* QLearning.py: This code uses tabular version of Q-Learning to predict the SQL fragments
* QLearning_selOpConst.py: This code uses Q-Learning to predict SQL fragment vectors with constant bins
* LSTM_RNN_Parallel.py and LSTM_RNN_Parallel_selOpConst.py contain LSTM and RNN-based implementation. Setting RNN_PREDICT_NOVEL_QUERIES=True and RNN_NOVEL_FIX_SQL_VIOLATIONS=True invokes synthesis-based RNNs described in our paper. Setting these parameters to False invokes the historical RNN baselines.
* CFCosineSim_Parallel.py and CF_SVD_selOpConst.py contain the cosine similarity-based and SVD-based implementation
* configDir contains the config files and scripts folder contains the shell scripts to run the query predictors

CourseWebsite (MINC) dataset is proprietary and cannot be released. The BusTracker dataset is from an earlier work http://www.cs.cmu.edu/~malin199/data/tiramisu-sample/ 

The version of the BusTracker dataset we used is available at
* https://www.dropbox.com/s/umqj1dnc7bhvpcw/BusTracker.zip?dl=0

Pre-created SQL fragment vectors for the BusTracker dataset are available at BusTracker/InputOutput/MincBitFragmentIntentSessions
## 项目架构说明
1.执行流程：
   - 首先，通过[MINC_FragmentIntent.py](MINC_FragmentIntent.py)脚本将MINC或BusTracker数据集转换为SQL片段向量。
    输入：[BusTracker_FragmentQueries_Keep_configFile.txt](configDir%2FBusTracker_FragmentQueries_Keep_configFile.txt)
     输入的数据：data/BusTracker/InputOutput/BakOutput/BusTrackerQueryLog_SPLIT_OUT_xxx
     python MINC_FragmentIntent.py -config configDir/BusTracker_FragmentQueries_Keep_configFile.txt
     输出：输出的SQL片段向量存储在配置文件参数
   - BIT_FRAGMENT_INTENT_SESSIONS(data/BusTracker/InputOutput/MincBitFragmentIntentSessions)、
   - QUERYSESSIONS(data/BusTracker/InputOutput/MincQuerySessions)
   - CONCURRENT_QUERY_SESSIONS(data/BusTracker/InputOutput/MincConcurrentSessions)、
   - BIT_FRAGMENT_TABLE_INTENT_SESSIONS(data/BusTracker/InputOutput/MincBitFragmentTableIntentSessions)
     指定的文件中。
   - 其次，将上面的输出文件作为输入，sh scripts/runApmNovelRNNSingularity.sh执行 LSTM_RNN_Parallel_selOpConst.py
# Running the code:
sh scripts/runBusTrackerNovelRNNSingularity.sh
sh scripts/runBusTrackerCFCosineSimSingularity.sh
sh scripts/runBusTrackerQLSingularity.sh

# 改造说明

## 按分钟尺度窗口进行query编码和训练