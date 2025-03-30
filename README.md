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
# 项目架构说明
## 数据预处理：
第一步：首先，通过[APM_FragmentIntent.py](apm/APM_FragmentIntent.py)脚本将APM数据集转换为SQL片段向量。

     输入配置：[APM_Table_Realtime_FragmentQueries_Keep_configFile.txt](config/APM_Table_Realtime_FragmentQueries_Keep_configFile.txt)
     逻辑：将输入sql查询所属会话进行随机shuffle打乱，并按session id分组将sql合并到一起存放到QUERYSESSIONS参数指定目录。
     核心参数说明：
     输入：
        BIT_FRAGMENT_SEQ_SPLIT_NAME_FORMAT: 指定输入查询及其编码文件路径。
     输出：
        BIT_FRAGMENT_INTENT_SESSIONS：指定输出将session打乱后的sql及其编码向量文件。
        CONCURRENT_QUERY_SESSIONS：指定输出将session打乱后的sql文件。
        BIT_FRAGMENT_TABLE_INTENT_SESSIONS：指定输出将session打乱后的sql及仅仅包含table编码向量文件。
        QUERYSESSIONS: 指定输出查询会话文件路径。将所有原始sql语句按session id分组到一起后输出到文件中。

     
`python APM_FragmentIntent.py -config config/APM_Table_Realtime_FragmentQueries_Keep_configFile.txt`
     
## 模型训练：
# 选择对应的训练模型进行训练:
将上面的输出文件作为下面执行脚本的输入：依然是使用APM_Table_Novel_RNN_singularity_configFile.txt配置文件。

RNN-simple增量训练：`sh scripts/runApmNovelRNNSingularity.sh` 对应执行 LSTM_RNN_Parallel_selOpConst.py
QLearning增量训练：`sh scripts/runApmQLSingularity.sh` 对应执行 QLearning_selOpConst.py

`sh scripts/runBusTrackerNovelRNNSingularity.sh`
`sh scripts/runBusTrackerQLSingularity.sh`

## 按分钟尺度窗口进行query编码和训练
select：维度，指标，每个指标聚合方式
from: 表 
where：滑动时间过滤条件
group by：维度，时间粒度










