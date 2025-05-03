# {"feed":"metrics","metric":"coordinator/time",
# "service":"druid/coordinator",
# "host":"localhost:8081",
# "value":0,
# "timestamp":"2025-04-07T14:48:56.021Z"}

# 示例输入
fields = [
    'dur','biz', 'fail', 'httperr', 'neterr', 'err', 'tolerated',
          'frustrated','exception','err_4xx', 'err_5xx'
    ]
# # 生成Druid任务的JSON指标格式
druid_metrics = generate_druid_metrics(fields)
# 打印输出
import json
print(json.dumps(druid_metrics, indent=4))