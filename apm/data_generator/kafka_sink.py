import time
import json
from datetime import datetime, timedelta
import random

from faker import Faker
from kafka import KafkaProducer
import numpy as np
from threading import Thread

fake = Faker('zh_CN')
def generate_timestamp(realtime_percent=0.8, recent_percent=0.15,recent_days=7):
    # 当前时间
    now = datetime.now()

    # 随机生成时间偏移
    probability = random.random()  # 生成 0-1 之间的随机数

    if probability < realtime_percent:  # 80% 概率：最近 10 分钟
        delta = timedelta(minutes=random.randint(0, 10))
    elif probability < realtime_percent+recent_percent:  # 15% 概率：最近 1 天
        delta = timedelta(days=1, minutes=random.randint(0, 1440))  # 1 天 = 1440 分钟
    else:  # 5% 概率：最近 1 周
        delta = timedelta(days=recent_days, minutes=random.randint(0, 10080))  # 1 周 = 10080 分钟

    # 生成时间戳
    ts = now - delta
    return ts
# 全局参数
# Kafka配置
producer = KafkaProducer(
    bootstrap_servers=['192.168.86.9:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
kafka_topic="request"
total_batch = 10_000
base_rate = 100  # 基础速率（条/分钟）
APPSYS_IDS = [f"APPSYS_{i:02d}" for i in range(10)]
APP_IDS = {sys_id: [f"{sys_id}_APP_{i:03d}" for i in range(50)] for sys_id in APPSYS_IDS}
SERVICE_IDS = {app_id: [f"{app_id}_SVC_{i:03d}" for i in range(100)] for sys_id in APPSYS_IDS for app_id in
               APP_IDS[sys_id]}

def generate_ods_request():
    """生成请求明细数据（含时间模式控制）"""
    count=0
    total=0
    while count<total_batch:
        count+=1
        # 模拟工作日高峰（9:00-18:00）
        current_hour = datetime.now().hour
        is_workday = datetime.now().weekday() < 5


        # 速率调整逻辑
        if is_workday and 9 <= current_hour < 18:
            rate = random.randint(base_rate, base_rate * 2)  # 工作日高峰
        else:
            rate = random.randint(base_rate - 10, base_rate + 10)
        print("已发送条数：",total,"当前速率（条/分钟）：",rate)
        random_time = generate_timestamp(realtime_percent=0.8, recent_percent=0.2,recent_days=5)

        for _ in range(rate // 60):  # 按秒均匀分布
            sys_id = np.random.choice(APPSYS_IDS, p=[0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
            app_id = np.random.choice(APP_IDS[sys_id])
            # 时间乱序
            # random_time += timedelta(minutes=random.randint(1, 5))
            record = {
                # "ts":  datetime.now().isoformat(),
                "ts": datetime.now().isoformat(),
                "appsysid": sys_id,
                "appid": app_id,
                "service_type": np.random.choice(["Java", "Go", "Python"], p=[0.6, 0.3, 0.1]),
                "status_code": int(np.random.choice([200, 400, 500], p=[0.9, 0.05, 0.05])),
                "dur": np.random.exponential(scale=50),  # 请求耗时（指数分布）
                # 其他字段生成逻辑...
            }
            producer.send(kafka_topic, record)
            total+=1
            # print("发送数据:", record)
            time.sleep(60 / rate)  # 控制速率


#
# 启动数据生成线程
Thread(target=generate_ods_request).start()