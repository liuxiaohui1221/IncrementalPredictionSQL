import re
import random
import argparse
from typing import Dict, Any
import time
import json
from datetime import datetime, timedelta

from faker import Faker
from kafka import KafkaProducer
import numpy as np
from threading import Thread

from apm.data_generator.request_data import chooseFieldRandomValue, generate_random_string
from apm.tool import getISOFormatTime


def get_field_all_values(cardinality,prefix,val_lenth=20):
    return [generate_random_string(val_lenth, prefix) for i in range(cardinality)]
def parse_log_line(line: str) -> Dict[str, Any]:
    """解析单行日志，提取指定字段并用随机值填充缺失字段。"""
    pattern = re.compile(
        r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)" "([^"]*)"$'
    )
    match = pattern.match(line.strip())
    if not match:
        return {}

    # 提取已知字段
    extracted = {
        'ts': getISOFormatTime(match.group(2)),
        'method': match.group(3),
        'path': match.group(4),
        'status_code': int(match.group(6)),
        'dur': int(match.group(7)),
        'agent': match.group(9),
        'ip_addr': match.group(1),
    }

    # 所有需要处理的字段列表
    all_fields = [
        'ts', 'type', 'group', 'appid', 'appsysid', 'agent', 'service_type',
        'path', 'method', 'root_appid', 'pappid', 'pappsysid', 'papp_type',
        'pagent', 'pagent_ip', 'uevent_model', 'uevent_id', 'user_id',
        'session_id', 'host', 'ip_addr', 'province', 'city', 'page_id',
        'page_group', 'status', 'err_4xx', 'err_5xx', 'status_code', 'tag',
        'code', 'is_model', 'exception', 'biz', 'fail', 'httperr', 'neterr',
        'err', 'tolerated', 'frustrated', 'dur'
    ]
    appid_values=get_field_all_values(500,'appid_',val_lenth=50)
    appsysid_values = get_field_all_values(100, 'appsysid_',val_lenth=20)
    result = {}
    for field in all_fields:
        if field in extracted:
            result[field] = extracted[field]
        else:
            condition=-1
            if extracted['status_code']!=200:
                condition=extracted['status_code']
            # 生成基于基数的随机整数
            if field=='appid':
                result[field] = appid_values[random.randint(0, len(appid_values) - 1)]
            elif field=='appsysid':
                result[field] = appsysid_values[random.randint(0, len(appsysid_values) - 1)]
            else:
                result[field] = chooseFieldRandomValue(field,condition=condition)

    return result


def main():
    """主函数，处理命令行参数并解析日志文件。"""
    parser = argparse.ArgumentParser(description='解析日志文件并提取字段，缺失字段用随机值填充。')
    parser.add_argument('input_file', help='日志文件路径')
    # parser.add_argument('--base', type=int, default=100, help='随机值的基数（默认：100）')
    args = parser.parse_args()
    # Kafka配置
    producer = KafkaProducer(
        bootstrap_servers=['192.168.86.9:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    kafka_topic="dwm_request"
    total,rate=0,0.001
    with open(args.input_file, 'r') as file:
        for line in file:
            record = parse_log_line(line)
            producer.send(kafka_topic, record)
            print("发送数据:", record)
            total += 1
            # print("发送数据:", record)
            if total%1000==0:
                time.sleep(rate)  # 控制速率sec


if __name__ == '__main__':
    main()