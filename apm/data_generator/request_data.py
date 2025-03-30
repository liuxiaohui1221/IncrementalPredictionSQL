from random import randint

from apm.tool import generate_normal_random_int

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

low_cardinality_str_fields = [
    'type', 'group', 'appid', 'appsysid', 'agent',
    'path', 'method', 'root_appid', 'pappid', 'pappsysid',
    'pagent', 'pagent_ip', 'uevent_model', 'uevent_id', 'user_id', 'session_id',
    'host', 'ip_addr', 'province', 'city', 'page_id', 'page_group', 'tag',
]
low_cardinality_int_fields = [
'biz', 'fail', 'httperr', 'neterr', 'err', 'tolerated', 'frustrated','exception','service_type','papp_type'
]
err_fields = ['err_4xx', 'err_5xx']
err4_status_code = [400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 422,
                   423, 424, 425, 426, 429, 431, 451]
err5_status_code = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]
low_cardinality = 500
low_cardinality_min = 1
low_cardinality_max = 1000

import hashlib
import random
import string

def generate_hash(input_str: str) -> str:
    """生成MD5哈希并补全前导零到32位"""
    md5_hash = hashlib.md5(input_str.encode()).hexdigest()
    # MD5固定生成32位十六进制字符串，此步骤为保证与Java逻辑一致
    return md5_hash.zfill(32)[-32:]  # 双重保证长度

def generate_random_string(length: int, prefix: str = None) -> str:
    """生成随机字符串（包含字母数字）"""
    # 定义字符集（62个可打印ASCII字符）
    characters = string.ascii_letters + string.digits
    # 生成随机字符列表
    rand_chars = [random.choice(characters) for _ in range(length)]
    content = ''.join(rand_chars)
    # 添加前缀（Python风格的空值判断）
    if prefix is not None:
        content = prefix + content
    return content
# 哈希函数测试
# print(generate_hash("hello"))  # 5d41402abc4b2a76b9719d911017c592
# print(len(generate_hash("test")))  # 32
#
# # 随机字符串测试
# print(generate_random_string(8))              # 类似"x7gA2vZf"
# print(generate_random_string(5, "appid_"))     # 类似"USER_9k3Gh"

def randomValueStr(prefix,cardinality):
    return prefix + str(randint(0,cardinality-1))
def randomValueInt(min,max):
    return randint(min,max)
def chooseFieldRandomValue(field,condition=-1):
    if field in low_cardinality_str_fields:
        return randomValueStr(field,low_cardinality)
    elif field in low_cardinality_int_fields:
        return generate_normal_random_int(min_value=low_cardinality_min, max_value=low_cardinality_max)
    elif field in err_fields:
        if condition<500 and condition>=400 and field == 'err_4xx':
            return err4_status_code[randomValueInt(0,len(err4_status_code)-1)]
        elif condition>=500 and field == 'err_5xx':
            return err5_status_code[randomValueInt(0,len(err5_status_code)-1)]
        else:
            return 0
    elif field == 'is_model':
        return randomValueInt(0,1)
    else:
        return None

