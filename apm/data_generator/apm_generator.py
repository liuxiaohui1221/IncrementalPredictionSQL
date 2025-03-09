import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker('zh_CN')

# 全局配置
APPSYS_IDS = [f"APPSYS_{i:02d}" for i in range(10)]
APP_IDS = {sys_id: [f"{sys_id}_APP_{i:03d}" for i in range(50)] for sys_id in APPSYS_IDS}
SERVICE_IDS = {app_id: [f"{app_id}_SVC_{i:03d}" for i in range(100)] for sys_id in APPSYS_IDS for app_id in APP_IDS[sys_id]}

TIME_RANGE = {
    'start': datetime(2024, 1, 1),
    'end': datetime(2024, 1, 7),
    'interval': 'min'  # 数据生成粒度（min/hour）
}


def generate_dwm_request(num_rows=1000000,max_rows_in_windows=1000):
    data = []
    current_time = TIME_RANGE['start']

    while len(data) < num_rows:
        # 基础信息
        sys_id = random.choice(APPSYS_IDS)
        app_id = random.choice(APP_IDS[sys_id])
        service_id = random.choice(SERVICE_IDS[app_id])

        # 构建上下游关联
        p_sys_id = random.choice(APPSYS_IDS) if random.random() < 0.3 else sys_id
        p_app_id = random.choice(APP_IDS[p_sys_id]) if p_sys_id else None

        # 错误模拟
        status_code = random.choices(
            [200, 302, 400, 401, 404, 500],
            weights=[85, 5, 3, 2, 3, 2]
        )[0]

        record = {
            "ts": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": random.choice(['HTTP', 'RPC', 'MQ']),
            "group": f"GROUP_{random.randint(1, 10)}",
            "appid": app_id,
            "appsysid": sys_id,
            "agent": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            "service_type": random.choice(['Java', 'Go', 'Node.js']),
            "path": f"/api/{fake.uri_path()}",
            "method": random.choice(['GET', 'POST', 'PUT']),
            "root_appid": app_id if random.random() < 0.8 else None,
            "pappid": p_app_id,
            "pappsysid": p_sys_id,
            "pagent": fake.ipv4(),
            "province": fake.province(),
            "city": fake.city(),
            "status_code": status_code,
            "err_4xx": 1 if 400 <= status_code < 500 else 0,
            "err_5xx": 1 if status_code >= 500 else 0,
            "dur": random.expovariate(1 / 150)  # 指数分布模拟延迟
        }
        data.append(record)

        # 时间推进
        current_time += timedelta(minutes=random.randint(1, 5))
        if current_time > TIME_RANGE['end']:
            break

    return pd.DataFrame(data)


def generate_error_table(table_type, num_rows=10000):
    error_types = {
        'crash': ['OOM', 'SegFault', 'NativeCrash'],
        'freeze': ['UIBlock', 'Deadlock', 'LongGC'],
        'exception': ['NullPointer', 'Timeout', 'DBConnectionFailed']
    }

    data = []
    for _ in range(num_rows):
        sys_id = random.choice(APPSYS_IDS)
        record = {
            "ts": fake.date_time_between(TIME_RANGE['start'], TIME_RANGE['end']).isoformat(),
            "group": f"GROUP_{random.randint(1, 10)}",
            "appsysid": sys_id,
            "appid": random.choice(APP_IDS[sys_id]),
            "error": random.choice(error_types[table_type]),
            "type": random.choice(['FATAL', 'WARNING', 'ERROR']),
            "desc": fake.sentence(),
            "ip": fake.ipv4(),
            "province": fake.province(),
            "city": fake.city(),
            "os": random.choice(['Android', 'iOS', 'Windows']),
            "app_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}"
        }
        if table_type == 'exception':
            record.update({
                "method": f"{fake.word()}.{fake.word()}()",
                "class": f"com.{fake.word()}.{fake.word()}"
            })
        data.append(record)
    return pd.DataFrame(data)


def generate_dwm_topo(num_edges=50000):
    data = []
    for _ in range(num_edges):
        # 随机选择两个服务建立调用关系
        from_sys = random.choice(APPSYS_IDS)
        from_app = random.choice(APP_IDS[from_sys])
        to_sys = random.choice(APPSYS_IDS)
        to_app = random.choice(APP_IDS[to_sys])

        data.append({
            "ts": fake.date_time_between(TIME_RANGE['start'], TIME_RANGE['end']).isoformat(),
            "req_hash": fake.sha256(),
            "status": random.choice(['SUCCESS', 'FAILED', 'TIMEOUT']),
            "min_dur": random.uniform(10, 100),
            "max_dur": random.uniform(100, 5000),
            "sum_dur": random.uniform(1000, 50000),
            "from_appid": from_app,
            "from_appsysid": from_sys,
            "to_appid": to_app,
            "to_appsysid": to_sys,
            "from_agent": fake.ipv4(),
            "to_agent": fake.ipv4()
        })
    return pd.DataFrame(data)


def inject_patterns(df):
    """注入周期性/突发性模式"""
    ts_date=pd.to_datetime(df['ts'])
    # 每天上午10点流量高峰
    peak_mask = (ts_date.dt.hour == 10)
    # 临时设置显示所有行
    # with pd.option_context('display.max_rows', None):
    #     print(pd.to_datetime(df['ts']).dt.hour == 10)

    df.loc[peak_mask, 'dur'] *= 1.5  # 延迟增加

    # 每周五下午错误率升高
    friday_mask = (ts_date.dt.weekday == 4) & (ts_date.dt.hour >= 14)
    # df.loc[friday_mask, 'err_5xx'] = np.where(
    #     np.random.rand(len(friday_mask)) < 0.2, 1, df.loc[friday_mask, 'err_5xx'].values
    # )

    # 定义一个函数来处理每行
    def update_err_5xx(row):
        if row.name in df[friday_mask].index and np.random.rand() < 0.2:
            return 1
        return row['err_5xx']

    # 应用函数
    df['err_5xx'] = df.apply(update_err_5xx, axis=1)
    return df


def generate_related_data():
    # 生成关联的request和exception数据
    request_df = generate_dwm_request()
    exception_df = generate_error_table('exception')

    # 为部分request生成关联exception
    related_exceptions = exception_df.sample(n=int(len(request_df) * 0.05))
    related_exceptions['trace_id'] = request_df.sample(n=len(related_exceptions))['req_hash'].values
    return request_df, exception_df
def export_to_druid_format(df, table_name):
    """转换为Druid需要的JSON格式"""
    df['__time'] = pd.to_datetime(df['ts']).astype(np.int64) // 10**6  # Druid时间戳格式
    df.to_json(
        f"{table_name}.json",
        orient='records',
        lines=True
    )

# 示例调用
request_df = generate_dwm_request()
request_df = inject_patterns(request_df)
export_to_druid_format(request_df, 'dwm_request')
