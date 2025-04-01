import json
import logging
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from collections import defaultdict
from threading import Lock
import logging
from flask import Flask, request, Response
from prometheus_client import (
    start_http_server,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from collections import defaultdict
import time
import threading

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局指标注册缓存（避免重复创建）
metric_registry = {
    'counters': defaultdict(dict),
    'gauges': defaultdict(dict),
    'histograms': defaultdict(dict)
}

# Prometheus 指标类型映射（根据Druid指标名称前缀自动识别）
METRIC_TYPE_MAPPING = {
    'query/time': Gauge,
    'query/count': Gauge,
    'jvm/mem/used': Gauge,
    'jvm/gc/time': Gauge,
    'segment/used': Gauge,
    # 可扩展更多映射规则
}

ALL_METRICS = list(METRIC_TYPE_MAPPING.keys())
ALL_METRICS = ALL_METRICS + ['query/time','query/cpu/time','jvm/mem/used','jvm/gc/mem/used',
                             '/materialized/view/query/hitRate','ingest/events/processed','ingest/events/processedWithError'
                                            'ingestion/kafka/consumer/recordsConsumed','ingest/kafka/partitionLag']
# 单位转换映射（将Druid单位转为Prometheus标准单位）
UNIT_CONVERSION = {
    'milliseconds': lambda x: x / 1000.0,  # 毫秒转秒
    'bytes': lambda x: x,  # 字节无需转换
    'count': lambda x: x
}

# 全局缓存与锁
metric_registry = defaultdict(lambda: defaultdict(dict))
# metric_registry_lock = Lock()
def get_or_create_metric(metric_name, metric_type, labels, help_text='', **kwargs):
    """动态获取或创建Prometheus指标"""
    registry_key = metric_type.__name__.lower() + 's'  # 例如: Counter -> counters
    sorted_labels = sorted(labels.items(), key=lambda x: x[0])
    # 处理不可哈希的标签值（如列表）
    processed_labels = {}
    for key, value in labels.items():
        if isinstance(value, list):
            processed_labels[key] = tuple(value)
        else:
            processed_labels[key] = value
    labels_key = tuple(processed_labels)  # 使用元组作为哈希键

    # with metric_registry_lock:
    # 检查缓存是否存在
    existing_metric = metric_registry[registry_key].get(metric_name, {})
    if existing_metric:
        return existing_metric

    # 创建新指标
    metric_class = metric_type
    if metric_class == Histogram:
        # 直方图需要特殊处理buckets
        metric = metric_class(
            metric_name,
            help_text,
            list(labels.keys()),
            buckets=kwargs.get('buckets', Histogram.DEFAULT_BUCKETS)
        )
    else:
        metric = metric_class(
            metric_name,
            help_text,
            list(labels.keys())
        )

    # 缓存指标
    metric_registry[registry_key][metric_name][labels_key] = metric
    # 注册指标
    try:
        REGISTRY.register(metric);
    except ValueError as e:
        pass
    return metric


def parse_druid_metric(event):
    """解析Druid指标事件，返回标准化结构"""
    metric_name = event.get('metric', '')
    if(metric_name not in ALL_METRICS):
        # logger.warning(f"Unknown metric: {metric_name}")
        return None
    value = event.get('value', 0)
    dimensions = {k: v for k, v in event.items() if k not in ['metric', 'value', 'timestamp']}

    # 自动识别单位并转换
    unit = 'unknown'
    if 'time' in metric_name:
        unit = 'milliseconds'
    elif 'bytes' in metric_name or 'mem' in metric_name:
        unit = 'bytes'
    converted_value = UNIT_CONVERSION.get(unit, lambda x: x)(value)

    # 生成Prometheus指标名称
    prom_metric_name = f'druid_{metric_name.replace("/", "_")}'
    if unit != 'unknown':
        prom_metric_name += f'_{unit}'

    # 确定指标类型
    metric_type = METRIC_TYPE_MAPPING.get(
        metric_name,
        Gauge if isinstance(value, (int, float)) else Counter  # 默认推断
    )
    print("metricName:",metric_name,"newMetricName:",prom_metric_name,"value:",converted_value,'labels', dimensions)
    return {
        'name': prom_metric_name,
        'type': metric_type,
        'value': converted_value,
        'labels': dimensions,
        'help': f'Druid metric: {metric_name}'
    }


@app.route('/druid-metrics', methods=['POST'])
def handle_druid_metrics():
    """处理Druid发送的指标数据"""
    try:
        events = request.get_json()
        if not isinstance(events, list):
            events = [events]

        for event in events:
            parsed = parse_druid_metric(event)
            if parsed is None:
                continue
            metric = get_or_create_metric(
                parsed['name'],
                parsed['type'],
                parsed['labels'],
                help_text=parsed['help']
            )
            # 更新指标值
            if isinstance(metric, Counter):
                metric.labels(**parsed['labels']).inc(parsed['value'])
            elif isinstance(metric, Gauge):
                metric.labels(**parsed['labels']).set(parsed['value'])
            elif isinstance(metric, Histogram):
                metric.labels(**parsed['labels']).observe(parsed['value'])

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error processing metrics: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/metrics')
def prometheus_metrics():
    """暴露Prometheus指标端点"""
    report_metrics_data = generate_latest(REGISTRY)
    print("Report metrics:",report_metrics_data)
    return Response(
        report_metrics_data,
        mimetype=CONTENT_TYPE_LATEST
    )


def background_cleanup():
    """定期清理不再更新的指标（可选）"""
    while True:
        time.sleep(3600)  # 每小时清理一次
        # 实现逻辑：记录最后更新时间，删除长时间未更新的指标


if __name__ == '__main__':
    # 启动Prometheus指标服务器（默认端口8000）
    start_http_server(8000)

    # 启动Flask应用（接收端口19091）
    app.run(host='0.0.0.0', port=19091, threaded=True)