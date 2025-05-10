import re


def extract_unique_values(file_path, pattern, encoding='utf-8'):
    """
    按行读取文件，通过正则表达式提取内容，返回去重后的结果列表

    参数：
    file_path : str      - 文件路径
    pattern   : str      - 正则表达式模式
    encoding  : str      - 文件编码（默认utf-8）

    返回：
    list - 包含所有唯一匹配值的列表，保持首次出现顺序

    示例：
    //>>> extract_unique_values('file_path', r'\b\d{3}-\d{4}\b')  # 匹配电话号码
    ['123-4567', '888-1234', ...]
    """
    seen = set()
    result = []

    try:
        compiled_re = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"无效的正则表达式: {e}") from None

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                matches = compiled_re.findall(line)
                for match in matches:
                    if isinstance(match, tuple):  # 处理正则分组情况
                        match = match[0]  # 取第一个分组
                    if match not in seen:
                        seen.add(match)
                        result.append(match)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except UnicodeDecodeError:
        raise ValueError(f"解码失败，请尝试指定正确的文件编码")

    return result


# 使用示例
if __name__ == "__main__":
    """ 从字符串中提取 appsysid = 'xxx' 格式中的值 """
    appsysid_pattern = r"""appsysid\s*=\s*'([^']*)'"""
    group_pattern = r"group\s*=\s*'([^']*)'"
    # 示例：提取所有邮箱地址
    values = extract_unique_values(
        file_path="0318_ApmQuerys.tsv",
        pattern=group_pattern,
        encoding='utf-8'
    )
    print("唯一匹配结果：\n", values)
