import hashlib
import random
import string


def generate_random_string():
    """生成随机的32位字符串"""
    characters = string.ascii_letters + string.digits  # 包含字母和数字
    return ''.join(random.choice(characters) for _ in range(32))


import hashlib
import json


def hash_dict(data_dict):
    """对字典进行哈希"""
    # 将字典转换为JSON字符串（确保顺序一致）
    dict_str = json.dumps(data_dict, sort_keys=True, ensure_ascii=False)

    # 创建哈希对象
    hash_obj = hashlib.sha256()
    hash_obj.update(dict_str.encode('utf-8'))

    return hash_obj.hexdigest()