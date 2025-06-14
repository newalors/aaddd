#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 标准库导入
import os
import sys
import time
import datetime
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Optional, Tuple

# 第三方库导入
import numpy as np
import pandas as pd

# 添加 xtquant 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 本地模块导入
from xtquant import xtdata
from xtdata_manager import get_stock_data

# 定义数据目录常量
XT_BASE_DATA_DIR = r"G:\迅投极速交易终端 睿智融科版\datadir"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(XT_BASE_DATA_DIR, "ma60_calculator.log")),
        logging.StreamHandler()
    ]
)

# 全局缓存
ma60_cache = {}  # 缓存MA60结果
CACHE_FILE = os.path.join(XT_BASE_DATA_DIR, 'ma60_results.json')  # 缓存文件放在数据目录下

def convert_dataframe_to_dict(df_data):
    """将DataFrame格式的数据转换为字典格式"""
    if df_data is None:
        return None
        
    if isinstance(df_data, dict):
        return df_data
        
    try:
        if isinstance(df_data, pd.DataFrame):
            result = {}
            for column in df_data.columns:
                result[column] = df_data[column].values
            return result
        return df_data
    except Exception as e:
        logging.error(f"数据格式转换失败: {e}")
        return None

def test_formula(code: str, xtdata_module, verbose: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """计算MA60 - 简化版
    
    Args:
        code: 股票代码
        xtdata_module: xtdata模块
        verbose: 是否输出详细日志
        
    Returns:
        (ma60, error): 计算结果和错误信息
    """
    try:
        # 检查缓存
        if code in ma60_cache:
            return ma60_cache[code], None
            
        # 直接获取历史数据 - 简化为一次调用
        fields = ['close']
        data = xtdata.get_local_data(fields, [code], '1d', count=60, data_dir=XT_BASE_DATA_DIR)
        
        # 处理数据
        if not data or code not in data or data[code].empty:
            logging.warning(f"{code} 无法获取历史数据")
            return None, "无历史数据"
            
        # 获取收盘价
        df = data[code]
        if 'close' not in df.columns:
            logging.warning(f"{code} 数据中没有close列")
            return None, "数据中没有close列"
            
        closes = df['close'].values
        
        # 过滤无效值
        valid_closes = [float(x) for x in closes if x is not None and x > 0]
        
        # 检查有效数据点数量
        if not valid_closes:
            logging.warning(f"{code} 没有有效的收盘价数据")
            return None, "无有效收盘价数据"
        
        # 数据点过少时记录警告，但仍然计算
        if len(valid_closes) < 60:
            logging.info(f"{code} 有效数据少于60天，只有{len(valid_closes)}天")
            if len(valid_closes) < 10:  # 如果少于10天，认为不足以计算有意义的MA值
                logging.warning(f"{code} 有效数据少于10天，不计算MA60")
                return None, "有效数据少于10天"
        
        # 计算MA60
        ma60 = sum(valid_closes) / len(valid_closes)
        logging.info(f"{code} MA60计算成功: {ma60:.2f} (基于{len(valid_closes)}天数据)")
        
        # 缓存结果
        ma60_cache[code] = ma60
        
        return ma60, None
    except Exception as e:
        logging.error(f"{code} 计算MA60出错: {e}")
        return None, f"{str(e)}"

def calculate_ma60_batch(stock_list: list, max_workers: int = 20) -> Dict[str, float]:
    """
    批量计算多只股票的MA60

    Args:
        stock_list: 股票代码列表
        max_workers: 最大线程数

    Returns:
        dict: {股票代码: MA60值}
    """
    # 记录传入的股票列表
    logging.info(f"开始计算 {len(stock_list)} 只股票的MA60...")

    results = {}
    success_count = 0
    error_count = 0

    for code in stock_list:
        ma60, error = test_formula(code, xtdata)
        if error is None and ma60 is not None:
            results[code] = ma60
            success_count += 1
        else:
            # 如果无法计算MA60，记录错误并返回None
            # 不使用默认值0，实事求是
            results[code] = None
            error_count += 1
            logging.warning(f"股票 {code} 无法计算MA60: {error}")

    # 计算成功率
    success_rate = success_count / len(stock_list) * 100 if stock_list else 0
    logging.info(f"MA60计算完成，成功率: {success_rate:.2f}% (成功: {success_count}, 失败: {error_count})")

    # 如果有错误，输出部分错误样例
    if error_count > 0:
        error_sample = [code for code, ma60 in results.items() if ma60 is None][:3]
        if error_sample:
            error_msg = "；".join(error_sample)
            logging.info(f"部分无法计算MA60的股票样例: {error_msg}...")

    return results
