#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@File: src/config/config.py
@Description:
    Store environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 系统环境变量优先级 > .env 配置的值
# 从 reasoning_bo 根目录加载 .env（不依赖 CWD）
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=False)


class Config:
    def __init__(self):
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
        self.DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME")
        self.QWQ_API_KEY = os.getenv('QWQ_API_KEY')
        self.QWQ_API_BASE = os.getenv('QWQ_API_BASE')
        self.QWQ_MODEL_NAME = os.getenv('QWQ_MODEL_NAME')
