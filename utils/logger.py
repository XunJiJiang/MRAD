

import logging
import os

# ===================== get_logger: 创建同时输出到文件和终端的日志记录器 =====================
# 参数 save_path: 日志文件保存目录；返回配置好的 logger 实例
def get_logger(save_path):
    # 如果保存路径不存在，则创建目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    # 清空 root logger 已有的 handler，避免重复输出
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    # 创建一个名为 'test' 的 logger，设置日志格式和级别
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    # 添加文件 handler：将日志写入 log.txt
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 添加控制台 handler：同时将日志输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
