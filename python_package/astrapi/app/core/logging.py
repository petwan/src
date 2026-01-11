"""
日志配置模块

使用 Loguru 作为日志库，支持不同级别的日志输出和格式化。

使用示例:
    from loguru import logger

    # 记录不同级别的日志
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
"""

from loguru import logger

from app.core.config import settings


# 定义日志格式
LOG_FORMAT_DEBUG = "<level>{level}</level>: {message}: {name}: {function}:{line}"
# DEBUG模式格式: 级别: 消息: 模块名: 函数名:行号


# 日志级别映射字典
LOG_LEVELS = {
    "INFO": "INFO",
    "WARN": "WARNING",  # WARN 映射到 WARNING
    "ERROR": "ERROR",
    "DEBUG": "DEBUG",
}


def configure_logging() -> None:
    """
    配置 Loguru 日志系统

    根据配置文件中的 LOG_LEVEL 设置日志级别和输出格式。

    支持的日志级别:
        - DEBUG: 调试信息，输出详细信息（模块名、函数名、行号）
        - INFO: 一般信息
        - WARNING: 警告信息
        - ERROR: 错误信息

    示例:
        configure_logging()
        logger.info("应用启动")
        logger.error("发生错误")
    """
    # 获取配置的日志级别并转大写
    log_level = settings.LOG_LEVEL.upper()

    # 验证日志级别是否有效
    if log_level not in LOG_LEVELS:
        # 无效的日志级别，使用 ERROR 作为默认值
        logger.remove()
        logger.add(lambda msg: None, level="ERROR")
        return

    # 移除 Loguru 默认的 handler
    logger.remove()

    # 获取实际的日志级别
    level = LOG_LEVELS.get(log_level, "ERROR")

    if log_level == "DEBUG":
        # DEBUG 模式使用详细格式，包含模块、函数、行号信息
        logger.add(
            lambda msg: print(msg, end=""),  # 输出到控制台
            level=level,                      # 日志级别
            format=LOG_FORMAT_DEBUG,          # 日志格式
        )
    else:
        # 其他模式使用简单格式，只显示级别和消息
        logger.add(
            lambda msg: print(msg, end=""),  # 输出到控制台
            level=level,                      # 日志级别
            format="<level>{level}</level>: {message}",  # 简单格式
        )
