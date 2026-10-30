"""
光子收费配置文件
"""

import os
from dptb_pilot.core.photon_service import PhotonChargeConfig
# The application entry point loads .env before importing this module.
# Do not search for another .env here: that can silently select a different
# deployment's configuration when the source and startup directories differ.
PHOTON_CONFIG = PhotonChargeConfig(
    # 应用的 SKU ID（需要从玻尔平台获取）
    sku_id=int(os.getenv("PHOTON_SKU_ID", "12345")),  # 请替换为实际的 SKU ID

    # 开发者 AccessKey（用于调试，生产环境应该使用用户的 AccessKey）
    dev_access_key=os.getenv("PHOTON_DEV_ACCESS_KEY"),  # 可选，用于调试

    # 客户端名称（默认值，实际使用时会从 cookie 中获取 clientName）
    client_name=os.getenv("PHOTON_CLIENT_NAME", "DeepTBPilot"),

    # 收费规则配置
    min_charge=int(os.getenv("PHOTON_MIN_CHARGE", "1")),  # 最小收费光子数
)

# 收费开关配置
CHARGING_ENABLED = os.getenv("CHARGING_ENABLED", "false").lower() == "true"

# 免费额度配置（每日免费 token 数量）
FREE_TOKEN_QUOTA = int(os.getenv("FREE_TOKEN_QUOTA", "1000"))
