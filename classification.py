import xgboost as xgb
import pandas as pd
import utils

def get_model(model_name):
    """加载模型"""

    if model_name == 'xgboost':
        model = xgb.Booster()
        model.load_model('./saved_models/best_xgboost_20250517.xgb')
    else:
        model = xgb.Booster()
        model.load_model('./saved_models/best_xgboost_20250517.xgb')
    
    return model


def parse_position_sequence(model_name, position):
    """解析轨迹序列，作为标准模型输入"""

    if model_name == 'xgboost':
        # 将变长位置序列转化成标准长度的模型输入
        position_parsed = utils.interpolate_position_sequence(position, M=600)
        position_parsed = xgb.DMatrix(pd.DataFrame(position_parsed))
    else:
        position_parsed = utils.interpolate_position_sequence(position, M=600)
        position_parsed = xgb.DMatrix(pd.DataFrame(position_parsed))
    
    return position_parsed