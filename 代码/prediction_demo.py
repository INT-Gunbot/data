"""
Wordle玩家表现预测 - 快速演示脚本
功能：加载已训练模型，对新历史数据进行预测并输出解读
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import warnings

warnings.filterwarnings('ignore')


def load_prediction_components(model_paths=None, fe_path='wordle_feature_engineering.pkl'):
    """
    加载预测所需的模型和特征工程对象
    """
    # 默认模型路径（需与训练时保存的路径一致）
    if model_paths is None:
        model_paths = {
            'lstm_reg': 'lstm_regression_model.h5',
            'lstm_cls': 'lstm_classification_model.h5',
            'bilstm_att_reg': 'bilstm_attention_regression_model.h5',
            'transformer_reg': 'transformer_regression_model.h5'
        }

    # 加载特征工程对象（含标准化器）
    try:
        with open(fe_path, 'rb') as f:
            fe = pickle.load(f)
        print("✅ 特征工程对象加载成功")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 未找到特征工程文件: {fe_path}，请先运行wordle_prediction_project.py生成")

    # 加载模型
    models = {}
    for model_name, path in model_paths.items():
        try:
            models[model_name] = load_model(path)
            print(f"✅ {model_name} 模型加载成功")
        except FileNotFoundError:
            print(f"⚠️  未找到{model_name}模型文件: {path}，将跳过该模型预测")

    return models, fe


def prepare_prediction_data(historical_data, fe, feature_columns):
    """
    准备预测数据（标准化+生成时间序列）
    historical_data: DataFrame，需包含7天的历史数据
    fe: 特征工程对象（含训练好的标准化器）
    feature_columns: 特征列列表
    """
    # 检查数据是否足够（需7天）
    if len(historical_data) != 7:
        raise ValueError(f"❌ 历史数据需包含7天，当前仅{len(historical_data)}天")

    # 复制数据避免修改原始数据
    data_copy = historical_data.copy()

    # 标准化特征（使用训练时的标准化器）
    for col in feature_columns:
        if col not in fe.scalers:
            raise KeyError(f"❌ 特征 {col} 不在标准化器中，请检查特征列是否正确")
        scaler = fe.scalers[col]
        data_copy[f'norm_{col}'] = scaler.transform(data_copy[[col]])

    # 生成时间序列（形状：(1, 7, num_features)）
    norm_feature_cols = [f'norm_{col}' for col in feature_columns]
    sequence = data_copy[norm_feature_cols].values.reshape(1, 7, len(feature_columns))

    return sequence


def predict_player_performance(historical_data, models, fe):
    """
    预测玩家表现：平均尝试次数（回归）+ 高成功率（分类）
    """
    # 定义特征列（需与训练时一致）
    feature_columns = [
        'Average_tries', 'Success_rate', 'Hard_mode_ratio', 'Tries_std',
        '1_try_pct', '2_tries_pct', '3_tries_pct', '4_tries_pct',
        '5_tries_pct', '6_tries_pct', '7_plus_tries_pct',
        'Reported_results', 'Hard_mode_count'
    ]

    # 检查历史数据是否包含所有必要列
    missing_cols = [col for col in feature_columns if col not in historical_data.columns]
    if missing_cols:
        raise ValueError(f"❌ 历史数据缺少必要列: {', '.join(missing_cols)}")

    # 准备预测数据
    sequence = prepare_prediction_data(historical_data, fe, feature_columns)

    # 存储预测结果
    predictions = {}

    # 1. 回归任务：预测平均尝试次数
    regression_models = ['lstm_reg', 'bilstm_att_reg', 'transformer_reg']
    for model_name in regression_models:
        if model_name in models:
            pred = models[model_name].predict(sequence, verbose=0)[0][0]
            predictions[model_name] = round(pred, 2)

    # 2. 分类任务：预测高成功率（概率）
    if 'lstm_cls' in models:
        pred_prob = models['lstm_cls'].predict(sequence, verbose=0)[0][0]
        predictions['high_success_prob'] = round(pred_prob, 4)
        predictions['high_success'] = 1 if pred_prob > 0.5 else 0

    # 计算集成预测（回归模型的平均值）
    regression_preds = [pred for model_name, pred in predictions.items()
                        if model_name in regression_models and model_name in models]
    if regression_preds:
        predictions['ensemble_avg_tries'] = round(np.mean(regression_preds), 2)

    return predictions


def print_prediction_report(predictions, fe):
    """
    打印预测报告（格式化输出）
    """
    print("\\n" + "=" * 80)
    print("Wordle玩家表现预测报告")
    print("=" * 80)

    # 1. 平均尝试次数预测（回归结果）
    print("\\n📊 平均尝试次数预测（越低表示玩家表现越好）:")
    regression_models = {
        'lstm_reg': '基础LSTM模型',
        'bilstm_att_reg': 'BiLSTM+Attention模型（推荐）',
        'transformer_reg': 'Transformer模型',
        'ensemble_avg_tries': '集成预测（多模型平均）'
    }

    for model_key, model_name in regression_models.items():
        if model_key in predictions:
            print(f"   - {model_name}: {predictions[model_key]} 次")

    # 2. 高成功率预测（分类结果）
    if 'high_success_prob' in predictions:
        success_threshold = fe.df['Success_rate'].mean() * 100  # 高成功率的阈值（训练数据的平均值）
        pred_prob = predictions['high_success_prob'] * 100
        pred_label = "是" if predictions['high_success'] == 1 else "否"

        print(f"\\n🎯 高成功率预测（阈值：≥{success_threshold:.1f}%）:")
        print(f"   - 预测结果: {pred_label}")
        print(f"   - 置信度: {pred_prob:.1f}%")

    # 3. 结果解读
    print("\\n📝 预测结果解读:")
    if 'ensemble_avg_tries' in predictions:
        avg_tries = predictions['ensemble_avg_tries']
        if avg_tries < 3.5:
            print(f"   - 平均尝试次数{avg_tries}，属于较低水平，预测玩家表现优秀（或单词难度低）")
        elif avg_tries < 4.5:
            print(f"   - 平均尝试次数{avg_tries}，属于中等水平，预测玩家表现正常")
        else:
            print(f"   - 平均尝试次数{avg_tries}，属于较高水平，预测玩家表现一般（或单词难度高）")

    if 'high_success' in predictions and predictions['high_success'] == 1:
        print("   - 高成功率置信度高，预测大部分玩家能在6次内猜对单词")
    elif 'high_success' in predictions:
        print("   - 高成功率置信度低，预测部分玩家可能无法在6次内猜对单词")

    print("\\n" + "=" * 80)


# ==============================================================================
# 演示入口
# ==============================================================================
if __name__ == "__main__":
    print("🎉 Wordle玩家表现预测演示")
    print("\\n步骤1：加载模型和特征工程对象...")

    try:
        # 1. 加载预测组件
        models, fe = load_prediction_components()

        # 2. 获取演示用的历史数据（使用训练数据的最后7天）
        print("\\n步骤2：准备历史数据（使用训练数据的最后7天）...")
        demo_historical_data = fe.df.tail(7).copy()

        # 显示历史数据基本信息
        print("\\n📅 历史数据信息（最后7天）:")
        print(
            f"   - 日期范围: {demo_historical_data['Date'].min().strftime('%Y-%m-%d')} ~ {demo_historical_data['Date'].max().strftime('%Y-%m-%d')}")
        print("   - 关键指标预览:")
        preview_cols = ['Date', 'Average_tries', 'Success_rate', 'Hard_mode_ratio']
        preview_data = demo_historical_data[preview_cols].copy()
        preview_data['Success_rate'] = (preview_data['Success_rate'] * 100).round(1)
        preview_data['Hard_mode_ratio'] = (preview_data['Hard_mode_ratio'] * 100).round(1)
        preview_data['Date'] = preview_data['Date'].dt.strftime('%Y-%m-%d')
        print(preview_data.to_string(index=False, col_space=12))

        # 3. 执行预测
        print("\\n步骤3：执行预测...")
        predictions = predict_player_performance(demo_historical_data, models, fe)

        # 4. 打印预测报告
        print_prediction_report(predictions, fe)

        print("\\n✅ 演示完成！若需预测新数据，可修改demo_historical_data为你的7天历史数据")

    except Exception as e:
        print(f"\\n❌ 演示出错: {str(e)}")
        print("\\n💡 解决方案:")
        print("   1. 确保已先运行 wordle_prediction_project.py 生成模型和特征工程文件")
        print("   2. 确保所有 .h5 模型文件和 wordle_feature_engineering.pkl 在同一目录")
        print("   3. 确保2023_MCM_Problem_C_Data.xlsx 数据文件路径正确")