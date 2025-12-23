"""
Wordle玩家表现预测项目 - 完整代码
包含数据处理、模型构建、训练、评估全流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Attention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import TransformerEncoder, TransformerEncoderLayer
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================
def load_and_preprocess_data(file_path):
    """加载并预处理Wordle数据"""
    # 读取数据
    df = pd.read_excel(file_path, header=1)

    # 设置列名
    new_columns = [
        'delete_col', 'Date', 'Contest_number', 'Word',
        'Reported_results', 'Hard_mode_count',
        '1_try_pct', '2_tries_pct', '3_tries_pct',
        '4_tries_pct', '5_tries_pct', '6_tries_pct',
        '7_plus_tries_pct'
    ]
    df.columns = new_columns

    # 数据清洗
    df = df.drop('delete_col', axis=1)
    df = df.dropna()

    # 数据类型转换
    df['Date'] = pd.to_datetime(df['Date'])
    numeric_cols = ['Contest_number', 'Reported_results', 'Hard_mode_count'] + \
                   [f'{i}_tries_pct' for i in range(1, 7)] + ['7_plus_tries_pct']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    df = df.sort_values('Date').reset_index(drop=True)

    # 计算衍生指标
    tries_weights = np.array([1, 2, 3, 4, 5, 6, 7])
    tries_matrix = df[[f'{i}_tries_pct' for i in range(1, 7)] + ['7_plus_tries_pct']].values / 100

    df['Average_tries'] = np.dot(tries_matrix, tries_weights)
    df['Success_rate'] = (df['1_try_pct'] + df['2_tries_pct'] + df['3_tries_pct'] +
                          df['4_tries_pct'] + df['5_tries_pct'] + df['6_tries_pct']) / 100
    df['Hard_mode_ratio'] = df['Hard_mode_count'] / df['Reported_results']
    df['Tries_std'] = np.sqrt(
        np.sum(tries_matrix * (tries_weights - df['Average_tries'].values.reshape(-1, 1)) ** 2, axis=1))
    df['high_success'] = (df['Success_rate'] > df['Success_rate'].mean()).astype(int)

    return df


# ==============================================================================
# 2. 特征工程 - 时间序列数据准备
# ==============================================================================
class WordleFeatureEngineering:
    def __init__(self, df, sequence_length=7):
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.scalers = {}

    def normalize_features(self, feature_columns):
        """标准化特征"""
        for col in feature_columns:
            scaler = StandardScaler()
            self.df[f'norm_{col}'] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
        return self.df

    def create_time_sequences(self, target_column, feature_columns):
        """创建时间序列数据"""
        X, y = [], []
        norm_feature_cols = [f'norm_{col}' for col in feature_columns]
        feature_data = self.df[norm_feature_cols].values
        target_data = self.df[target_column].values

        for i in range(len(self.df) - self.sequence_length):
            seq_features = feature_data[i:i + self.sequence_length]
            seq_target = target_data[i + self.sequence_length]
            X.append(seq_features)
            y.append(seq_target)

        return np.array(X), np.array(y)


# ==============================================================================
# 3. 模型构建 - LSTM系列模型
# ==============================================================================
class WordleLSTMModels:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_basic_lstm_regressor(self):
        """基础LSTM回归模型（预测平均尝试次数）"""
        inputs = Input(shape=self.input_shape)
        lstm1 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        lstm2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(lstm1)
        dense1 = Dense(16, activation='relu')(lstm2)
        dropout1 = Dropout(0.2)(dense1)
        outputs = Dense(1, activation='linear')(dropout1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_basic_lstm_classifier(self):
        """基础LSTM分类模型（预测高成功率）"""
        inputs = Input(shape=self.input_shape)
        lstm1 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        lstm2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(lstm1)
        dense1 = Dense(16, activation='relu')(lstm2)
        dropout1 = Dropout(0.2)(dense1)
        outputs = Dense(1, activation='sigmoid')(dropout1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_bilstm_attention_regressor(self):
        """BiLSTM+Attention回归模型（增强版）"""
        inputs = Input(shape=self.input_shape)
        bilstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
        attention = Attention()([bilstm, bilstm])
        attention_flatten = tf.keras.layers.Flatten()(attention)
        dense1 = Dense(32, activation='relu')(attention_flatten)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(16, activation='relu')(dropout1)
        outputs = Dense(1, activation='linear')(dense2)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


# ==============================================================================
# 4. 模型构建 - Transformer模型（扩展任务）
# ==============================================================================
class WordleTransformerModels:
    def __init__(self, input_shape, d_model=64, nhead=4, num_layers=2):
        self.input_shape = input_shape
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def _positional_encoding(self, seq_len, d_model):
        """生成位置编码"""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pos_enc = np.zeros((seq_len, d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)

        return tf.constant(pos_enc, dtype=tf.float32)[np.newaxis, ...]

    def build_transformer_regressor(self):
        """Transformer回归模型"""
        inputs = Input(shape=self.input_shape)
        projection = Dense(self.d_model)(inputs)
        seq_len = self.input_shape[0]
        pos_encoding = self._positional_encoding(seq_len, self.d_model)
        x = projection + pos_encoding

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.2
        )
        transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        x = transformer_encoder(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


# ==============================================================================
# 5. 模型训练与评估工具函数
# ==============================================================================
def train_model(model, X_train, y_train, epochs=50, batch_size=8, validation_split=0.2):
    """训练模型（含早停机制）"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history


def evaluate_regression_model(model, X_test, y_test, model_name):
    """评估回归模型"""
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"\n{model_name} 回归模型评估结果:")
    print(f" - MSE (均方误差): {mse:.4f}")
    print(f" - RMSE (均方根误差): {rmse:.4f}")
    print(f" - MAE (平均绝对误差): {mae:.4f}")

    return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae}


def evaluate_classification_model(model, X_test, y_test, model_name):
    """评估分类模型"""
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{model_name} 分类模型评估结果:")
    print(f" - 准确率 (Accuracy): {accuracy:.4f}")
    print(f" - 精确率 (Precision): {report['1']['precision']:.4f}")
    print(f" - 召回率 (Recall): {report['1']['recall']:.4f}")
    print(f" - F1分数: {report['1']['f1-score']:.4f}")

    return y_pred, y_pred_prob, {'accuracy': accuracy, 'precision': report['1']['precision'],
                                 'recall': report['1']['recall'], 'f1': report['1']['f1-score']}


# ==============================================================================
# 6. 主函数 - 项目执行入口
# ==============================================================================
def main(file_path, sequence_length=7):
    """主函数：执行完整项目流程"""
    print("=" * 60)
    print("Wordle玩家表现预测项目 - 开始执行")
    print("=" * 60)

    # 1. 数据加载与预处理
    print("\\n1. 数据加载与预处理...")
    df = load_and_preprocess_data(file_path)
    print(f"✅ 数据处理完成，数据形状: {df.shape}")

    # 2. 特征工程与时间序列准备
    print("\\n2. 特征工程与时间序列数据准备...")
    feature_columns = [
        'Average_tries', 'Success_rate', 'Hard_mode_ratio', 'Tries_std',
        '1_try_pct', '2_tries_pct', '3_tries_pct', '4_tries_pct',
        '5_tries_pct', '6_tries_pct', '7_plus_tries_pct',
        'Reported_results', 'Hard_mode_count'
    ]

    # 初始化特征工程
    fe = WordleFeatureEngineering(df, sequence_length=sequence_length)
    df_processed = fe.normalize_features(feature_columns)

    # 生成回归和分类任务的时间序列数据
    X_reg, y_reg = fe.create_time_sequences('Average_tries', feature_columns)  # 回归：预测平均尝试次数
    X_cls, y_cls = fe.create_time_sequences('high_success', feature_columns)  # 分类：预测高成功率

    # 划分训练集（80%）和测试集（20%）
    split_idx = int(len(X_reg) * 0.8)
    X_reg_train, X_reg_test = X_reg[:split_idx], X_reg[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    X_cls_train, X_cls_test = X_cls[:split_idx], X_cls[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]

    print(f"✅ 时间序列数据准备完成:")
    print(f"   - 回归任务: 训练集{X_reg_train.shape}, 测试集{X_reg_test.shape}")
    print(f"   - 分类任务: 训练集{X_cls_train.shape}, 测试集{X_cls_test.shape}")

    # 3. 模型构建与训练
    input_shape = (X_reg_train.shape[1], X_reg_train.shape[2])  # (sequence_length, num_features)
    print(f"\\n3. 模型构建与训练 (输入形状: {input_shape})...")

    # 3.1 训练基础LSTM回归模型
    print("\\n3.1 训练基础LSTM回归模型...")
    lstm_builder = WordleLSTMModels(input_shape)
    lstm_reg_model = lstm_builder.build_basic_lstm_regressor()
    lstm_reg_model, _ = train_model(lstm_reg_model, X_reg_train, y_reg_train)
    lstm_reg_model.save('lstm_regression_model.h5')
    print("✅ 基础LSTM回归模型已保存")

    # 3.2 训练基础LSTM分类模型
    print("\\n3.2 训练基础LSTM分类模型...")
    lstm_cls_model = lstm_builder.build_basic_lstm_classifier()
    lstm_cls_model, _ = train_model(lstm_cls_model, X_cls_train, y_cls_train)
    lstm_cls_model.save('lstm_classification_model.h5')
    print("✅ 基础LSTM分类模型已保存")

    # 3.3 训练BiLSTM+Attention回归模型
    print("\\n3.3 训练BiLSTM+Attention回归模型...")
    bilstm_att_reg_model = lstm_builder.build_bilstm_attention_regressor()
    bilstm_att_reg_model, _ = train_model(bilstm_att_reg_model, X_reg_train, y_reg_train)
    bilstm_att_reg_model.save('bilstm_attention_regression_model.h5')
    print("✅ BiLSTM+Attention回归模型已保存")

    # 3.4 训练Transformer回归模型（扩展任务）
    print("\\n3.4 训练Transformer回归模型...")
    transformer_builder = WordleTransformerModels(input_shape)
    transformer_reg_model = transformer_builder.build_transformer_regressor()
    transformer_reg_model, _ = train_model(transformer_reg_model, X_reg_train, y_reg_train)
    transformer_reg_model.save('transformer_regression_model.h5')
    print("✅ Transformer回归模型已保存")

    # 4. 模型评估
    print("\\n4. 模型评估...")

    # 评估回归模型
    y_pred_lstm_reg, metrics_lstm_reg = evaluate_regression_model(lstm_reg_model, X_reg_test, y_reg_test, "基础LSTM")
    y_pred_bilstm_att_reg, metrics_bilstm_att_reg = evaluate_regression_model(bilstm_att_reg_model, X_reg_test,
                                                                              y_reg_test, "BiLSTM+Attention")
    y_pred_transformer_reg, metrics_transformer_reg = evaluate_regression_model(transformer_reg_model, X_reg_test,
                                                                                y_reg_test, "Transformer")

    # 评估分类模型
    y_pred_lstm_cls, y_pred_lstm_cls_prob, metrics_lstm_cls = evaluate_classification_model(lstm_cls_model, X_cls_test,
                                                                                            y_cls_test, "基础LSTM")

    # 5. 保存特征工程对象（用于后续预测）
    import pickle
    with open('wordle_feature_engineering.pkl', 'wb') as f:
        pickle.dump(fe, f)
    print("\\n✅ 特征工程对象已保存为 wordle_feature_engineering.pkl")

    # 6. 生成模型对比报告
    print("\\n5. 生成模型对比报告...")
    complete_comparison_data = {
        '模型类型': ['基础LSTM回归', 'BiLSTM+Attention回归', 'Transformer回归', '基础LSTM分类'],
        '任务类型': ['回归（平均尝试次数）', '回归（平均尝试次数）', '回归（平均尝试次数）', '分类（高成功率预测）'],
        '主要评估指标': ['RMSE', 'RMSE', 'RMSE', 'Accuracy'],
        '主要指标值': [f'{metrics_lstm_reg["rmse"]:.4f}',
                       f'{metrics_bilstm_att_reg["rmse"]:.4f}',
                       f'{metrics_transformer_reg["rmse"]:.4f}',
                       f'{metrics_lstm_cls["accuracy"]:.4f}'],
        '辅助指标1': [f'MAE: {metrics_lstm_reg["mae"]:.4f}',
                      f'MAE: {metrics_bilstm_att_reg["mae"]:.4f}',
                      f'MAE: {metrics_transformer_reg["mae"]:.4f}',
                      f'F1: {metrics_lstm_cls["f1"]:.4f}'],
        '辅助指标2': [f'MSE: {metrics_lstm_reg["mse"]:.4f}',
                      f'MSE: {metrics_bilstm_att_reg["mse"]:.4f}',
                      f'MSE: {metrics_transformer_reg["mse"]:.4f}',
                      f'Precision: {metrics_lstm_cls["precision"]:.4f}'],
        '模型复杂度': ['低', '中', '高', '低']
    }

    complete_comparison_df = pd.DataFrame(complete_comparison_data)
    complete_comparison_df.to_csv('complete_model_comparison_report.csv', index=False, encoding='utf-8')
    print("✅ 完整模型对比报告已保存")

    print("\\n" + "=" * 60)
    print("Wordle玩家表现预测项目 - 执行完成")
    print("=" * 60)

    # 返回关键结果（供后续使用）
    return {
        'data': df,
        'models': {
            'lstm_reg': lstm_reg_model,
            'lstm_cls': lstm_cls_model,
            'bilstm_att_reg': bilstm_att_reg_model,
            'transformer_reg': transformer_reg_model
        },
        'metrics': {
            'lstm_reg': metrics_lstm_reg,
            'lstm_cls': metrics_lstm_cls,
            'bilstm_att_reg': metrics_bilstm_att_reg,
            'transformer_reg': metrics_transformer_reg
        }
    }


# ==============================================================================
# 7. 执行项目（需指定数据文件路径）
# ==============================================================================
if __name__ == "__main__":
    # 请将此处的文件路径替换为你的2023_MCM_Problem_C_Data.xlsx实际路径
    DATA_FILE_PATH = "2023_MCM_Problem_C_Data.xlsx"

    # 执行主函数
    try:
        results = main(DATA_FILE_PATH, sequence_length=7)

        # 打印最终性能总结
        print("\\n📋 最终模型性能总结:")
        print(f"1. 基础LSTM回归模型 - RMSE: {results['metrics']['lstm_reg']['rmse']:.4f}")
        print(f"2. BiLSTM+Attention回归模型 - RMSE: {results['metrics']['bilstm_att_reg']['rmse']:.4f}")
        print(f"3. Transformer回归模型 - RMSE: {results['metrics']['transformer_reg']['rmse']:.4f}")
        print(f"4. 基础LSTM分类模型 - Accuracy: {results['metrics']['lstm_cls']['accuracy']:.4f}")

    except FileNotFoundError:
        print(f"❌ 错误：未找到数据文件 {DATA_FILE_PATH}")
        print("请确保2023_MCM_Problem_C_Data.xlsx文件在当前目录下，或修改DATA_FILE_PATH为正确路径")
    except Exception as e:
        print(f"❌ 项目执行出错: {str(e)}")