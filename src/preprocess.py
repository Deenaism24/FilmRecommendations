import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_users(users_df: pd.DataFrame) -> pd.DataFrame:
    # Заполняем пропуски в данных
    users_df['income'].fillna('unknown', inplace=True)
    users_df['sex'].fillna('unknown', inplace=True)
    users_df['kids_flg'].fillna(0, inplace=True)
    users_df['education'].fillna('unknown', inplace=True)

    # Преобразуем категориальные признаки в числовые
    label_encoders = {}
    for col in ['age_category', 'income', 'sex', 'education']:
        le = LabelEncoder()
        users_df[col] = le.fit_transform(users_df[col].astype(str))
        label_encoders[col] = le

    return users_df, label_encoders
