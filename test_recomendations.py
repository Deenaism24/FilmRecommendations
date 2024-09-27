import pandas as pd
import pickle


def get_recommendations_for_user(user_id: int, model_path: str, topn: int = 5):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    user_df = pd.DataFrame({"user_id": [user_id]})
    recommend = model.predict(user_df, topn=topn)

    items_df = pd.read_csv("./data/items_df.csv")
    print(f"Рекомендации для пользователя {user_id}:")
    for id in recommend[0]:
        title = items_df[items_df["id"] == id]["title"].values[0]
        print(title)