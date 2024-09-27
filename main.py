import pandas as pd
import pickle
from src.models import ALS
from src.models import TopPopular
from src.evaluation import evaluate_recommender
from test_recomendations import get_recommendations_for_user


def main():
    model_type = "ALS"

    data_folder = "./data/"
    train_part = pd.read_csv(data_folder + "train_data.csv", parse_dates=["datetime"])
    test_part = pd.read_csv(data_folder + "test_data.csv")
    test_part = test_part.groupby("user_id").agg({"movie_id": list}).reset_index()
    if model_type == 'ALS':
        model = ALS(iterations=15, factors=100, random_state=42)
    else:
        model = TopPopular()
    model.fit(train_part)
    test_part["recommendations"] = model.predict(test_part)

    metrics = evaluate_recommender(test_part, model_preds="recommendations")
    print(f"{model_type} metrics: {metrics}")
    with open(f"./saved_models/{model_type}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # предсказания для пользователя под номером 10 ALS
    get_recommendations_for_user(10, f"./saved_models/{model_type}_model.pkl", 3)


if __name__ == "__main__":
    main()


