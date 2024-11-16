
#データの読み込み
import pandas as pd

df = pd.read_csv("/Users/iidzukawataru/Documents/create/archive/results.csv")

print(df.head())
print(df.info())

#日本代表のデータを抽出
japan_data = df[(df["home_team"] == "Japan") | (df["away_team"] == "Japan")]
print(japan_data.head())
print(f"日本代表の試合数： {len(japan_data)}")

#勝敗のラベル付
#試合結果に基づいて、勝ち・引き分け・負けを「result」列に追加します。

def determine_result(row):
    if row["home_team"] == "Japan" and row["home_score"] > row["away_score"]:
        return "win"
    elif row["away_team"] == "Japan" and row["away_score"] > row["home_score"]:
        return "win"
    elif row["home_score"] == row["away_score"]:
        return "draw"
    else:
        return "lose"
    
japan_data["result"] = japan_data.apply(determine_result, axis = 1)

 #勝敗列の確認
print(japan_data[["home_team", "away_team", "home_score", "away_score", "result"]].head())


#データの前処理

#1日付データの変換
japan_data["date"] = pd.to_datetime(japan_data["date"])
japan_data["year"] = japan_data["date"].dt.year
japan_data["month"] = japan_data["date"].dt.year

#2カテゴリ変数のエンコーディング
japan_data = pd.get_dummies(japan_data, columns = ["country","home_team", "away_team", "tournament", "city"], drop_first = True)

#3ターゲット列の作成
japan_data["target"] = (japan_data["result"] == "win").astype(int)

#モデル用データ作成
X = japan_data.drop(["date", "result", "target", "home_score", "away_score"], axis = 1)
y = japan_data["target"]

print(X.head())
print(X.dtypes)

#モデル作成と学習
#1データ分割
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#2モデルの構築と学習
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

#3モデルの評価
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")