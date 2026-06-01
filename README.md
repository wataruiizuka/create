# 作業引き継ぎ記録：オーナーズカード該当ユーザーID抽出・Databricksテーブル化

## 1. 作業背景

依頼元から、オーナーズカード該当ユーザーの抽出について以下の確認依頼があった。

> オーナーズカードの取り方がずれていると問題があるので、データ確認して案を出してください。

そのため、SSMSで `[IAEONPROD].[L2_MST_MEMBER_INFO]` を確認し、オーナーズカード該当ユーザーの定義候補を調査した。

最終的に担当者から、オーナーズカード該当ユーザーの定義には **`owners_card_number_app` を使う** と確認が取れた。

---

## 2. 確認対象テーブル

SSMS対象テーブル：

```sql
[IAEONPROD].[L2_MST_MEMBER_INFO]
```

テーブル構造確認結果：

| カラム名             | データ型      | 内容             |
| ---------------- | --------- | -------------- |
| IAEON_MEMBER_ID  | nvarchar  | iAEON会員ID      |
| DATA_KEY         | nvarchar  | 会員情報の種類を表すキー   |
| DATA_VALUE       | nvarchar  | DATA_KEYに対応する値 |
| REGISTER_DT      | datetime2 | 登録日時           |
| UPDATE_DT        | datetime2 | 更新日時           |
| DATA_UPDATE_DATE | decimal   | データ更新日時らしき数値   |
| DW_UPDATE_DT     | datetime  | DWH更新日時        |

このテーブルは、1会員1行ではなく、`IAEON_MEMBER_ID` × `DATA_KEY` のKey-Value形式の会員情報テーブルと判断した。

---

## 3. DATA_KEY確認結果

`DATA_KEY` は全部で65種類あった。

オーナーズカード関連候補として以下が確認できた。

| DATA_KEY               |  row_count | 見立て                         |
| ---------------------- | ---------: | --------------------------- |
| owners_card_number     |  6,640,801 | 通常のオーナーズカード番号候補。ただし空文字が多い   |
| use_owners_card        |    489,094 | オーナーズカード利用有無フラグ候補           |
| owners_card_number_app |    489,092 | アプリ側オーナーズカード番号候補            |
| membership_card_type   | 10,690,522 | 会員カード種別候補。ただしコード意味不明のため補助扱い |

---

## 4. DATA_VALUE確認結果

候補キーごとの `DATA_VALUE` 品質確認結果：

| DATA_KEY               |  row_count | blank_value_count | non_blank_value_count | numeric_only_count | 見立て               |
| ---------------------- | ---------: | ----------------: | --------------------: | -----------------: | ----------------- |
| membership_card_type   | 10,690,522 |         1,059,729 |             9,630,793 |          8,239,008 | 複雑。補助項目扱い         |
| owners_card_number     |  6,640,801 |         6,432,806 |               207,995 |            207,995 | 空文字が非常に多く、単独採用は危険 |
| owners_card_number_app |    489,092 |            36,200 |               452,892 |            452,892 | アプリ側番号として有力       |
| use_owners_card        |    489,094 |                 0 |               489,094 |            489,094 | 1/0フラグとして有力       |

`use_owners_card` の値分布：

| DATA_VALUE | row_count | 解釈     |
| ---------- | --------: | ------ |
| 1          |   451,625 | 利用あり候補 |
| 0          |    37,469 | 利用なし候補 |

この時点では、分析上は `use_owners_card = '1'` も有力だったが、担当者確認により、最終的には **`owners_card_number_app` に値がある会員をオーナーズカード該当ユーザーとする** 方針になった。

---

## 5. 最終定義

オーナーズカード該当ユーザーの定義：

```sql
DATA_KEY = N'owners_card_number_app'
AND LTRIM(RTRIM(DATA_VALUE)) <> N''
```

つまり、`L2_MST_MEMBER_INFO` において、`owners_card_number_app` に空文字ではない値が入っている `IAEON_MEMBER_ID` を抽出対象とする。

---

## 6. SSMSで実行したID抽出クエリ

SSMSで以下のクエリを実行し、オーナーズカード該当ユーザーIDのみを抽出した。

```sql
/* =========================================================
  オーナーズカード該当ユーザーID抽出

  定義：
    DATA_KEY = 'owners_card_number_app'
    かつ DATA_VALUE が空文字ではない

  出力：
    IAEON_MEMBER_ID のみ

  注意：
    同一ユーザーが複数行存在する可能性があるため、
    DISTINCT で重複排除する
========================================================= */

SELECT DISTINCT
    IAEON_MEMBER_ID
FROM [IAEONPROD].[L2_MST_MEMBER_INFO]
WHERE DATA_KEY = N'owners_card_number_app'
  AND LTRIM(RTRIM(DATA_VALUE)) <> N'';
```

抽出したCSVファイル名：

```text
owners_card_MEMBER_ID.csv
```

---

## 7. Databricksアップロード先

CSVはDatabricks上の以下にアップロード済み。

```text
/Volumes/cdp_bi_poc/iizuka/20260601_owners_card_app_member_id/owners_card_MEMBER_ID.csv
```

Databricksで保存したいテーブル名：

```sql
cdp_bi_poc.iizuka.`20260601_owners_card_app_member_id`
```

注意点：

テーブル名が数字始まりなので、SQL上では必ずバッククォートで囲む。

---

## 8. Databricks Step 1：ファイル存在確認

実行コード：

```python
# =========================================================
# Step 1：アップロード済みCSVファイルの存在確認
# =========================================================

csv_path = "/Volumes/cdp_bi_poc/iizuka/20260601_owners_card_app_member_id/owners_card_MEMBER_ID.csv"

catalog_name = "cdp_bi_poc"
schema_name = "iizuka"
table_name = "20260601_owners_card_app_member_id"

full_table_name = f"{catalog_name}.{schema_name}.`{table_name}`"

print("CSVファイルパス:")
print(csv_path)

print("\n保存先テーブル名:")
print(full_table_name)

print("\nファイル存在確認:")
display(dbutils.fs.ls("/Volumes/cdp_bi_poc/iizuka/20260601_owners_card_app_member_id/"))
```

結果：

| 項目   | 結果                                                                                           |
| ---- | -------------------------------------------------------------------------------------------- |
| path | dbfs:/Volumes/cdp_bi_poc/iizuka/20260601_owners_card_app_member_id/owners_card_MEMBER_ID.csv |
| name | owners_card_MEMBER_ID.csv                                                                    |
| size | 8,184,764                                                                                    |
| 判定   | OK                                                                                           |

---

## 9. Databricks Step 2：CSV読み込み・件数確認

実行コード：

```python
# =========================================================
# Step 2：owners_card_MEMBER_ID.csv の件数・NULL・重複確認
# =========================================================

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import functions as F

csv_path = "/Volumes/cdp_bi_poc/iizuka/20260601_owners_card_app_member_id/owners_card_MEMBER_ID.csv"

schema = StructType([
    StructField("IAEON_MEMBER_ID", StringType(), True)
])

df = (
    spark.read
    .option("header", "true")
    .option("encoding", "UTF-8")
    .schema(schema)
    .csv(csv_path)
)

print("===== Step 2：CSV読み込み後のスキーマ確認 =====")
df.printSchema()

print("===== Step 2-1：件数・NULL・空文字・重複確認 =====")

check_df = df.agg(
    F.count(F.lit(1)).alias("total_row_count"),
    F.count("IAEON_MEMBER_ID").alias("not_null_member_id_count"),
    F.sum(F.when(F.col("IAEON_MEMBER_ID").isNull(), 1).otherwise(0)).alias("null_member_id_count"),
    F.sum(F.when(F.trim(F.col("IAEON_MEMBER_ID")) == "", 1).otherwise(0)).alias("blank_member_id_count"),
    F.countDistinct("IAEON_MEMBER_ID").alias("distinct_member_id_count")
).withColumn(
    "duplicate_row_count",
    F.col("total_row_count") - F.col("distinct_member_id_count")
)

display(check_df)

print("===== Step 2-2：IAEON_MEMBER_ID サンプル確認 =====")

display(
    df.select("IAEON_MEMBER_ID")
      .limit(20)
)
```

結果：

| 項目                       |      件数 |
| ------------------------ | ------: |
| total_row_count          | 454,708 |
| not_null_member_id_count | 454,708 |
| null_member_id_count     |       0 |
| blank_member_id_count    |       0 |
| distinct_member_id_count | 454,708 |
| duplicate_row_count      |       0 |

判定：

* NULLなし
* 空文字なし
* 重複なし
* `IAEON_MEMBER_ID` は string 型
* CSVは正常

---

## 10. Databricks Step 3：クレンジング済みDataFrame作成

実行コード：

```python
# =========================================================
# Step 3：クレンジング済みDataFrame作成
# =========================================================

from pyspark.sql import functions as F

df_clean = (
    df
    .select(
        F.trim(F.col("IAEON_MEMBER_ID")).alias("IAEON_MEMBER_ID")
    )
    .where(
        F.col("IAEON_MEMBER_ID").isNotNull()
        & (F.col("IAEON_MEMBER_ID") != "")
    )
    .dropDuplicates(["IAEON_MEMBER_ID"])
)

print("===== Step 3：クレンジング済みDataFrame スキーマ確認 =====")
df_clean.printSchema()

print("===== Step 3-1：クレンジング後 件数確認 =====")

check_clean_df = df_clean.agg(
    F.count(F.lit(1)).alias("clean_total_row_count"),
    F.count("IAEON_MEMBER_ID").alias("clean_not_null_member_id_count"),
    F.sum(F.when(F.col("IAEON_MEMBER_ID").isNull(), 1).otherwise(0)).alias("clean_null_member_id_count"),
    F.sum(F.when(F.trim(F.col("IAEON_MEMBER_ID")) == "", 1).otherwise(0)).alias("clean_blank_member_id_count"),
    F.countDistinct("IAEON_MEMBER_ID").alias("clean_distinct_member_id_count")
).withColumn(
    "clean_duplicate_row_count",
    F.col("clean_total_row_count") - F.col("clean_distinct_member_id_count")
)

display(check_clean_df)

print("===== Step 3-2：クレンジング後 サンプル確認 =====")

display(
    df_clean
    .select("IAEON_MEMBER_ID")
    .limit(20)
)
```

結果は問題なし。
`df_clean` は保存可能な状態。

---

## 11. Databricks Step 4：Deltaテーブル保存

最初はSQL CTASで保存しようとした。

```python
df_clean.createOrReplaceTempView(temp_view_name)

spark.sql(f"""
CREATE OR REPLACE TABLE {full_table_name}
USING DELTA
AS
SELECT
    IAEON_MEMBER_ID
FROM {temp_view_name}
""")
```

しかし、以下のエラーが発生。

```text
[INSUFFICIENT_PERMISSIONS] Insufficient privileges:
User does not have permission SELECT on any file. SQLSTATE: 42501
```

`df_clean.count()` は成功しており、保存対象件数454,708件までは確認できていたため、CSV読み込み・DataFrame作成は問題なし。
SQL CTAS経由でファイル由来の一時Viewを読むところで権限エラーが出た可能性が高いと判断。

そのため、DataFrame APIの `saveAsTable()` に変更した。

修正版コード：

```python
# =========================================================
# Step 4 修正版：DataFrame APIでDeltaテーブルとして保存
# =========================================================

catalog_name = "cdp_bi_poc"
schema_name = "iizuka"
table_name = "20260601_owners_card_app_member_id"

full_table_name = f"{catalog_name}.{schema_name}.`{table_name}`"

print("===== Step 4 修正版：保存先テーブル確認 =====")
print(full_table_name)

save_row_count = df_clean.count()
print(f"保存対象件数: {save_row_count}")

(
    df_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(full_table_name)
)

print("===== Step 4 修正版：Deltaテーブル保存完了 =====")
print(f"保存先: {full_table_name}")
```

この修正版でテーブル作成成功。

作成済みテーブル：

```sql
cdp_bi_poc.iizuka.`20260601_owners_card_app_member_id`
```

保存対象件数：

```text
454,708件
```

---

## 12. Databricks Step 5：保存後の確認

保存後の確認を実施し、問題なさそうと判断済み。

確認観点：

* テーブルが存在する
* 件数が454,708件
* `IAEON_MEMBER_ID` がstring型
* NULLなし
* 空文字なし
* 重複なし
* サンプル確認OK

保存後確認用コード：

```python
# =========================================================
# Step 5：保存後のDeltaテーブル確認
# =========================================================

from pyspark.sql import functions as F

catalog_name = "cdp_bi_poc"
schema_name = "iizuka"
table_name = "20260601_owners_card_app_member_id"

full_table_name = f"{catalog_name}.{schema_name}.`{table_name}`"

print("===== Step 5：確認対象テーブル =====")
print(full_table_name)

owners_df = spark.table(full_table_name)

print("===== Step 5-1：スキーマ確認 =====")
owners_df.printSchema()

print("===== Step 5-2：件数・NULL・空文字・重複確認 =====")

check_saved_df = owners_df.agg(
    F.count(F.lit(1)).alias("total_row_count"),
    F.count("IAEON_MEMBER_ID").alias("not_null_member_id_count"),
    F.sum(F.when(F.col("IAEON_MEMBER_ID").isNull(), 1).otherwise(0)).alias("null_member_id_count"),
    F.sum(F.when(F.trim(F.col("IAEON_MEMBER_ID")) == "", 1).otherwise(0)).alias("blank_member_id_count"),
    F.countDistinct("IAEON_MEMBER_ID").alias("distinct_member_id_count")
).withColumn(
    "duplicate_row_count",
    F.col("total_row_count") - F.col("distinct_member_id_count")
)

display(check_saved_df)

print("===== Step 5-3：サンプル確認 =====")

display(
    owners_df
    .select("IAEON_MEMBER_ID")
    .limit(20)
)

print("===== Step 5-4：テーブル詳細確認 =====")

display(
    spark.sql(f"DESCRIBE DETAIL {full_table_name}")
)
```

---

## 13. 現在の完了状態

ここまでで完了していること：

1. SSMSで `L2_MST_MEMBER_INFO` の `DATA_KEY` を確認
2. オーナーズカード関連候補を確認
3. 担当者から `owners_card_number_app` を使う定義で確定
4. SSMSでオーナーズカード該当ユーザーIDのみ抽出
5. CSVをDatabricks Volumeへアップロード
6. DatabricksでCSV存在確認
7. CSVを文字列型で読み込み
8. 件数・NULL・空文字・重複確認
9. クレンジング済みDataFrame作成
10. Deltaテーブルとして保存
11. 保存後確認も問題なし

作成済みテーブル：

```sql
cdp_bi_poc.iizuka.`20260601_owners_card_app_member_id`
```

テーブル内容：

| カラム             | 型      | 内容                       |
| --------------- | ------ | ------------------------ |
| IAEON_MEMBER_ID | string | オーナーズカード該当ユーザーのiAEON会員ID |

件数：

```text
454,708件
```

定義：

```text
[IAEONPROD].[L2_MST_MEMBER_INFO] において
DATA_KEY = 'owners_card_number_app'
かつ DATA_VALUE が空文字ではない会員
```

---

## 14. 次に進める候補

次の作業として考えられるもの：

### 案A：他の会員・購買・属性テーブルとJOINする

作成したオーナーズカードIDテーブルを、カスタマーDNAや会員属性テーブルとJOINして、オーナーズカード該当者の属性集計に進む。

JOINキー候補：

```text
IAEON_MEMBER_ID
```

### 案B：オーナーズカードフラグを付与した分析用ベーステーブルを作る

他の会員母集団に対して、以下のようなフラグを付与する。

```sql
CASE
  WHEN owners.IAEON_MEMBER_ID IS NOT NULL THEN 1
  ELSE 0
END AS is_owners_card_app_member
```

### 案C：報告用に作業結果をまとめる

報告内容：

* オーナーズカード判定候補を確認した
* `owners_card_number` は空文字が多く単独採用は危険
* 担当者確認により `owners_card_number_app` を採用
* 該当ユーザーIDを454,708件抽出
* DatabricksにDeltaテーブルとして保存完了

---

## 15. 次回チャットで最初に伝えるとよいこと

次回は以下のように開始するとよい。

> オーナーズカード該当ユーザーIDの抽出とDatabricksテーブル化までは完了しています。
> 作成済みテーブルは `cdp_bi_poc.iizuka.\`20260601_owners_card_app_member_id``で、件数は454,708件、カラムは`IAEON_MEMBER_ID`のみです。  
> 定義は`[IAEONPROD].[L2_MST_MEMBER_INFO]`における`DATA_KEY = 'owners_card_number_app'`かつ`DATA_VALUE` が空文字ではない会員です。
> 次は、このテーブルを使って会員属性や購買データとのJOIN、またはオーナーズカードフラグ付与に進みたいです。
