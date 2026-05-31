import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
from pyspark import StorageLevel

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    agg_reviews = review_data.groupBy("asin").agg(
        F.avg("overall").alias('meanRating'),
        F.count("overall").alias("countRating")
    )

    joined_df = product_data.join(agg_reviews, on="asin", how="left").persist(StorageLevel.MEMORY_AND_DISK)

    stats = joined_df.select(
        F.avg("meanRating"), 
        F.variance("meanRating"), 
        F.sum(F.col("meanRating").isNull().cast("int")),
        F.avg("countRating"), 
        F.variance("countRating"), 
        F.sum(F.col("countRating").isNull().cast("int"))
    ).collect()[0]

    count_total = joined_df.count()

    joined_df.unpersist()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': count_total,
        'mean_meanRating': stats[0],
        'variance_meanRating': stats[1],
        'numNulls_meanRating': stats[2],
        'mean_countRating': stats[3],
        'variance_countRating': stats[4],
        'numNulls_countRating': stats[5]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    df_with_cat = product_data.withColumn(
        "category", 
        F.when(F.col("categories")[0][0] != "", F.col("categories")[0][0]).otherwise(None)
    )
    
    df_with_sales = df_with_cat.withColumn("map_keys", F.map_keys(F.col("salesRank")))\
                               .withColumn("map_vals", F.map_values(F.col("salesRank")))
    
    processed_df = df_with_sales.withColumn(
        "bestSalesCategory", F.col("map_keys")[0]
    ).withColumn(
        "bestSalesRank", F.col("map_vals")[0]
    ).persist(StorageLevel.MEMORY_AND_DISK)
    
    count_total = processed_df.count()
    stats = processed_df.select(
        F.avg("bestSalesRank"),
        F.variance("bestSalesRank"),
        F.sum(F.col("category").isNull().cast("int")),
        F.countDistinct("category"),
        F.sum(F.col("bestSalesCategory").isNull().cast("int")),
        F.countDistinct("bestSalesCategory")
    ).collect()[0]
    
    processed_df.unpersist()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_bestSalesRank': stats[0],
        'variance_bestSalesRank': stats[1],
        'numNulls_category': stats[2],
        'countDistinct_category': stats[3],
        'numNulls_bestSalesCategory': stats[4],
        'countDistinct_bestSalesCategory': stats[5]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    df_with_count = product_data.withColumn(
        "countAlsoViewed",
        F.when(
            F.col("related.also_viewed").isNotNull() & (F.size(F.col("related.also_viewed")) > 0),
            F.size(F.col("related.also_viewed"))
        ).otherwise(None)
    )
    
    # Explode array to cross-reference product pricing
    exploded_df = product_data.select("asin", F.explode("related.also_viewed").alias("viewed_asin"))
    prices_lookup = product_data.select(F.col("asin").alias("viewed_asin"), F.col("price").alias("viewed_price"))
    
    joined_prices = exploded_df.join(prices_lookup, on="viewed_asin", how="inner")\
                               .filter(F.col("viewed_price").isNotNull())
    
    mean_prices_df = joined_prices.groupBy("asin").agg(F.avg("viewed_price").alias("meanPriceAlsoViewed"))
    
    final_df = df_with_count.join(mean_prices_df, on="asin", how="left").persist(StorageLevel.MEMORY_AND_DISK)
    
    count_total = final_df.count()
    stats = final_df.select(
        F.avg("meanPriceAlsoViewed"),
        F.variance("meanPriceAlsoViewed"),
        F.sum(F.col("meanPriceAlsoViewed").isNull().cast("int")),
        F.avg("countAlsoViewed"),
        F.variance("countAlsoViewed"),
        F.sum(F.col("countAlsoViewed").isNull().cast("int"))
    ).collect()[0]
    
    final_df.unpersist()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_meanPriceAlsoViewed': stats[0],
        'variance_meanPriceAlsoViewed': stats[1],
        'numNulls_meanPriceAlsoViewed': stats[2],
        'mean_countAlsoViewed': stats[3],
        'variance_countAlsoViewed': stats[4],
        'numNulls_countAlsoViewed': stats[5]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    df = product_data.withColumn("price", F.col("price").cast("float"))
    
    mean_val = df.select(F.avg(F.when(~F.isnan(F.col("price")), F.col("price")))).collect()[0][0]
    mean_price = float(mean_val) if mean_val is not None else 0.0
    
    quantiles = df.filter(~F.isnan(F.col("price")) & F.col("price").isNotNull()).approxQuantile("price", [0.5], 0.001)
    median_price = float(quantiles[0]) if (quantiles and quantiles[0] is not None) else 0.0
    
    imputed_df = df.withColumn(
        "meanImputedPrice",
        F.coalesce(F.when(F.col("price").isNotNull() & F.isnan(F.col("price")), None).otherwise(F.col("price")), F.lit(mean_price))
    ).withColumn(
        "medianImputedPrice",
        F.coalesce(F.when(F.col("price").isNotNull() & F.isnan(F.col("price")), None).otherwise(F.col("price")), F.lit(median_price))
    ).withColumn(
        "unknownImputedTitle",
        F.when((F.col("title").isNull()) | (F.col("title") == ""), "unknown").otherwise(F.col("title"))
    ).persist(StorageLevel.MEMORY_AND_DISK)
                   
    count_total = imputed_df.count()
    stats = imputed_df.select(
        F.avg("meanImputedPrice"),
        F.variance("meanImputedPrice"),
        F.sum(F.col("meanImputedPrice").isNull().cast("int")),
        F.avg("medianImputedPrice"),
        F.variance("medianImputedPrice"),
        F.sum(F.col("medianImputedPrice").isNull().cast("int")),
        F.sum((F.col("unknownImputedTitle") == "unknown").cast("int"))
    ).collect()[0]
    
    imputed_df.unpersist()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': count_total,
        'mean_meanImputedPrice': stats[0],
        'variance_meanImputedPrice': stats[1],
        'numNulls_meanImputedPrice': stats[2],
        'mean_medianImputedPrice': stats[3],
        'variance_medianImputedPrice': stats[4],
        'numNulls_medianImputedPrice': stats[5],
        'numUnknowns_unknownImputedTitle': stats[6]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    tokenized_df = product_processed_data.withColumn(
        "titleArray", 
        F.split(F.lower(F.col("title")), " ")
    )
    
    word2Vec = (M.feature.Word2Vec()
                .setVectorSize(16)
                .setMinCount(100)
                .setSeed(SEED)
                .setNumPartitions(4)
                .setInputCol(titleArray_column)
                .setOutputCol(titleVector_column))
    
    model = word2Vec.fit(tokenized_df)
    transformed_df = model.transform(tokenized_df)

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': transformed_df.count(),
        'size_vocabulary': model.getVectors().count(),
        'word_0_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_0, 10).collect()],
        'word_1_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_1, 10).collect()],
        'word_2_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_2, 10).collect()]
    }
    # Modify res:



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    indexer = M.feature.StringIndexer(inputCol="category", outputCol="categoryIndex")
    indexed_df = indexer.fit(product_processed_data).transform(product_processed_data)
    
    encoder = M.feature.OneHotEncoder(inputCol="categoryIndex", outputCol="categoryOneHot", dropLast=False)
    encoded_df = encoder.fit(indexed_df).transform(indexed_df)
    
    pca = M.feature.PCA(k=15, inputCol="categoryOneHot", outputCol="categoryPCA")
    pca_model = pca.fit(encoded_df)
    transformed_df = pca_model.transform(encoded_df).cache()
    
    summary_row = transformed_df.select(
        M.stat.Summarizer.mean(F.col("categoryOneHot")),
        M.stat.Summarizer.mean(F.col("categoryPCA"))
    ).collect()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': int(transformed_df.count()),
        'meanVector_categoryOneHot': [float(x) for x in summary_row[0].toArray()],
        'meanVector_categoryPCA': [float(x) for x in summary_row[1].toArray()]
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    dt = M.regression.DecisionTreeRegressor(
        featuresCol="features",
        labelCol="overall",
        maxDepth=5
    )
    
    model = dt.fit(train_data)
    predictions = model.transform(test_data)
    
    evaluator = M.evaluation.RegressionEvaluator(
        labelCol="overall",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    test_rmse = evaluator.evaluate(predictions)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': float(test_rmse)
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    train_split, valid_split = train_data.randomSplit([0.75, 0.25], seed=SEED)
    
    evaluator = M.evaluation.RegressionEvaluator(
        labelCol="overall",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    valid_rmses = {}
    models = {}
    
    for depth in [5, 7, 9, 12]:
        dt = M.regression.DecisionTreeRegressor(
            featuresCol="features",
            labelCol="overall",
            maxDepth=depth
        )
        model = dt.fit(train_split)
        valid_preds = model.transform(valid_split)
        valid_rmses[depth] = float(evaluator.evaluate(valid_preds))
        models[depth] = model
    
    best_depth = min(valid_rmses, key=valid_rmses.get)
    best_model = models[best_depth]
    
    test_preds = best_model.transform(test_data)
    test_rmse = float(evaluator.evaluate(test_preds))
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': test_rmse,
        'valid_rmse_depth_5': valid_rmses[5],
        'valid_rmse_depth_7': valid_rmses[7],
        'valid_rmse_depth_9': valid_rmses[9],
        'valid_rmse_depth_12': valid_rmses[12],
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

