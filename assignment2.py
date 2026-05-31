import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec, StringIndexer, OneHotEncoder, PCA
from pyspark.ml.stat import Summarizer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ==========================================
# PART 1: FEATURE ENGINEERING
# ==========================================

def task_1(data_io, review_data, product_data):
    # Aggregate reviews by product (asin)
    agg_reviews = review_data.groupBy("asin").agg(
        F.avg("overall").alias("meanRating"),
        F.count("overall").alias("countRating")
    )
    
    # Left join to retain all products, even those without reviews
    joined_df = product_data.join(agg_reviews, on="asin", how="left").cache()
    
    count_total = joined_df.count()
    stats = joined_df.select(
        F.avg("meanRating"), F.variance("meanRating"), F.sum(F.col("meanRating").isNull().cast("int")),
        F.avg("countRating"), F.variance("countRating"), F.sum(F.col("countRating").isNull().cast("int"))
    ).collect()[0]
    
    res = {
        'count_total': int(count_total),
        'mean_meanRating': float(stats[0]) if stats[0] is not None else None,
        'variance_meanRating': float(stats[1]) if stats[1] is not None else None,
        'numNulls_meanRating': int(stats[2]) if stats[2] is not None else 0,
        'mean_countRating': float(stats[3]) if stats[3] is not None else None,
        'variance_countRating': float(stats[4]) if stats[4] is not None else None,
        'numNulls_countRating': int(stats[5]) if stats[5] is not None else 0
    }
    data_io.save(res, 'task_1')
    return res


def task_2(data_io, product_data):
    # Unpack nested structures safely
    flat_categories_df = product_data.withColumn("category_arr", F.flatten(F.col("categories")))
    df_with_cat = flat_categories_df.withColumn(
        "category", 
        F.when(F.size("category_arr") > 0, F.col("category_arr")[0]).otherwise(None)
    )
    
    df_with_sales = df_with_cat.withColumn("map_keys", F.map_keys(F.col("salesRank")))\
                               .withColumn("map_vals", F.map_values(F.col("salesRank")))
    
    processed_df = df_with_sales.withColumn(
        "bestSalesCategory", F.when(F.size("map_keys") > 0, F.col("map_keys")[0]).otherwise(None)
    ).withColumn(
        "bestSalesRank", F.when(F.size("map_vals") > 0, F.col("map_vals")[0].cast("int")).otherwise(None)
    ).cache()
    
    count_total = processed_df.count()
    stats = processed_df.select(
        F.avg("bestSalesRank"), F.variance("bestSalesRank"),
        F.sum(F.col("category").isNull().cast("int")), F.countDistinct("category"),
        F.sum(F.col("bestSalesCategory").isNull().cast("int")), F.countDistinct("bestSalesCategory")
    ).collect()[0]
    
    res = {
        'count_total': int(count_total),
        'mean_bestSalesRank': float(stats[0]) if stats[0] is not None else None,
        'variance_bestSalesRank': float(stats[1]) if stats[1] is not None else None,
        'numNulls_category': int(stats[2]) if stats[2] is not None else 0,
        'countDistinct_category': int(stats[3]) if stats[3] is not None else 0,
        'numNulls_bestSalesCategory': int(stats[4]) if stats[4] is not None else 0,
        'countDistinct_bestSalesCategory': int(stats[5]) if stats[5] is not None else 0
    }
    data_io.save(res, 'task_2')
    return res


def task_3(data_io, product_data):
    df_with_count = product_data.withColumn(
        "countAlsoViewed",
        F.when(F.col("related.also_viewed").isNotNull(), F.size(F.col("related.also_viewed"))).otherwise(0)
    )
    
    exploded_df = product_data.select("asin", F.explode_outer("related.also_viewed").alias("viewed_asin"))
    prices_lookup = product_data.select(F.col("asin").alias("viewed_asin"), F.col("price").alias("viewed_price"))
    
    joined_prices = exploded_df.join(prices_lookup, on="viewed_asin", how="inner")\
                               .filter(F.col("viewed_price").isNotNull())
    
    mean_prices_df = joined_prices.groupBy("asin").agg(F.avg("viewed_price").alias("meanPriceAlsoViewed"))
    final_df = df_with_count.join(mean_prices_df, on="asin", how="left").cache()
    
    count_total = final_df.count()
    stats = final_df.select(
        F.avg("meanPriceAlsoViewed"), F.variance("meanPriceAlsoViewed"), F.sum(F.col("meanPriceAlsoViewed").isNull().cast("int")),
        F.avg("countAlsoViewed"), F.variance("countAlsoViewed"), F.sum(F.col("countAlsoViewed").isNull().cast("int"))
    ).collect()[0]
    
    res = {
        'count_total': int(count_total),
        'mean_meanPriceAlsoViewed': float(stats[0]) if stats[0] is not None else None,
        'variance_meanPriceAlsoViewed': float(stats[1]) if stats[1] is not None else None,
        'numNulls_meanPriceAlsoViewed': int(stats[2]) if stats[2] is not None else 0,
        'mean_countAlsoViewed': float(stats[3]) if stats[3] is not None else None,
        'variance_countAlsoViewed': float(stats[4]) if stats[4] is not None else None,
        'numNulls_countAlsoViewed': int(stats[5]) if stats[5] is not None else 0
    }
    data_io.save(res, 'task_3')
    return res


def task_4(data_io, product_data):
    df = product_data.withColumn("price", F.col("price").cast("float"))
    
    mean_val = df.select(F.avg("price")).collect()[0][0]
    mean_price = float(mean_val) if mean_val is not None else 0.0
    
    quantiles = df.approxQuantile("price", [0.5], 0.001)
    median_price = float(quantiles[0]) if (quantiles and quantiles[0] is not None) else 0.0
    
    imputed_df = df.withColumn("meanImputedPrice", F.coalesce(F.col("price"), F.lit(mean_price)))\
                   .withColumn("medianImputedPrice", F.coalesce(F.col("price"), F.lit(median_price)))\
                   .withColumn("unknownImputedTitle", F.when((F.col("title").isNull()) | (F.col("title") == ""), "unknown").otherwise(F.col("title"))).cache()
                   
    count_total = imputed_df.count()
    stats = imputed_df.select(
        F.avg("meanImputedPrice"), F.variance("meanImputedPrice"), F.sum(F.col("meanImputedPrice").isNull().cast("int")),
        F.avg("medianImputedPrice"), F.variance("medianImputedPrice"), F.sum(F.col("medianImputedPrice").isNull().cast("int")),
        F.sum((F.col("unknownImputedTitle") == "unknown").cast("int"))
    ).collect()[0]
    
    res = {
        'count_total': int(count_total),
        'mean_meanImputedPrice': float(stats[0]) if stats[0] is not None else None,
        'variance_meanImputedPrice': float(stats[1]) if stats[1] is not None else None,
        'numNulls_meanImputedPrice': int(stats[2]) if stats[2] is not None else 0,
        'mean_medianImputedPrice': float(stats[3]) if stats[3] is not None else None,
        'variance_medianImputedPrice': float(stats[4]) if stats[4] is not None else None,
        'numNulls_medianImputedPrice': int(stats[5]) if stats[5] is not None else 0,
        'numUnknowns_unknownImputedTitle': int(stats[6]) if stats[6] is not None else 0
    }
    data_io.save(res, 'task_4')
    return res


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    tokenized_df = product_processed_data.withColumn(
        "titleArray", 
        F.split(F.lower(F.trim(F.col("title"))), r"\s+")
    )
    
    word2Vec = Word2Vec(
        vectorSize=16, 
        minCount=100, 
        seed=102, 
        numPartitions=4, 
        inputCol="titleArray", 
        outputCol="titleVector"
    )
    
    model = word2Vec.fit(tokenized_df)
    transformed_df = model.transform(tokenized_df).cache()
    
    res = {
        'count_total': int(transformed_df.count()),
        'size_vocabulary': int(model.getVectors().count()),
        'word_0_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_0, 10).collect()],
        'word_1_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_1, 10).collect()],
        'word_2_synonyms': [(row[0], float(row[1])) for row in model.findSynonyms(word_2, 10).collect()]
    }
    data_io.save(res, 'task_5')
    return res


def task_6(data_io, product_processed_data):
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex", handleInvalid="keep")
    indexed_df = indexer.fit(product_processed_data).transform(product_processed_data)
    
    encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryOneHot", dropLast=False)
    encoded_df = encoder.fit(indexed_df).transform(indexed_df)
    
    pca = PCA(k=15, inputCol="categoryOneHot", outputCol="categoryPCA")
    pca_model = pca.fit(encoded_df)
    transformed_df = pca_model.transform(encoded_df).cache()
    
    summary_row = transformed_df.select(
        Summarizer.mean(F.col("categoryOneHot")),
        Summarizer.mean(F.col("categoryPCA"))
    ).collect()[0]
    
    res = {
        'count_total': int(transformed_df.count()),
        'meanVector_categoryOneHot': [float(x) for x in summary_row[0].toArray()],
        'meanVector_categoryPCA': [float(x) for x in summary_row[1].toArray()]
    }
    data_io.save(res, 'task_6')
    return res

# ==========================================
# PART 2: MODEL SELECTION
# ==========================================

def task_7(data_io, train_data, test_data):
    dt = DecisionTreeRegressor(maxDepth=5, featuresCol="features", labelCol="label")
    model = dt.fit(train_data)
    predictions = model.transform(test_data)
    
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    test_rmse = float(evaluator.evaluate(predictions))
    
    res = {
        'test_rmse': test_rmse
    }
    data_io.save(res, 'task_7')
    return res


def task_8(data_io, train_data, test_data):
    # Split train data 75/25 with hardcoded assignment seed
    sub_train, validation_df = train_data.randomSplit([0.75, 0.25], seed=102)
    sub_train.cache()
    validation_df.cache()
    
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    
    depth_metrics = {}
    models_pool = {}
    target_depths = [5, 7, 9, 12]
    
    for depth in target_depths:
        dt = DecisionTreeRegressor(maxDepth=depth, featuresCol="features", labelCol="label")
        model = dt.fit(sub_train)
        predictions = model.transform(validation_df)
        
        rmse_score = float(evaluator.evaluate(predictions))
        depth_metrics[depth] = rmse_score
        models_pool[depth] = model
        
    best_depth = min(depth_metrics, key=depth_metrics.get)
    best_model = models_pool[best_depth]
    
    test_predictions = best_model.transform(test_data)
    final_test_rmse = float(evaluator.evaluate(test_predictions))
    
    res = {
        'test_rmse': final_test_rmse,
        'valid_rmse_depth_5': depth_metrics[5],
        'valid_rmse_depth_7': depth_metrics[7],
        'valid_rmse_depth_9': depth_metrics[9],
        'valid_rmse_depth_12': depth_metrics[12],
    }
    data_io.save(res, 'task_8')
    return res