# Databricks notebook source
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext

schema = StructType([ StructField("c"+str(i),IntegerType()) for i in range(8)])
# schema to cast data, can use inferschema also

raw_data = sqlContext.read.csv("/FileStore/tables/Task3/regression.csv",schema=schema)
raw_data_nn = raw_data.dropna()
# remove null

assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(7)],outputCol="features")
data=assembler.transform(raw_data_nn)
data.show()

# COMMAND ----------

(trainingData, testData) = data.randomSplit([0.9, 0.1])

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Train a DecisionTree model.
dt = DecisionTreeRegressor(labelCol="c7", featuresCol="features")
# Train model.  This also runs the indexer.
model = dt.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="c7", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = {}".format(rmse))

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

dtparamGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 5, 10, 20, 30])
             .addGrid(dt.maxBins, [16, 32, 64, 128, 256])
             .build())

dtcv = CrossValidator(estimator = dt,
                      estimatorParamMaps = dtparamGrid,
                      evaluator = evaluator,
                      numFolds = 5)

cv_model = dtcv.fit(trainingData)
cv_predictions = cv_model.transform(testData)
rmse = evaluator.evaluate(cv_predictions)
print("Root Mean Squared Error (RMSE) on test data = {}".format(rmse))

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol="c7", featuresCol="features")
# Train model.  This also runs the indexer.
model = rf.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
rf_evaluator = RegressionEvaluator(
    labelCol="c7", predictionCol="prediction", metricName="rmse")
rmse = rf_evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

rf_parameter_grid_ = (ParamGridBuilder()
                      .addGrid(rf.maxBins, [25, 28, 31])
                      .addGrid(rf.maxDepth, [4, 6, 8])
                      .build())

rfcv = CrossValidator(estimator = rf,
                      estimatorParamMaps = rf_parameter_grid_,
                      evaluator = rf_evaluator,
                      numFolds = 5)

rf_model = rfcv.fit(trainingData)
rf_predictions = rf_model.transform(testData)
rmse = rf_evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

lr = LinearRegression(featuresCol = 'features', labelCol='c7')
lr_model = lr.fit(trainingData)

lr_predictions = lr_model.transform(testData)
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="c7",metricName="rmse")
print("RMSE = {}".format(lr_evaluator.evaluate(lr_predictions)))

# COMMAND ----------

lr_parameter_grid_ = (ParamGridBuilder()
                      .addGrid(lr.maxIter, [25, 50, 100])
                      .addGrid(lr.regParam, [0.0, 0.1, 0.01])
                      .addGrid(lr.elasticNetParam, [0, 0.1, 0.5, 1])
                      .addGrid(lr.fitIntercept, [False, True])
                      .build())

lrcv = CrossValidator(estimator = lr,
                      estimatorParamMaps = lr_parameter_grid_,
                      evaluator = lr_evaluator,
                      numFolds = 5)

lr_model = lrcv.fit(trainingData)
lr_predictions = lr_model.transform(testData)
print("RMSE = {}".format(lr_evaluator.evaluate(lr_predictions)))

# COMMAND ----------

print("Best model")
print("=" * 30)
print("Reg Param = {}".format(lr_model.bestModel._java_obj.getRegParam()))
print("Max Iter = {}".format(lr_model.bestModel._java_obj.getMaxIter()))
print("Fit intercept = {}".format(lr_model.bestModel._java_obj.getFitIntercept()))
print("Elastic net param = {}".format(lr_model.bestModel._java_obj.getElasticNetParam()))
print("=" * 30)
print("Coefficients: {}".format(lr_model.bestModel.coefficients))
print("Intercept: {}".format(lr_model.bestModel.intercept))

# COMMAND ----------

# DBTITLE 1,Linear regression seems to be the best model here.
predict_schema = StructType([ StructField("c"+str(i),IntegerType()) for i in range(7)])
# schema to cast data, can use inferschema also

predict_raw_data = sqlContext.read.csv("/FileStore/tables/Task3/predict.csv",schema=predict_schema)
predict_raw_data_nn = predict_raw_data.dropna()
# remove null

assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(7)],outputCol="features")
predict_data=assembler.transform(predict_raw_data_nn)
predict_data.show()

# COMMAND ----------

predict_lr_predictions = lr_model.transform(predict_data).select("prediction")
predict_lr_predictions.show()

# COMMAND ----------

display(predict_lr_predictions)

# COMMAND ----------


