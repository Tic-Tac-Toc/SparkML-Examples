# Databricks notebook source
# DBTITLE 1,Task 1 : Classification
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

letters_table = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "?"]

def letter_to_int(letters):
  transform_values = []
  for letter in letters:
    if (letter not in letters_table): transform_values.append(int(letter))
    else : transform_values.append(letters_table.index(letter))
  return transform_values

data = sc.textFile("/FileStore/tables/Task1/trainingData.csv")
data = data.map(lambda x: x.split(','))
data = data.map(letter_to_int)

schema = StructType([ StructField("c"+str(i),IntegerType()) for i in range(22)] + [StructField("label", IntegerType())])
# schema to cast data, can use inferschema also

data = sqlContext.createDataFrame(data, schema)
data = data.dropna()

assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(22)],outputCol="features")
data=assembler.transform(data)
data.show()

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Split the data into training and test sets (10% held out for testing)
(trainingData, testData) = data.randomSplit([0.9, 0.1])
# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

# Train model.  This also runs the indexers.
model = dt.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName='areaUnderROC')
result = evaluator.evaluate(predictions)

print("Area under ROC curve = {}".format(result))

# COMMAND ----------

# DBTITLE 1,We will know try to improve our model by using a cross validation.
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder

dtparamGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, range(3, 8))
             .addGrid(dt.maxBins, range(4, 16, 4))
             .build())

dtcv = CrossValidator(estimator = dt,
                      estimatorParamMaps = dtparamGrid,
                      evaluator = evaluator,
                      numFolds = 5)

cv_model = dtcv.fit(trainingData)
cv_predictions = cv_model.transform(testData)
print("Area under ROC curve : {}".format(evaluator.evaluate(cv_predictions)))

# COMMAND ----------

print("Best model")
print("=" * 30)
print(cv_model.bestModel.toDebugString)

# COMMAND ----------

# DBTITLE 1,Now, we will create predictions.csv based on predictData.csv
predict_schema = StructType([ StructField("c"+str(i),IntegerType()) for i in range(22)])
# schema to cast data, can use inferschema also

predict_data = sc.textFile("/FileStore/tables/Task1/predictData.csv")
predict_data = predict_data.map(lambda x: x.split(','))
predict_data = predict_data.map(letter_to_int)

predict_data = sqlContext.createDataFrame(predict_data, predict_schema)
predict_data = predict_data.dropna()

assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(22)],outputCol="features")
predict_data=assembler.transform(predict_data)
predict_data.show()

# COMMAND ----------

final_predictions = cv_model.transform(predict_data).select("prediction")
final_predictions.show()

# COMMAND ----------

display(final_predictions)

# COMMAND ----------


