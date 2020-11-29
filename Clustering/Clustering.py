# Databricks notebook source
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext

schema = StructType([ StructField("c"+str(i),DoubleType()) for i in range(7)])
# schema to cast data, can use inferschema also

raw_data = sqlContext.read.csv("/FileStore/tables/Task2/clustering.csv",schema=schema)
raw_data_nn = raw_data.dropna()
# remove null

assembler=VectorAssembler(inputCols=["c"+str(i) for i in range(7)],outputCol="features")
data=assembler.transform(raw_data_nn)
data.show()

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import BisectingKMeans

x, y1, y2 = [], [], []
for k in range(2, 27):
  bkm = BisectingKMeans(k = k)
  model = bkm.fit(data)
  # Make predictions
  predictions = model.transform(data)
  evaluator = ClusteringEvaluator()
  silhouette = evaluator.evaluate(predictions)
  x.append(k)
  y1.append(model.summary.trainingCost) #trainingCost is equivalent to sklearn inertia, so the SSE
  y2.append(silhouette)

# COMMAND ----------

import matplotlib.pyplot as plt

max_silhouette = x[y2.index(max(y2))]

f = plt.figure(figsize=(10,15))
ax = f.add_subplot(211)
ax2 = f.add_subplot(212)
ax.plot(x, y1, 'bx-')
ax.axvline(x=max_silhouette, c="red")
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Sum of SSE')
ax.set_title('Elbow Curve')

ax2.plot(x, y2, 'bx-')
ax2.axvline(x=max_silhouette, c="red")
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette with squared eclidean distance')
ax2.set_title('Silhouette Curve')

# COMMAND ----------

print("Maximum of silhouette curve : {}".format(max_silhouette))

# COMMAND ----------

# DBTITLE 1,From the elbow method and the silhouette method, it seems that the optimal k for this dataset is 5
best_kmeans = BisectingKMeans(k = 5)
best_model = best_kmeans.fit(data)
# Make predictions
best_predictions = best_model.transform(data)
best_predictions = best_predictions.select("prediction")

best_predictions.show()

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import StringType

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

#for predict in best_predictions.collect():
#  letters_predict.append(letters[predict.prediction])
  
#print(letters_predict)

def get_letter(letter):
  return letters[letter]

#convert to a UDF Function by passing in the function and return type of function. UDF Function is equivalent to map function on dataframe in a way.
get_letter_func = F.udf(get_letter, StringType())

best_predictions_letters = best_predictions.withColumn("letter", get_letter_func("prediction")).select("letter")
best_predictions_letters.show()

# COMMAND ----------

display(best_predictions_letters)

# COMMAND ----------


