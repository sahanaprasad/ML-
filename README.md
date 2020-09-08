# ML Based Format Mapping
MultinomialNB algorithm in Naive byes is used to classify the sourceField
ngram model is used to check the text similarity
drivercode.py contains the driver code of the service
All the APIs are written in flask webframework


API 1
takes the source and target fields, maps the respective fields and also determines the confidence leve. 
More the input data, more will be the confidence.

API 2 
the model learns the input data


API 3
This API is used to map the sourceField to particular target field. Model has learned the mappings before hand. TH
In order to increase the value, the sample data is used to train the model. This sample data is tored in csv file with two columns : title, and category, which refers to the source and target fields respectively.
