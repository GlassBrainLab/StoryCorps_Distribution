"""
Example of running Zelasko et al.'s DASeg model directly, without the GBL pipeline.
"""
from transformers import pipeline

# initialize the pipeline for token classification
pipe = pipeline("token-classification", model="pzelasko/longformer-swda-nolower")

text = "Your example dialogue text here."

results = pipe(text)

print(results)