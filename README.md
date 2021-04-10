# recommenderSystem

Three Python scripts included in this repository for recommender system:
1. Sentiment Analysis
2. Recommender System 1
3. Recommender System 2

1. Sentiment Analysis
- Turning text in to numeric score, >0 = positive, <0 = negative
- Using Textblob library
+ able to translate Chinese and Malay into English for sentiment Analysis

2. Recommeder System 1
- Calculate consine similarity map of given dataset
- input 1. User ID, 2. Movie ID, 3. Movie rating and feed to SVD for trainning
- return with similarity score
- using movie title dataset
- suggestion: observe the user and current viewing material to get the recommended materials  

3. Recommender System 2
- item-item collaborative filtering: recommend material by material attributes(year, subject)
- using cosine similarity to calcuate the nearest matching score
+ fuzzywuzzy library to corrent the typo or nearest input text 
- return score of the highest similarity as recommended item
