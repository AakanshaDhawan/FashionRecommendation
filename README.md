### Project title: H&M Personalized Fashion Recommendations

## How to compile the code: 
  Step 1) Go to the Kaggle page and download the data(articles.csv, customer.csv, transaction_train.csv)
  
  Step 2) Keep all the files in one folder
  
  Step 3) Run the Python notebooks in the following sequence:
  
            a)data analysiss and extraction.ipynb : contains analysis regarding the data
            b) collaborative filtering.ipynb : contains user-based filtering
            c) item-based collaborative filter.ipynb : shows how item-based filtering works
            d) content-based filtering.ipynb : content-based filter is implemented here
            e)comparison.ipynb: this notebook contains the comparison of all the algorithms
            
            
  P.S. Notebooks can be run in any sequence as the data extracted and used for algorithms is present in women_wear_2018_winter.csv 
            

## Description

H & M (Hennes & Mauritz) group launched a competition on Kaggle where given the previous transactions of the customers is given and a product recommendation system has to be built.

The main aim behind this problem is to help customers by predicting what are they more likely to buy within 7 days after the training data ends and more importantly, assisting consumers in making sound decisions benefits sustainability since it lowers returns and, as a result, lowers transportation-related emissions.

## Data Description

### Link to [data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

The data is nearly 35 GB in size and consists of:
  1) articles: metadata about the products that are available for purchase
  2) customer: metadata for each customer
  3) transactions_train: the training data, which includes extra data as well as the purchases made by each consumer on each day. Multiple purchases of the same item                              are represented by duplicate rows.
  
## Algorithms 

  1) Collaborative filtering
      a) User Based
      b) Item Based
  2) Content-based filtering
  
## Libraries used

 1) numpy
 2) pandas
 3) matplotlib
 4) seaborn
 5) sci-kit learn
 6) TensorFlow
 7) PyTorch
  
## Work Done
 1) Data analysis
 2) User-based Collaborative filtering 
 3) Function for Item-based Collaborative filtering
 4) Finding the top N items recommended for each user
 5) Apply Content-based filtering
 6) Apply item-based collaborative filtering
 7) Compare the two
 8) Right now because only small data is being taken two-week categories have been made need to think of a better solution to include more week data

## References
 
 Link to the [competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)
 
 [How to build a recommendation engine](https://www.netguru.com/blog/product-recommendation-machine-learning)
 
 [Product Recommendation engine](https://towardsdatascience.com/what-are-product-recommendation-engines-and-the-various-versions-of-them-9dcab4ee26d5)
 
 [Content Based Filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics)
 
 [Collabrative Filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics)
 
 [Mean Average Precision](https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html)
