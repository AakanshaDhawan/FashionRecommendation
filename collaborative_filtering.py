import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def item_based_filter(df_x, similar_item_number=1):
    """
    for each data point in df we do item based collaborative filtering
    """
    # add a value=1 column to data
    new_df = pd.DataFrame({"user": df_x.iloc[:, 0],
                           "item": df_x.iloc[:, 1],
                           "value": [1] * len(df_x)})

    # if there are more than one purchases just drop them
    new_df.drop_duplicates(inplace=True)

    # replace nans with 0 to convert df to item-by-user matrix
    df_item_user = new_df.pivot(index=new_df.columns[1], columns=new_df.columns[0])['value'].fillna(0).astype('int8')

    # df for articles with a serial index
    df_article_index = pd.DataFrame({'article': df_item_user.index})

    # convert df_item_user to numpy array
    array = np.array(df_item_user)

    # cosine similarity
    cos_sim = cosine_similarity(array)

    # all the unique customer ids
    c_ids = new_df.user.unique()

    # dict to save customer_id and recommended article_ids
    dict_recommendation = {}

    for c in c_ids:

        # all items that c customer purchased
        item_purchased = list(new_df.loc[new_df.user == c].item.unique())

        # list to save similar score for each item
        item_similar = []

        # for each id find similar items' similarities
        for item in item_purchased:

            # article_id's index
            index_item = np.where(df_article_index.article == item)[0]

            if index_item.size == 1:
                index_item = index_item[0]
                # getting id's similarity vectors  and appending it to list
                item_similar.append(cos_sim[index_item])

        # if there are no similar items go to next user
        if len(item_similar) == 0:
            break

        item_similar = np.asarray(item_similar)

        # if similarity if close to 1 replace it with 0 as it will the item itself
        item_similar[item_similar > 0.99] = 0

        # 10 highest similar items for each purchased item
        highest_similar = np.argpartition(item_similar, -10)[:, -10:].flatten()

        # Get similarity using above index info
        high_similarity = np.take(item_similar, highest_similar).flatten()

        # final df
        df_item_index_sim = pd.DataFrame({"item_index": highest_similar, "similarity": high_similarity})

        # sorting by similarity and getting top 10
        df_item_index_sim = df_item_index_sim.sort_values('similarity', ascending=False).reset_index(drop=True)[0:12]

        # get article id using index
        article_ids_recommended = list(df_article_index.article[list(df_item_index_sim.item_index)])

        # used_id and their recommended items
        dict_recommendation[c] = article_ids_recommended

    return dict_recommendation


def user_based_filter(df_x, similar_user_number=1, max_items=None):
    """
    for each user in df we do user based collaborative filtering
    """

    # add a value=1 column to data
    new_df = pd.DataFrame({"user": df_x.iloc[:, 0],
                           "item": df_x.iloc[:, 1],
                           "value": [1] * len(df_x)})

    # dropping duplicate
    new_df.drop_duplicates(inplace=True)

    # replace nans with 0 to convert df to user-by-item matrix
    df_user_item = new_df.pivot(index=new_df.columns[0], columns=new_df.columns[1])['value'].fillna(0).astype('int8')

    # convert df_user_item to numpy matrix
    array = np.array(df_user_item)

    # cosine similarity
    cos_sim = cosine_similarity(array)

    # sorting indexes of user by similarity
    sorted_index = np.fliplr(np.argsort(cos_sim))

    total_users = array.shape[0]

    item_list = list(df_user_item.columns)

    dict_recommendation = {}

    # for each  user
    for i in range(total_users):
        u_id = df_user_item.index[i]

        items_recommended = []

        # Look at similar users
        for j in range(similar_user_number):
            not_purchased = array[sorted_index[i, j + 1]] - array[i]

            # check out item names
            for k in np.where(not_purchased == 1)[0]:
                if item_list[k] not in items_recommended:
                    items_recommended.append(item_list[k])
                    if max_items == None:
                        continue
                    elif len(items_recommended) == max_items:
                        break
            if max_items == None:
                continue
            elif len(items_recommended) == max_items:
                break

        dict_recommendation[u_id] = items_recommended

        # if items_recommended less than max_items
        # if max_items != None and len(items_recommended) < max_items:
        #     print(u_id, "has items less than", max_items)

    return dict_recommendation


def average_precision(x):
    """
    average precision of each user
    """
    x = np.array(x)
    k = len(x)
    precision = np.sum(np.cumsum(x) / (np.arange(k) + 1))

    return precision / k


def mean_average_precision(X, y):
    '''
    A function to calculate Mean Average Precision:
    X: dict having ids and their recommended values
    y: two column df 1st column is user id; 2nd is item id
    return: mean_average_precision, user_number
    '''
    user_number = 0
    no_of_unique_users = y.iloc[:, 0].unique()
    # average precision of each user
    avg_per_user = []
    # for each user
    for user, items in tqdm(X.items()):
        # when found in df
        if user in no_of_unique_users:
            user_number += 1
            # actual purchased items
            subset = y.loc[y.iloc[:, 0] == user]
            items_purchased = set(subset.iloc[:, 1])
            purchased = [1 if item in items_purchased else 0 for item in items]

            # calculate Average Precision
            if len(purchased) > 0:
                avg_per = average_precision(purchased)
            else:
                avg_per = 0
            # append to AP_res
            avg_per_user.append(avg_per)

    if len(avg_per_user) > 0:
        return sum(avg_per_user) / len(avg_per_user), user_number
    else:
        return 0, user_number
     


def average_of_precision(X, y):
    """
    A function to calculate Average of Precision:
    count of purchased recommended items / total recommended items
    X: dict having ids and their recommended values
    y: two column df 1st column is user id; 2nd is item id
    return: average_of_precision, user_number
    """
    user_number = 0

    no_of_unique_users = y.iloc[:, 0].unique()

    # store average precision for each target user
    avg_percision_per_user = []

    # for each target user
    for user, items in tqdm(X.items()):

        # when found in test data
        if user in no_of_unique_users:
            user_number += 1

            # get actual purchased items from y
            subset = y.loc[y.iloc[:, 0] == user]
            items_purchased = set(subset.iloc[:, 1])

            purchased = []

            # for each item
            for item in items:

                # if actually purchased
                if item in items_purchased:
                    purchased.append(1)
                # if not purchased
                else:
                    purchased.append(0)

            avg_per = 0

            #  average of precision
            try:
                if len(purchased) > 0:
                    avg_per = sum(purchased) / len(purchased)
            except:
                print("division by 0")
            avg_percision_per_user.append(avg_per)

    print("Number of users:", user_number)

    # aop =0

    if len(avg_percision_per_user) > 0:
        return sum(avg_percision_per_user) / len(avg_percision_per_user), user_number
    else:
        return 0, user_number
