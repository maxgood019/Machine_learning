import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)


#list of list
transaction = []
rows , columns = dataset.shape
#list of list
for i in range(rows):
    transaction.append([str(dataset.values[i,j]) for j in range(columns)])
    
from apyori import apriori

# min_support: three times a day, 7 days, 7500 dataset

rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2 )

#visualising result
result = list(rules)


# result[3] :
#RelationRecord(items=frozenset({'fromage blanc', 'honey'}), support=0.003332888948140248, 
#ordered_statistics=[OrderedStatistic(items_base=frozenset({'fromage blanc'}), items_add=frozenset({'honey'}), confidence=0.2450980392156863, lift=5.164270764485569)])



