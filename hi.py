import recordlinkage
from recordlinkage.datasets import load_febrl4
from recordlinkage.datasets import load_febrl1
from recordlinkage.datasets import load_krebsregister

import pandas



df_a = pandas.DataFrame(load_febrl4)
df_b = pandas.DataFrame(load_febrl1)
#dfA, dfB = load_febrl4()

block_class = recordlinkage.BlockIndex('surname')
candidate_links = block_class.index(df_a, df_b)

c = recordlinkage.Compare()

c.string('name_a', 'name_b', method='jarowinkler', threshold=0.85)
c.exact('sex', 'gender')
c.date('dob', 'date_of_birth')
c.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7)
c.exact('place', 'placename')
c.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5)

# The comparison vectors
feature_vectors = c.compute(candidate_links, df_a, df_b)


# Initialize the classifier
true_linkage = pandas.Series(YOUR_GOLDEN_DATA, index=pandas.MultiIndex(YOUR_MULTI_INDEX))

logrg = recordlinkage.LogisticRegressionClassifier()
logrg.learn(feature_vectors[true_linkage.index], true_linkage)

logrg.predict(feature_vectors)


logrg.predict(feature_vectors)
ecm = recordlinkage.ECMClassifier()
ecm.learn(feature_vectors)
