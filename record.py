import pandas
import recordlinkage
from recordlinkage.datasets import load_febrl4

dfA, dfB = load_febrl4()

#dfA = load_febrl1()

# Indexation step
indexer = recordlinkage.FullIndex()
#indexer = recordlinkage.BlockIndex(on='given_name')
pairs = indexer.index(dfA, dfB)

print (len(dfA), len(dfB), len(pairs))

indexer = recordlinkage.BlockIndex(on='given_name')
pairs = indexer.index(dfA, dfB)

print (len(pairs))


# Comparison step
compare_cl = recordlinkage.Compare()

compare_cl.exact('given_name', 'given_name', label='given_name')
compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('suburb', 'suburb', label='suburb')
compare_cl.exact('state', 'state', label='state')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

features = compare_cl.compute(pairs, dfA, dfB)


#print (features)


#print (features.describe())

# Sum the comparison results.
print(features.sum(axis=1).value_counts().sort_index(ascending=False))

# Classification step

#print(features[features.sum(axis=1) > 3])
matches = features[features.sum(axis=1) > 3]
print(len(matches))







