import recordlinkage
from recordlinkage.datasets import load_febrl1

dfA = load_febrl1()

print(dfA.head())

indexer = recordlinkage.FullIndex()
pairs = indexer.index(dfA)

print (len(dfA), len(pairs))
# (1000*1000-1000)/2 = 499500

indexer = recordlinkage.BlockIndex(on='given_name')
pairs = indexer.index(dfA)

print (len(pairs))

# This cell can take some time to compute.
compare_cl = recordlinkage.Compare()

compare_cl.exact('given_name', 'given_name', label='given_name')
compare_cl.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('suburb', 'suburb', label='suburb')
compare_cl.exact('state', 'state', label='state')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

features = compare_cl.compute(pairs, dfA)

print(features.head(10))

print(features.describe())

# Sum the comparison results.
print(features.sum(axis=1).value_counts().sort_index(ascending=False))

matches = features[features.sum(axis=1) > 3]

print(len(matches))
matches.head(10)