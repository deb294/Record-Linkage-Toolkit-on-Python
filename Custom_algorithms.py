import pandas

import recordlinkage
from recordlinkage.datasets import load_febrl4





from recordlinkage.base import BaseIndexator
class FirstLetterIndex(BaseIndexator):
    """Custom class for indexing"""

    def __init__(self, letter):
        super(FirstLetterIndex, self).__init__()

        # the letter to save
        self.letter = letter

    def _link_index(self, df_a, df_b):
        """Make record pairs that agree on the first letter of the given name."""

        # Select records with names starting with a 'letter'.
        a_startswith_w = df_a[df_a['given_name'].str.startswith(self.letter) == True]
        b_startswith_w = df_b[df_b['given_name'].str.startswith(self.letter) == True]

        # Make a product of the two numpy arrays
        return pandas.MultiIndex.from_product(
            [a_startswith_w.index.values, b_startswith_w.index.values],
            names=[df_a.index.name, df_b.index.name]
        )
def compare_zipcodes(s1, s2):
    """
    If the zipcodes in both records are identical, the similarity
    is 0. If the first two values agree and the last two don't, then
    the similarity is 0.5. Otherwise, the similarity is 0.
    """

    # check if the zipcode are identical (return 1 or 0)
    sim = (s1 == s2).astype(float)

    # check the first 2 numbers of the distinct comparisons
    sim[(sim == 0) & (s1.str[0:2] == s2.str[0:2])] = 0.5

    return sim

def compare_addresses(s1_1, s1_2, s2_1, s2_2):
    """
    Compare addresses. Compare address_1 of file A with
    address_1 and address_2 of file B. The same for address_2
    of dataset 1.

    """

    return ((s1_1 == s2_1) | (s1_2 == s2_2) | (s1_1 == s2_2) | (s1_2 == s2_1)).astype(float)




df_a, df_b = load_febrl4()
indexer = FirstLetterIndex('w')
candidate_pairs = indexer.index(df_a, df_b)

print ('Number of record pairs (letter w):', len(candidate_pairs))
for letter in 'wxa':

    indexer = FirstLetterIndex(letter)
    candidate_pairs = indexer.index(df_a, df_b)

    print ('Number of record pairs (letter %s):' % letter, len(candidate_pairs))

# Make an index of record pairs
pcl = recordlinkage.BlockIndex('given_name')
candidate_pairs = pcl.index(df_a, df_b)

comparer = recordlinkage.Compare()
comparer.compare_vectorized(compare_zipcodes, 'postcode', 'postcode', label='sim_postcode')
features = comparer.compute(candidate_pairs, df_a, df_b)

print(features['sim_postcode'].value_counts())

comparer = recordlinkage.Compare()

# naive
comparer.exact('address_1', 'address_1', label='sim_address_1')
comparer.exact('address_2', 'address_2', label='sim_address_2')

# better
comparer.compare_vectorized(
    compare_addresses,
    ('address_1', 'address_2'), ('address_1', 'address_2'),
    label='sim_address'
)

features = comparer.compute(candidate_pairs, df_a, df_b)

print(features.mean())