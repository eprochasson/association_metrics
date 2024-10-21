"""
Quick and dirty association metrics on simple baskets.

E. Prochasson, 2024.
MIT
"""

import math
from typing import List, Dict, Set

Item = str
Basket = Set[Item]  # In this example, we don't consider quantity / item / basket, but just the presence / absence of
                    # an item in a given basket. Considering quantities may be relevant in some applications,
                    # but likely not in most.
ITEMS: List[Item] = ['Car', 'Tennis Ball', 'Tennis Racket', 'Laundry Detergent', 'Softener', 'Flour', 'Milk',
                     'Screwdriver',
                     'Toothpaste']


class ContingencyTable:
    """
    Compute and store a contingency table, that is, counts of co-occurrence between i and j
    Cells record the count of coocurrences. ¬x indicates everything but x. Marginal values C_x, R_x and N are the sum of
    the rows / columns / everything

                 j  |  ¬j   |
    --------------------------------
      i    |  O_11  | O_12  |    R_1
     ¬i    |  O_21  | O_22  |    R_2
           |    C1  |   C2  |      N
    """


    def __init__(self, i: Item, j: Item, baskets: List[Basket]):
        self.i = i
        self.j = j
        self.baskets = baskets

        # how many times i and j appear in the same basket
        self.O_11 = sum([1 for b in baskets if i in b and j in b])
        # How many times i appears without j
        self.O_12 = sum([1 for b in baskets if i in b and j not in b])
        # How many times j appears without i
        self.O_21 = sum([1 for b in baskets if i not in b and j in b])
        # How many baskets without i or j
        self.O_22 = sum([1 for b in baskets if i not in b and j not in b])

        # Marginal values:
        self.C_1 = self.O_11 + self.O_21
        self.C_2 = self.O_12 + self.O_22
        self.R_1 = self.O_11 + self.O_12
        self.R_2 = self.O_21 + self.O_22

        # Totals. Should be strictly equal for all contingency table on a given basket corpus.
        # Can be calculated many ways.
        self.N = self.C_1 + self.C_2

        # Expectations under null-hypothesis
        self.E_11 = self.R_1 * self.C_1 / self.N
        self.E_21 = self.R_2 * self.C_1 / self.N
        self.E_12 = self.R_1 * self.C_2 / self.N
        self.E_22 = self.R_2 * self.C_2 / self.N

    def mutual_information(self):
        """From the definition,

        MI(i, j) = \sum_{i, j} O_ij log_2 \frac{O_ij}{E_ij}


        That is, some log-dampened ratio of what we observed in the corpus, vs. expectation if basket were randomly and
        uniformly assembled, for each cells in the contingency table. It's a generalization of the
        local_mutual_information using much more of the available information.
        """
        # We calculate each element of the sum separately, to avoid passing a 0 value to the log.
        x11 = self.O_11 * math.log2(self.O_11 / self.E_11) if self.O_11 > 0 else 0
        x12 = self.O_12 * math.log2(self.O_12 / self.E_12) if self.O_12 > 0 else 0
        x21 = self.O_21 * math.log2(self.O_21 / self.E_21) if self.O_21 > 0 else 0
        x22 = self.O_22 * math.log2(self.O_22 / self.E_22) if self.O_22 > 0 else 0
        # print(x11, x12, x21, x22)
        return x11 + x12 + x21 + x22


    def log_likelihood(self):
        """
        Eerily similar to the mutual information, this is also a fantastic approximation of
        Fisher's Exact Test, which, as the name implies, is exact, but also computationally intractable.

        Unlike MI, it uses the natural log (and not log_2).

        ll(i, j) = 2 * \sum_{i, j} O_ij log \frac{O_ij}{E_ij}
        """
        # We calculate each element of the sum separately, to avoid passing a 0 value to the log.
        x11 = self.O_11 * math.log(self.O_11 / self.E_11) if self.O_11 > 0 else 0
        x12 = self.O_12 * math.log(self.O_12 / self.E_12) if self.O_12 > 0 else 0
        x21 = self.O_21 * math.log(self.O_21 / self.E_21) if self.O_21 > 0 else 0
        x22 = self.O_22 * math.log(self.O_22 / self.E_22) if self.O_22 > 0 else 0

        return 2 * (x11 + x12 + x21 + x22)

    def __str__(self) -> str:
        """Pretty-ish print of the main contingency table"""
        return f"""
    Observations:
{len(self.i) * ' '} | {self.j} | ¬{self.j} |
 {self.i} | {self.O_11} | {self.O_12} | {self.R_1}
¬{self.i} | {self.O_21} | {self.O_22} | {self.R_2}
 {len(self.i) * ' '} | {self.C_1} | {self.C_2} | {self.N}
    Expectations:
{len(self.i) * ' '} | {self.j} | ¬{self.j} |
 {self.i} | {self.E_11} | {self.E_12} 
¬{self.i} | {self.E_21} | {self.E_22} 
"""


def local_mutual_information(i: Item, j: Item, baskets: List[Basket]) -> float:
    """
    "Simple" Mutual information. Look at the ratio of co-occurrence observed vs. co-occurrence expected under the
    null hypothesis that items are randomly (and uniformly) distributed among baskets.

    If local_mutual_information(i, j) = 0, this means the 2 items are not associated (buying one says nothing about
    the likelihood of buying the other).
    If lmi(i, j) > 0, items are positively associated (buying one predicts the purchase of the other)
    if lmi(i, j) < 0, items are negatively associated (buying one indicates it's less likely the other will be purchased).

    This measure does not take into account the size of the measured samples, and will return the same results for
    O = 2 and E = 1, than for O = 2*10^25 and E = 10^25. It will therefore kind of break for rarely purchased items.

    :param i: Item
    :param j: Item
    :param baskets: all the baskets considered
    :return: float
    """
    # Observed co-occurrences:
    O = sum([1 for b in baskets if i in b and j in b ])
    if O == 0:
        return 0
    freq_i = sum([1 for b in baskets if i in b])
    freq_j = sum([1 for b in baskets if j in b])
    N = sum([len(b) for b in baskets])

    E = freq_i * freq_j / N

    return math.log2(O / E)



if __name__ == '__main__':

    # Manually generated basket. I got lazy and copy/pasted half the list.
    # The intent is to have products that are very common, and bought in many baskets (toothpaste, milk),
    # as well as have some products that are tightly associated (milk/flour and laundry stuff).
    # This is just good enough to showcase the metrics, but they would work much better on real life data
    # (provided I didn't screw up the implementation).
    baskets = [{i for i in ITEMS},   # so that each item is paired with all the other items at least once.
               {'Car', 'Toothpaste'},
               {'Tennis Ball', 'Tennis Racket', 'Toothpaste'},
               {'Laundry Detergent', 'Softener'},
               {'Laundry Detergent', 'Softener'},
               {'Laundry Detergent', 'Softener', 'Milk'},
               {'Flour', 'Milk'},
               {'Flour', 'Milk'},
               {'Toothpaste', 'Flour', 'Laundry Detergent'},
               {'Screwdriver', 'Car', 'Milk'},
               {'Toothpaste', 'Milk'},
               {'Toothpaste', 'Milk'},
               {'Milk', 'Flour'},
               {'Car', 'Screwdriver'},
               {'Car', 'Toothpaste'},
               {'Tennis Ball', 'Tennis Racket', 'Toothpaste'},
               {'Laundry Detergent', 'Softener'},
               {'Laundry Detergent', 'Softener'},
               {'Laundry Detergent', 'Softener', 'Milk'},
               {'Flour', 'Milk'},
               {'Flour', 'Milk'},
               {'Toothpaste', 'Flour', 'Laundry Detergent'},
               {'Screwdriver', 'Car', 'Milk'},
               {'Toothpaste', 'Milk'},
               {'Toothpaste', 'Milk'},
               {'Milk', 'Flour'},
               {'Car', 'Screwdriver'},
               {'Toothpaste', 'Milk'},
               {'Toothpaste', 'Milk'},
               {'Toothpaste', 'Milk'}]

    # Check that we didn't misspell any items
    assert(all([c in ITEMS for b in baskets for c in b]))

    # In the following, we use the i < j condition in the list comprehensions because all the metrics are symetrical and
    # don't need to be calculated both ways.

    # Local mutual information: Simple but not very smart.
    lmi = {(i, j): local_mutual_information(i, j, baskets) for i in ITEMS for j in ITEMS if i < j}
    list(reversed(sorted([(i, j, v) for (i, j), v in lmi.items()], key=lambda x: x[2])))
    # From the above list, we can see that:
    # - Tennis Ball predicts the purchase of Tennis Racket (and vice-versa), as well as car predicts the purchase of a screwdriver
    # - If you buy softener, you're less likely to buy toothpaste (although it's weak).
    # - Milk and Toothpaste are frequent items, most of the time their purchase is not strongly associated with another product
    # - Screwdrivers and Toothpaste are bought together pretty much as expected if people were to randomly put stuff in their basket, and are therefore not associated (close to 0).

    # Using contingency table: smarter, but costly.

    # Calculate contingency tables for all pair of items
    ct = {(i,j): ContingencyTable(i, j, baskets) for i in ITEMS for j in ITEMS if i < j}

    # Then it's easy to calculate the other measures.
    # generalized MI
    gmi = {(i, j): ct[(i, j)].mutual_information() for i in ITEMS for j in ITEMS if i < j}
    list(reversed(sorted([(i, j, v) for (i, j), v in gmi.items()], key=lambda x: x[2])))
    # This shows:
    # - Detergent/softener, Tennis racket/ball and car/screwdriver are very associated
    # - Milk and screwdrivers really are not
    # - Everything else is in between.
    #
    # In practice, one would use this to answer the question "given I see the purchase X,
    # what are the top 5 items I should recommend, and how good is the recommendation".

    # log-likelihood
    ll = {(i, j): ct[(i, j)].log_likelihood() for i in ITEMS for j in ITEMS if i < j}
    list(reversed(sorted([(i, j, v) for (i, j), v in ll.items()], key=lambda x: x[2])))
    # This exhibits the same outcome as the generalized mutual information (which is good news).
