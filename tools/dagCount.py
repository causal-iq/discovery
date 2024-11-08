
# Compute some example number of DAGS against number of nodes, from
# Robinson, R.W., 1977. Counting unlabeled acyclic digraphs.
# In Combinatorial mathematics V (pp. 28-43). Springer, Berlin, Heidelberg.

from math import factorial


def combinations(n, r):
    return int(factorial(n) / (factorial(r) * factorial(n - r)))


def dagCount(p, counts):
    count = 0
    for s in range(1, p + 1):
        count += (-1)**(s + 1) * \
            combinations(p, s) * 2**(s * (p - s)) * counts[p - s]
    DGcount = int(3**((n * (n - 1)) / 2))
    print(("With {} nodes there are {:.4e} labelled DAGS and {:.4e} DGs " +
          "({:.3e}% acyclic)")
          .format(p, count, DGcount, (count * 100 / DGcount), grouping=True))
    counts.append(count)


counts = [1]
unlabelledDAGs = [1, 1, 2, 6, 31, 302, 5984, 243668, 20286025, 3424938010,
                  1165948612902, 797561675349580, 1094026876269892596]

for n in range(1, 13):
    dagCount(n, counts)
    print(" ... and {:.4e} unlabelled DAGS".format(unlabelledDAGs[n]))
