
# Functions to perform statistical analysis

def rank_values(scores):
    """
        Ranks "scores" by repacing absolute score with the ranking

        :param dict scores: dict of absolute score values

        :returns dict: has same keys as "scores" but with values replaced by
                       a rank
    """
    if (not isinstance(scores, dict) or not len(scores) 
            or not all([isinstance(k, str) for k in scores])
            or not all([isinstance(v, (str, int, float))
                        for v in scores.values()])
            or any([isinstance(v, bool) for v in scores.values()])):
        raise TypeError("rank_values() called with bad args")

    # Separate valid scores from failures which are indicated by str typevalues

    valid = sorted([(s, a) for a, s in scores.items()
                    if not isinstance(s, str)], reverse=True)
    previous_score = None
    rank = 0
    tie_count = 1
    ranked = {}
    for score, algo in valid:
        if score != previous_score:
            rank += tie_count
            tie_count = 1
        else:
            tie_count += 1

        # print(algo, rank)
        ranked[algo] = rank
        previous_score = score

    if len(scores) > len(valid):
        ranked.update({a: len(valid) + 1 for a, s in scores.items()
                       if isinstance(s, str)})
    
    return ranked
