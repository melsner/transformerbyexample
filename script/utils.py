CACHE = {}
def edist(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if (s1, s2) in CACHE:
        return CACHE[(s1, s2)]

    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 2

    op1 = edist(s1[:-1], s2) + 1
    op2 = edist(s1, s2[:-1]) + 1
    op3 = edist(s1[:-1], s2[:-1]) + cost
    mincost = min(op1, min(op2, op3))
    CACHE[(s1, s2)] = mincost
    return mincost

CACHE_ALT = {}
def edist_alt(s1, s2):
    if len(s1) == 0:
        return len(s2), (tuple(), tuple([(char, False) for char in s2]))
    if len(s2) == 0:
        return len(s1), (tuple([(char, False) for char in s1]), tuple())

    if (s1, s2) in CACHE_ALT:
        return CACHE_ALT[(s1, s2)]

    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 2

    op1, sol1 = edist_alt(s1[:-1], s2)
    op1 += 1
    op2, sol2 = edist_alt(s1, s2[:-1])
    op2 += 1
    op3, sol3 = edist_alt(s1[:-1], s2[:-1])
    op3 += cost

    mincost = min(op1, min(op2, op3))
    if op1 == mincost:
        solution1, solution2 = sol1
        solution = (solution1 + ((s1[-1], False),), solution2)
    elif op2 == mincost:
        solution1, solution2 = sol2
        solution = (solution1, solution2 + ((s2[-1], False),))
    else:
        solution1, solution2 = sol3
        solution = (solution1 + ((s1[-1], cost == 0),), solution2 + ((s2[-1], cost == 0),))

    CACHE_ALT[(s1, s2)] = (mincost, solution)
    return mincost, solution
