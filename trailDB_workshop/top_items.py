from traildb import TrailDB
from collections import Counter
import timeit

def string_top():
    tdb = TrailDB('pydata-tutorial')
    return Counter(event.title for uuid, trail in tdb.trails()
                               for event in trail).most_common(5)

def item_top():
    tdb = TrailDB('pydata-tutorial')
    stats = Counter(event.title for uuid, trail in tdb.trails(rawitems=True)
                                for event in trail)
    return [(tdb.get_item_value(item), f) for item, f in stats.most_common(5)]

print 'string_top', timeit.timeit(string_top, number=3)
print 'item_top', timeit.timeit(item_top, number=3)