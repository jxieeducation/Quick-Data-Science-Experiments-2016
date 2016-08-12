from traildb import TrailDB

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

def get_events(tdb):
    query = [('title', 'Prince (musician)')]
    for i in range(len(tdb)):
        events = list(tdb.trail(i, event_filter=query))
        if events:
            yield events[0].time, events

def get_dataframe():
    tdb = TrailDB('pydata-tutorial.tdb')
    base = tdb.min_timestamp()
    types = []
    xs = []
    ys = []
    #try this:
    #for y, (first_ts, events) in enumerate(sorted(get_events(tdb), reverse=True)):
    for y, (first_ts, events) in enumerate(get_events(tdb)):
        for event in events:
            xs.append(int(event.time - base) / (24 * 3600))
            ys.append(y)
            types.append('user' if event.user else 'anon')
    data = pd.DataFrame({'x': xs, 'y': ys})
    data['type'] = pd.Series(types, dtype='category')
    return data

cnv = ds.Canvas(400, 300)
agg = cnv.points(get_dataframe(), 'x', 'y', ds.count_cat('type'))
colors = {'anon': 'red', 'user': 'blue'}
img=tf.set_background(tf.colorize(agg, colors, how='eq_hist'), 'white')
with open('prince.png', 'w') as f:
    f.write(img.to_bytesio().getvalue())
