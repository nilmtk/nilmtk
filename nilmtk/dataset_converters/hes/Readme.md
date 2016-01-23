There are a couple of small fixes/issues with the current converter. Till we get a chance to fix them in the main converter, please use the following fix.

````python
st = pd.HDFStore("/data/hes/hes_3.h5")
keys = st.keys()
for key in keys:
    df = st[key]
    if not df.index.is_monotonic:
        df = df.sort()
        st.put(key, df, format='table')
st.close()
```

This will sort the meters in ascending order.

There is also an issue with data from one of the home (nilmtk id #234). For now, the quick fix would be to ignore this home in the analysis.

Some exploratory analysis of the dataset can be found on [Nipun's blog](http://nipunbatra.github.io/2016/01/nilmtk-hes/)
