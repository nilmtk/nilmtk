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

This will sort the 
