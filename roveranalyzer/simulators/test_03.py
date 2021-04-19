import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(100000, 3), columns=list("ABC"))
df["one"] = "foo"
df["two"] = "bar"
df.loc[50000:, "two"] = "bah"
mi = df.set_index(["one", "two"])
print(mi)

store = pd.HDFStore("test.h5", mode="w")
store.append("df", mi)

print(store.get_storer("df").levels)
print(store)

print(store.select("df", columns=["one"]))
print(store.select("df", columns=["A"]))
print(store.select_column("df", "one"))


def f():
    level_1 = store.select_column("df", "one")
    level_2 = store.select_column("df", "two")
    return pd.MultiIndex.from_arrays([level_1, level_2])
