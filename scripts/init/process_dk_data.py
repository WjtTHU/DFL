import pandas as pd

print("Reading CSV...")
df = pd.read_csv("DK_wip.csv")  # path to the csv file of DK

pairs = df[["LotId", "parentLotId"]].drop_duplicates()
parent_map = dict(zip(pairs["LotId"], pairs["parentLotId"], strict=True))


def find_root(lot_id):
    visited = set()
    current = lot_id
    while current in parent_map and current not in visited:
        visited.add(current)
        p = parent_map[current]
        if p == current:
            return current
        current = p
    return current


print("Finding roots for all unique LotIds...")
unique_lots = df["LotId"].unique()
root_map = {lid: find_root(lid) for lid in unique_lots}

print("Replacing parentLotId with root LotId...")
df["parentLotId"] = df["LotId"].map(root_map)

output_path = "outputs/DK_wip_root.csv"  # path to the csv file of DK with root LotId
print("Writing to %s ..." % output_path)
df.to_csv(output_path, index=False)
print("Done! Shape: %s" % str(df.shape))
