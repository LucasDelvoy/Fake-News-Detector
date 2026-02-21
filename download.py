import pandas as pd

df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# 2. Créer la colonne 'label' pour chaque fichier
df_fake["label"] = "0"
df_real["label"] = "1"

# 3. Fusionner les deux (on les empile l'un sur l'autre)
df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

# 4. Mélanger les lignes (optionnel mais conseillé pour les stats)
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("news.csv", index=False)