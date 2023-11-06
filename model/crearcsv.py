import pandas as pd

df = pd.read_csv('data/milknew.csv')

# Identificar las columnas categóricas
categorical_columns = ['Grade']

# Crear un mapeo de categorías únicas a números para cada columna categórica
category_mappings = {}
for column in categorical_columns:
    category_mappings[column] = {category: index for index, category in enumerate(df[column].unique())}

# Aplicar la codificación de categorías a cada columna categórica
for column in categorical_columns:
    new_column_name = f"{column}_category"
    df[new_column_name] = df[column].map(category_mappings[column])

#Eliminar columnas
columnas_a_eliminar = ['Grade']
df = df.drop(columns=columnas_a_eliminar)

print(df.head())

# Exportar
nombre_csv = 'model/milk.csv'

df.to_csv(nombre_csv, index=False)