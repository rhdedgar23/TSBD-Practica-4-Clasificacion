import pandas as pd
import numpy as np
from PIL.TiffImagePlugin import X_RESOLUTION
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

df_train= pd.read_csv("train.csv", low_memory=False)
df_test= pd.read_csv("test.csv", low_memory=False)

print("Dataset Cat in the Dat -- Train\n", df_train)
print("Dataset Cat in the Dat -- Test\n", df_test)
df_train_bin= df_train.iloc[:, 1:6]
print("\nTrain --> valores Binarios:\n", df_train_bin)
df_train_nom= df_train.iloc[:, 6:16]
print("\nTrain --> valores Nominales:\n", df_train_nom)
df_train_ord= df_train.iloc[:, 16:22]
print("\nTrain --> valores Ordinales:\n", df_train_ord)
df_train_cic= df_train.iloc[:, 22:24]
print("\nTrain --> valores Ciclicos:\n", df_train_cic)

""" 
Contamos los datos faltantes NaN en Train y Test
"""
#isnull() regresa un boolean si una celda es NaN
#sum() regresa suma total despues de contar cada booleano y regresa en tabla de dato: num de datos faltantes
datos_faltantes_train= df_train.isnull().sum()
datos_faltantes_test= df_test.isnull().sum()
print("Datos faltantes Train:\n", datos_faltantes_train)
print("Datos faltantes Test:\n", datos_faltantes_test)



"""
Calculamos el porcentaje de datos faltantes para Train y Test
"""
#shape regresa una tupla indicando el numero de filas y columnas en el Dataframe
train_size= df_train.shape
print("\nTrain\nFilas: ", train_size[0], "\nColumnas: ", train_size[1])
num_celdas_train= np.prod(df_train.shape)
print("\nNumero de celdas Train: ", num_celdas_train)
num_celdas_faltantes_train= datos_faltantes_train.sum() #regresa la suma total de los datos faltantes de cada dato
porcentaje_datos_faltantes_train= (num_celdas_faltantes_train/num_celdas_train)*100
print("\nPorcentaje de datos faltantes en Train: ", porcentaje_datos_faltantes_train)

test_size= df_test.shape
print("\nTest\nFilas: ", test_size[0], "\nColumnas: ", test_size[1])
num_celdas_test= np.prod(df_test.shape)
print("\nNumero de celdas Test: ", num_celdas_test)
num_celdas_faltantes_test= datos_faltantes_test.sum()
porcentaje_datos_faltantes_test= (num_celdas_faltantes_test/num_celdas_test)*100
print("\nPorcentaje de datos faltantes en Test: ", porcentaje_datos_faltantes_test)



"""
Rellenamos las celdas faltantes para Train y Test
"""
df_train_sub= df_train.bfill(axis=0).fillna(0)
#fillna(0) al final se usa para aquellos ultimos valores que no tienen valor siguiente
print("\nDataset original Train:", df_train)
print("\nTrain Dataset despues de rellenar valores nulos:\n", df_train_sub)

df_test_sub= df_test.bfill(axis=0).fillna(0)
print("\nDataset original Test:", df_test)
print("\nTest Dataset despues de rellenar valores nulos:\n", df_test_sub)



"""
Codificamos con Codificacion Ordinal
"""
# Para evitar redundancia al codificar las columnas de valores binarios,
# optamos por convertir los valores de las columnas bin_3 y bin_4 a 1's y 0's
# Esto es para evitar que crezca innecesariamente la dimension del dataset despues de la codificacion

"""
        Train
"""
# Las columnas con valores de tipo str son:
# bin_3, bin_4,
# nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9
# ord_1, ord_2, ord_3, ord_4, ord_5
X_train = df_train_sub.iloc[:, 1:-1]

X_train['bin_3'] = X_train['bin_3'].map({'T': 1, 'F': 0})
X_train['bin_4'] = X_train['bin_4'].map({'Y': 1, 'N': 0})

X_train_str = df_train_sub.loc[:, ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9",
                                   "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]]
#X_train --> Todas las columnas del dataset Train
print("\nTrain Data:\n", X_train)
#X_train_str --> Columnas que contiene unicamente strings
print("\nTrain Data Str:\n", X_train_str)
print("\nTrain Data Str Types:\n", X_train_str.dtypes)

#Checamos si hay algun elemento diferente de string
for col in X_train_str.columns:
    types = X_train_str[col].apply(type).unique()
    print(f"{col}: {types}")
    #y lo corregimos en caso de que si exista
    if len(types) > 1:
        print(f"Fixing mixed types in {col}: {types}")
        X_train_str[col] = X_train_str[col].astype(str)

"""
        Test
"""
# Las columnas con valores de tipo str son:
# bin_3, bin_4,
# nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9
# ord_1, ord_2, ord_3, ord_4, ord_5
X_test = df_test_sub.iloc[:, 1:]

X_test['bin_3'] = X_test['bin_3'].map({'T': 1, 'F': 0})
X_test['bin_4'] = X_test['bin_4'].map({'Y': 1, 'N': 0})

X_test_str = df_test_sub.loc[:, ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9",
                                 "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]]
#X_test --> Todas las columnas del dataset Test
print("\nTest Data:\n", X_test)
#X_test_str --> Columnas que contiene unicamente strings
print("\nTest Data Str:\n", X_test_str)
print("\nTest Data Str Types:\n", X_test_str.dtypes)

#Checamos si hay algun elemento diferente de string
for col in X_test_str.columns:
    types = X_test_str[col].apply(type).unique()
    print(f"{col}: {types}")
    #y lo corregimos en caso de que si exista
    if len(types) > 1:
        print(f"Fixing mixed types in {col}: {types}")
        X_test_str[col] = X_test_str[col].astype(str)



"""
    Codificacion One Hot
"""
"""
        Train
"""
encoder = OneHotEncoder(sparse_output=False)
X_train_str_encoded = encoder.fit_transform(X_train_str)

# Intentamos obtener nombres de columnas de forma compatible
try:
    feature_names = encoder.get_feature_names_out(X_train_str.columns)
except AttributeError:
    feature_names = encoder.get_feature_names(X_train_str.columns)

X_train_str_encoded_df = pd.DataFrame(X_train_str_encoded, columns=feature_names)

print("\n\n\n\nCodificación One Hot --> Train (columnas str) ", X_train_str_encoded_df)

"""
        Test
"""
#encoder = OneHotEncoder(sparse_output=False)
X_test_str_encoded = encoder.fit_transform(X_test_str)

# Intentamos obtener nombres de columnas de forma compatible
try:
    feature_names = encoder.get_feature_names_out(X_test_str.columns)
except AttributeError:
    feature_names = encoder.get_feature_names(X_test_str.columns)

X_test_str_encoded_df = pd.DataFrame(X_test_str_encoded, columns=feature_names)

print("\n\n\n\nCodificación One Hot --> Test (columnas str) ", X_test_str_encoded_df)



"""
    Codificacion Ordinal
"""
"""
        Train
"""
# Crear codificador OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Ajustar y transformar
X_train_str_ordinal = ordinal_encoder.fit_transform(X_train_str)

# Convertir a DataFrame para verlo mejor
X_train_str_ordinal_df = pd.DataFrame(X_train_str_ordinal, columns=X_train_str.columns)
print("\n\n\n\nCodificación Ordinal --> Train (columnas str)\n", X_train_str_ordinal_df)

"""
        Test
"""
# Crear codificador OrdinalEncoder
#ordinal_encoder = OrdinalEncoder()

# Ajustar y transformar
X_test_str_ordinal = ordinal_encoder.fit_transform(X_test_str)

# Convertir a DataFrame para verlo mejor
X_test_str_ordinal_df = pd.DataFrame(X_test_str_ordinal, columns=X_test_str.columns)
print("\n\n\n\nCodificación Ordinal --> Test (columnas str)\n", X_test_str_ordinal_df)




"""
Juntamos los datos codificados con los no codificados
"""
# Los datos que debemos concatenar son:
# Los datos ya codificados (que son los datos str) + datos que no se codificaron

# Los datos no codificados para Train y Test son:
# bin_0, bin_1, bin_2, bin_3, bin_4
# ord_0
# day, month
"""
    Train
"""
X_train_not_enc = X_train.loc[:, ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                                  'ord_0',
                                  'day', 'month']]
X_train_ordinal = X_train_str_ordinal_df.iloc[:, :]
print("\n\nDatos Train Codificados (Ordinal): \n\n", X_train_ordinal)
print("\n\nDatos Train Int: \n\n", X_train_not_enc)
for col in X_train_not_enc.columns:
    X_train_ordinal[col] = X_train_not_enc[col].values
Y_train_ordinal = df_train_sub.loc[:, ["target"]]
print("\n\nX Train (Ordinal)\n\n", X_train_ordinal)
print("\n\nY Train (Ordinal)\n\n", Y_train_ordinal)


"""
    Test
"""
X_test_not_enc = X_test.loc[:, ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                                  'ord_0',
                                  'day', 'month']]
X_test_ordinal = X_test_str_ordinal_df.iloc[:, :]
print("\n\nDatos Test Codificados (Ordinal): \n\n", X_test_ordinal)
print("\n\nDatos Test Int: \n\n", X_test_not_enc)
for col in X_test_not_enc.columns:
    X_test_ordinal[col] = X_test_not_enc[col].values
print("\n\nX Test (Ordinal)\n\n", X_test_ordinal)



"""
Aplicamos los dataframes Train y Test ya codificados a los algoritmos Arbol de Busqueda y KNN
"""
"""
    Train
"""
# Dividir en entrenamiento y prueba
X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(X_train_ordinal, Y_train_ordinal, test_size=0.2, random_state=42)

# ===== Modelo: Árbol de decisión =====
arbol = DecisionTreeClassifier(max_depth=8, random_state=42, criterion='entropy', class_weight='balanced', min_samples_split=5, min_samples_leaf=2)
arbol.fit(X_train_ord, y_train_ord)
y_pred_arbol = arbol.predict(X_test_ord)
# Evaluation metrics
acc_arbol = accuracy_score(y_test_ord, y_pred_arbol)
prec_arbol = precision_score(y_test_ord, y_pred_arbol)
rec_arbol = recall_score(y_test_ord, y_pred_arbol)
f1_arbol = f1_score(y_test_ord, y_pred_arbol)
# Print results
print("=== Decision Tree Metrics ===")
print(f"Accuracy:  {acc_arbol:.4f}")
print(f"Precision: {prec_arbol:.4f}")
print(f"Recall:    {rec_arbol:.4f}")
print(f"F1 Score:  {f1_arbol:.4f}")


# ===== Modelo: K-Nearest Neighbors =====
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_ord, y_train_ord.values.ravel())
y_pred_knn = knn.predict(X_test_ord)
# Evaluation metrics
acc_knn = accuracy_score(y_test_ord, y_pred_knn)
prec_knn = precision_score(y_test_ord, y_pred_knn)
rec_knn = recall_score(y_test_ord, y_pred_knn)
f1_knn = f1_score(y_test_ord, y_pred_knn)
# Print results
print("=== KNN Metrics ===")
print(f"Accuracy:  {acc_knn:.4f}")
print(f"Precision: {prec_knn:.4f}")
print(f"Recall:    {rec_knn:.4f}")
print(f"F1 Score:  {f1_knn:.4f}")

