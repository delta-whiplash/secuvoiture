from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def charger_donnees():
    chemin_caracteristiques = 'C:\\Users\\delta\\Desktop\\secuvoiture\\carcteristiques-2022.csv'
    chemin_lieux = 'C:\\Users\\delta\\Desktop\\secuvoiture\\lieux-2022.csv'
    chemin_usagers = 'C:\\Users\\delta\\Desktop\\secuvoiture\\usagers-2022.csv'
    chemin_vehicules = 'C:\\Users\\delta\\Desktop\\secuvoiture\\vehicules-2022.csv'

    df_caracteristiques = pd.read_csv(chemin_caracteristiques, sep=';')
    df_lieux = pd.read_csv(chemin_lieux, sep=';', low_memory=False)
    df_usagers = pd.read_csv(chemin_usagers, sep=';')
    df_vehicules = pd.read_csv(chemin_vehicules, sep=';')
    logging.info("Chargement des données terminé.")

    return df_caracteristiques, df_lieux, df_usagers, df_vehicules
def nettoyer_donnees(df, colonnes_importantes, seuil_nan=0.5):
    # Supprimer les doublons
    df = df.drop_duplicates()

    # Supprimer les lignes avec un trop grand nombre de valeurs manquantes
    df = df.dropna(thresh=int(seuil_nan * len(df.columns)))

    # Supprimer les lignes où les colonnes importantes ont des valeurs manquantes
    df = df.dropna(subset=colonnes_importantes)

    return df

def limiter_donnees(df, nb_lignes=1500):
    return df.head(nb_lignes)

def enregistrer_donnees(df, nom_fichier, chemin_base='C:\\Users\\delta\\Desktop\\secuvoiture\\'):
    chemin = f'{chemin_base}{nom_fichier}'
    df.to_csv(chemin, index=False, sep=';')

def transformer_et_normaliser(df, colonnes_numeriques, colonnes_categorielles):
    # Assurez-vous que les colonnes numériques sont réellement numériques
    colonnes_numeriques_effectives = df.select_dtypes(include=[np.number]).columns.intersection(colonnes_numeriques)
    
    # Normalisation des colonnes numériques
    scaler = MinMaxScaler()
    df[colonnes_numeriques_effectives] = scaler.fit_transform(df[colonnes_numeriques_effectives])

    # Codage des variables catégorielles
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    df_categorielles = pd.DataFrame(encoder.fit_transform(df[colonnes_categorielles]))
    df_categorielles.columns = encoder.get_feature_names_out(colonnes_categorielles)

    # Concaténation des colonnes numériques et catégorielles transformées
    df = pd.concat([df.drop(colonnes_categorielles, axis=1), df_categorielles], axis=1)

    return df

def transcoder_sexe(df):
    df['sexe'] = df['sexe'].replace({1: 'H', 2: 'F'}) 
    return df

def extraire_donnees_pour_analyse():
    # Chargement des données
    df_caracteristiques, df_lieux, df_usagers, df_vehicules = charger_donnees()

    # Colonnes importantes pour chaque DataFrame
    colonnes_importantes_caracteristiques = ['Accident_Id', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com']
    colonnes_importantes_lieux = ['Num_Acc', 'catr', 'voie', 'circ', 'nbv', 'vosp', 'prof', 'pr', 'pr1']
    colonnes_importantes_usagers = ['Num_Acc', 'id_usager', 'id_vehicule', 'num_veh', 'place', 'catu', 'grav', 'sexe', 'an_nais']
    colonnes_importantes_vehicules = ['Num_Acc', 'id_vehicule', 'catv']

    # Nettoyage des données avec un filtrage moins agressif
    df_caracteristiques = nettoyer_donnees(df_caracteristiques, colonnes_importantes_caracteristiques)
    df_lieux = nettoyer_donnees(df_lieux, colonnes_importantes_lieux)
    df_usagers = nettoyer_donnees(df_usagers, colonnes_importantes_usagers)
    df_vehicules = nettoyer_donnees(df_vehicules, colonnes_importantes_vehicules)

    # Limitation des données à 1500 lignes
    df_caracteristiques = limiter_donnees(df_caracteristiques)
    df_lieux = limiter_donnees(df_lieux)
    df_usagers = limiter_donnees(df_usagers)
    df_vehicules = limiter_donnees(df_vehicules)

    # Filtrage des colonnes numériques
    colonnes_numeriques_usagers = df_usagers.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_numeriques_vehicules = df_vehicules.select_dtypes(include=[np.number]).columns.tolist()

    # Colonnes pour la transformation et normalisation
    colonnes_numeriques_caracteristiques = ['jour', 'mois', 'an', 'lum']
    colonnes_categorielles_caracteristiques = ['dep', 'com', 'agg', 'int', 'atm', 'col']

    colonnes_numeriques_lieux = ['catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'vma']
    colonnes_categorielles_lieux = []  # À ajuster si nécessaire

    colonnes_numeriques_usagers = ['place', 'catu', 'grav', 'sexe', 'an_nais', 'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'actp', 'etatp']
    colonnes_categorielles_usagers = []  # À ajuster si nécessaire

    colonnes_numeriques_vehicules = ['senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor']
    colonnes_categorielles_vehicules = []  # À ajuster si nécessaire

    # Transcodage
    df_usagers = transcoder_sexe(df_usagers)

    # Transformation et normalisation
    df_caracteristiques = transformer_et_normaliser(df_caracteristiques, colonnes_numeriques_caracteristiques, colonnes_categorielles_caracteristiques)
    df_lieux = transformer_et_normaliser(df_lieux, colonnes_numeriques_lieux, colonnes_categorielles_lieux)
    df_usagers = transformer_et_normaliser(df_usagers, colonnes_numeriques_usagers, colonnes_categorielles_usagers)
    df_vehicules = transformer_et_normaliser(df_vehicules, colonnes_numeriques_vehicules, colonnes_categorielles_vehicules)

    # Transcodage
    df_usagers = transcoder_sexe(df_usagers)

    # Fusion des tables
    df_fusionnee = df_caracteristiques.merge(df_lieux, on='Num_Acc')
    df_fusionnee = df_fusionnee.merge(df_usagers, on='Num_Acc')
    df_fusionnee = df_fusionnee.merge(df_vehicules, on='Num_Acc')

    return df_fusionnee

if __name__ == '__main__':
    logging.info("Début du script.")
    # Chargement des données
    df_caracteristiques, df_lieux, df_usagers, df_vehicules = charger_donnees()
    print(df_usagers.head())

    # Colonnes importantes pour chaque DataFrame
    colonnes_importantes_caracteristiques = ['Accident_Id', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com']
    colonnes_importantes_lieux = ['Num_Acc', 'catr', 'voie', 'circ', 'nbv', 'vosp', 'prof', 'pr', 'pr1']
    colonnes_importantes_usagers = ['Num_Acc', 'id_usager', 'id_vehicule', 'num_veh', 'place', 'catu', 'grav', 'sexe', 'an_nais']
    colonnes_importantes_vehicules = ['Num_Acc', 'id_vehicule', 'catv']

    # Nettoyage des données avec un filtrage moins agressif
    logging.info("Nettoyage des données en cours.")
    df_caracteristiques = nettoyer_donnees(df_caracteristiques, colonnes_importantes_caracteristiques)
    df_lieux = nettoyer_donnees(df_lieux, colonnes_importantes_lieux)
    df_usagers = nettoyer_donnees(df_usagers, colonnes_importantes_usagers)
    df_vehicules = nettoyer_donnees(df_vehicules, colonnes_importantes_vehicules)

    # Limitation des données à 1500 lignes
    logging.info("Limitation des données à 1500 lignes.")
    df_caracteristiques = limiter_donnees(df_caracteristiques)
    df_lieux = limiter_donnees(df_lieux)
    df_usagers = limiter_donnees(df_usagers)
    df_vehicules = limiter_donnees(df_vehicules)

    logging.info("Filtrage des colonnes numériques.")
    colonnes_numeriques_usagers = df_usagers.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_numeriques_vehicules = df_vehicules.select_dtypes(include=[np.number]).columns.tolist()

    # Colonnes pour la transformation et normalisation
    logging.info("Transformation et normalisation des données.")
    colonnes_numeriques_caracteristiques = ['jour', 'mois', 'an', 'lum']
    colonnes_categorielles_caracteristiques = ['dep', 'com', 'agg', 'int', 'atm', 'col']

    colonnes_numeriques_lieux = ['catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'vma']
    colonnes_categorielles_lieux = []  # à Ajuster si nécessaire

    colonnes_numeriques_usagers = ['place', 'catu', 'grav', 'sexe', 'an_nais', 'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'actp', 'etatp']
    colonnes_categorielles_usagers = []  # à Ajuster si nécessaire

    colonnes_numeriques_vehicules = ['senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor']
    colonnes_categorielles_vehicules = []  # à Ajuster si nécessaire

    # Transcodage
    df_usagers = transcoder_sexe(df_usagers)

    # Transformation et normalisation
    df_caracteristiques = transformer_et_normaliser(df_caracteristiques, colonnes_numeriques_caracteristiques, colonnes_categorielles_caracteristiques)
    df_lieux = transformer_et_normaliser(df_lieux, colonnes_numeriques_lieux, colonnes_categorielles_lieux)
    df_usagers = transformer_et_normaliser(df_usagers, colonnes_numeriques_usagers, colonnes_categorielles_usagers)
    df_vehicules = transformer_et_normaliser(df_vehicules, colonnes_numeriques_vehicules, colonnes_categorielles_vehicules)
    
    # Transcodage
    df_usagers = transcoder_sexe(df_usagers)

    # Enregistrement des données limitées dans de nouveaux fichiers CSV
    logging.info("Enregistrement des données transformées.")
    enregistrer_donnees(df_caracteristiques, 'caracteristiques_limitees.csv')
    enregistrer_donnees(df_lieux, 'lieux_limites.csv')
    enregistrer_donnees(df_usagers, 'usagers_limites.csv')
    enregistrer_donnees(df_vehicules, 'vehicules_limites.csv')

    df_caracteristiques.rename(columns={'Accident_Id': 'Num_Acc'}, inplace=True)

    df_fusionnee = df_caracteristiques.merge(df_lieux, on='Num_Acc')
    df_fusionnee = df_fusionnee.merge(df_usagers, on='Num_Acc')
    df_fusionnee = df_fusionnee.merge(df_vehicules, on='Num_Acc')

    # Affichage des premières lignes pour vérifier
    print(df_fusionnee.head())
    enregistrer_donnees(df_fusionnee, 'fusion.csv')

    df_final = extraire_donnees_pour_analyse()
    logging.info("Script terminé avec succès.")