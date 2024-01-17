import pandas as pd
import numpy as np

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
    df_final = extraire_donnees_pour_analyse()
    print(df_final.head())
