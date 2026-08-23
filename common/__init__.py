"""Utilitaires partagés entre l'ingestion, les agents et l'évaluation.

Ce paquet existe parce que la détection de langue et le comptage de tokens sont
nécessaires des deux côtés de la chaîne : à l'ingestion (pour avertir d'un
dépassement de contexte) et à l'évaluation (scorer de conformité linguistique).
Les y mutualiser évite une dépendance croisée entre `extraction` et `evaluation`.
"""
