"""Orchestration du benchmark : génération, évaluation, contrôle de chaîne.

Trois entrées, dans cet ordre :

1. ``python -m extraction.ingest``          — une fois, avant tout
2. ``python -m orchestration.smoke_test``   — valide la chaîne en quelques minutes
3. ``python -m orchestration.run_agents``   — génération (longue, reprenable)
4. ``python -m orchestration.run_eval``     — scorers et juges sur les sorties
"""
