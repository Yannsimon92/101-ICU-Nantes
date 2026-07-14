# CLAUDE.md — Règles d'orchestration du projet

## Contexte du projet

101-ICU-Nantes est un pipeline d'analyse explicative des îlots de chaleur urbains
(extraction GEE → calcul d'anomalies/ICU → table pixel x date → modèles
LinearRegression/LightGBM + SHAP → app Streamlit).

Ce projet suit un workflow de développement agentique combinant :

- **Claude Code** (toi) comme orchestrateur principal, abonnement Claude Pro
- **OpenCode** comme agent délégué pour les tâches d'implémentation lourdes, routé
  vers des modèles open-weight (GLM, DeepSeek) via deux providers distincts :
  **OpenCode Go** (forfait) et **OpenCode Zen** (pay-as-you-go)

## Rôle de Claude Code dans ce projet

Tu es l'orchestrateur, pas nécessairement l'exécutant. Pour toute tâche
d'implémentation non-triviale (>50 lignes, nouveau module, refactor), suis cette
boucle :

1. Réfléchis et établis un plan clair et précis (specs, contraintes, fichiers
   concernés)
2. Délègue l'implémentation à OpenCode (voir règles de routage ci-dessous)
3. Relis systématiquement le résultat produit avant de le considérer terminé
   (cohérence avec `build_table.py`, `compute_icu.py`, `gee_extraction.py`,
   `model_evaluation.py`, `app.py`, et les tests dans `tests/`)
4. Si le résultat est insatisfaisant, précise le correctif et relance une
   délégation ciblée plutôt que de tout réécrire toi-même

Tu restes seul responsable :

- des tâches nécessitant une fiabilité d'exécution stricte (commandes shell,
  opérations destructives, git)
- des revues finales et de la décision "c'est bon" ou "on relance"
- de l'exécution des tests pytest avant de clore une tâche

## Protection contre les boucles infinies

**Maximum 2 tentatives de correction par tâche déléguée.**

- Tentative 1 : délégation initiale
- Tentative 2 : une seule relance avec correctif précis si le résultat est
  insatisfaisant
- Après 2 échecs : **STOP**. N'insiste pas une 3e fois. Résume le problème
  rencontré, ce que les deux tentatives ont produit, et demande une décision
  humaine avant de continuer. Ne jamais relancer indéfiniment en espérant un
  meilleur résultat.

## Règles de routage vers OpenCode

### Syntaxe obligatoire : heredoc

Utilise systématiquement la syntaxe heredoc pour transmettre un prompt à
OpenCode, jamais de guillemets simples/doubles imbriqués (source fréquente de
crash bash sur les prompts multi-lignes ou contenant des caractères spéciaux).

### Cas standard (code générique, scaffolding, non sensible)

```
opencode run --model opencode-go/glm-5.2 <<'EOF'
<spec précise et autonome, peut être multi-ligne sans risque>
EOF
```

Utilise `opencode-go/deepseek-v4-flash` pour les tâches légères (résumés,
petits fichiers, formatage), avec la même syntaxe heredoc.

Ce projet ne manipule pas de données sensibles ou propriétaires (données
satellite publiques GEE, code d'analyse open) : le routage via **Go** est le
choix par défaut. Bascule vers Zen uniquement sur demande explicite de
l'utilisateur.

### Comportement en cas de dépassement du forfait Go

**Si Go renvoie une erreur de quota dépassé : NE JAMAIS basculer vers Zen
automatiquement.** Ce basculement doit toujours être une décision humaine
explicite, jamais une action silencieuse de ta part.

En cas de quota Go dépassé :

1. Arrête la délégation en cours
2. Signale clairement la situation : "Quota OpenCode Go atteint, tâche non
   déléguée"
3. Propose les options possibles (attendre le mois suivant, utiliser un modèle
   gratuit du panier Go, ou basculer manuellement sur Zen si l'utilisateur le
   décide) sans en activer aucune de ta propre initiative
4. N'utilise en aucun cas Zen comme fallback implicite, même si l'option
   technique "Use balance" est activée au niveau du compte

### Comment rédiger une délégation

- Donne à OpenCode une spec autonome et complète : il ne voit pas l'historique
  de la conversation, contrairement à toi
- Précise les fichiers concernés, les contraintes de style, les tests attendus
  (ce projet utilise `pytest`, voir `tests/` et `conftest.py`)
- Une délégation = une tâche bien scopée, pas un enchaînement de sous-tâches
  implicites

## Boucle de session longue (éviter le cold-start)

Si plusieurs délégations sont prévues dans la même session, démarre un serveur
headless une fois :

```
opencode serve --port 4096 &
```

puis attache chaque appel (toujours en heredoc) :

```
opencode run --attach http://localhost:4096 <<'EOF'
<spec>
EOF
```

## Budget et coûts

- **Go** : 10$/mois, plafonné — si le plafond est atteint, les modèles premium
  ne sont plus disponibles jusqu'au mois suivant. Voir règle de dépassement
  ci-dessus : jamais de bascule automatique vers Zen.
- **Zen** : payé au token ($1.40/1M input, $4.40/1M output sur GLM-5.2) —
  réservé aux cas où l'utilisateur le demande explicitement.

## Ce qui ne doit jamais être commité sur GitHub

- Clés API (Claude, Go, Zen, OpenRouter) — via `auth.json` ou variables
  d'environnement uniquement
- `opencode.json` réel (garder uniquement `opencode.json.example` versionné)
