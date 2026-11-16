# Collaboration

[Retour au README](../README.md)

Répartir les tâches et les notebooks avant modification. Depuis la branche principale à jour (adapter `main` au nom réel) :

```powershell
git switch main
git pull --ff-only
git switch -c documentation/installation
# Après modification :
git status
git diff
git add README.md
git commit -m "Documenter l'installation"
git push -u origin documentation/installation
```

Sur GitHub, ouvrir une **pull request** vers `main`, décrire les changements et vérifications, puis demander une relecture au collègue. Après fusion, récupérer les modifications avec `git switch main` puis `git pull --ff-only`. Enregistrer ou committer le travail local avant de changer de branche.

Éviter de modifier simultanément le même notebook : les conflits JSON sont difficiles à relire. Avant un commit courant, utiliser **Clear All Outputs** dans VS Code et enregistrer, pour limiter les différences de sorties et compteurs. Conserver séparément les résultats utiles dans `docs/`. Pour la remise, préparer une version exécutée et relue en coordonnant la conservation des sorties finales. Relire `git diff` et ajouter explicitement les fichiers souhaités. Ne pas versionner les données ni les CSV générés; ajouter les `.gitkeep` lors du premier commit.
