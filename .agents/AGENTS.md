# Project-Scoped Rules

## Branching & Release Strategy
- **dev Branch**: All daily changes, bug fixes, features, and refactoring modifications MUST be committed and pushed immediately to the `dev` branch.
- **main Branch**: Do NOT push daily development changes directly to the `main` branch. Merging to the `main` branch is reserved only for triggering production builds when explicitly requested by the user.
- **Current Active Branch**: Keep the local repository checkout on the `dev` branch for all standard workspace modifications.
