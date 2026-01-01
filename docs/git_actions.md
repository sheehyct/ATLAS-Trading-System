# GitHub UI actions for pull request diffs

The GitHub dropdown shown in the screenshot offers three shortcuts when you view a PR or diff:

1. **Create draft PR** – opens a pull request in *draft* state. Draft PRs share code early without requesting review or running required checks, and they must be marked “Ready for review” before merge.
2. **Copy git apply** – copies a shell command that fetches the patch and applies it locally with `git apply`, letting you test or stack the change without creating a commit yet.
3. **Copy patch** – copies the raw unified diff text to your clipboard so you can save it as a `.patch` file or apply it manually (e.g., with `git apply` or `patch`).

These options are useful when you want to experiment with a change locally, share work-in-progress, or apply someone else’s diff without manually downloading files.

## Create draft PR vs. Create PR (local main branch context)
- **Create draft PR** opens the pull request in *draft* state. It is meant for sharing code early; reviewers are not formally requested and required checks may not run until you click “Ready for review.” No changes are pushed to your local `main`—it only creates a remote PR placeholder.
- **Create PR** opens a standard pull request, immediately requesting review and triggering required checks. Like draft PRs, it does **not** modify your local `main`; it just prepares the PR on the remote so you can merge after approval and CI.

## Fetch a PR locally without touching `main`
To review a remote PR locally while keeping your `main` branch untouched, fetch the PR into a scratch branch and check it out:

```bash
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>-review
git checkout pr-<PR_NUMBER>-review
```

This downloads the PR’s commits without merging them. When you finish reviewing, you can delete the scratch branch with `git branch -D pr-<PR_NUMBER>-review` and your local `main` remains unchanged.
