# NEXUS-BTC

GMOコイン BTC自動売買Bot（GitHub Actions + GitHub Pages）

## セットアップ

### 1. リポジトリをPrivateで作成してGitHubにpush

### 2. GitHub Secrets に以下を登録
`Settings → Secrets and variables → Actions → New repository secret`

| キー名 | 値 |
|---|---|
| `GMO_API_KEY` | GMOコインのAPIキー |
| `GMO_API_SECRET` | GMOコインのAPIシークレット |

### 3. GitHub Pages を有効化
`Settings → Pages → Source: gh-pages branch`

### 4. Actions を有効化
`Actions タブ → Enable workflows`

これだけで毎時自動実行されます。

## ダッシュボード
`https://<your-username>.github.io/<repo-name>/`

## 設定変更
`config.yml` で取引金額・損切り率・利確率などを調整できます。
