# house-of-knowledge

RAG app over Obsidian D&D session notes. See `docs/superpowers/specs/2026-05-29-dnd-rag-design.md` for design.

## AWS setup (Bedrock)

The app calls **one API**: `bedrock:InvokeModel` on the Bedrock Runtime endpoint in `eu-west-2`. Embeddings run locally on the NUC (BGE-M3) — AWS is only used for text generation.

### Permissions required

| Permission | Required? | Purpose |
|------------|-----------|---------|
| `bedrock:InvokeModel` | Yes | Generate chat responses |
| `bedrock:InvokeModelWithResponseStream` | No | Not used |
| S3, DynamoDB, IAM, etc. | No | Not used |

Scope the IAM policy to **only the model(s) you plan to use**. Default is Amazon Nova Lite; you can swap to Claude Haiku 4.5 via `BEDROCK_MODEL_ID` in `.env`.

### Step 1: Submit Anthropic use case form (Haiku only)

If you plan to use Claude Haiku 4.5, complete the one-time Anthropic first-use form in the AWS console the first time you invoke an Anthropic model. Nova Lite does not require this.

1. Open [Amazon Bedrock console](https://eu-west-2.console.aws.amazon.com/bedrock/) (region: **Europe London / eu-west-2**)
2. Try a test invocation in **Chat / Text playground**, or invoke Haiku once from the CLI
3. If prompted, submit the Anthropic use case form and wait for approval

### Step 2: Create IAM user

1. Open [IAM → Users → Create user](https://console.aws.amazon.com/iam/home#/users)
2. User name: `house-of-knowledge-bedrock`
3. **Do not** attach any AWS managed policies (no `AmazonBedrockFullAccess`)
4. Create the user

### Step 3: Attach inline policy

IAM → Users → `house-of-knowledge-bedrock` → **Add permissions** → **Create inline policy** → JSON:

**Nova Lite only (default, cheapest):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeNovaLite",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:eu-west-2::foundation-model/amazon.nova-lite-v1:0"
    }
  ]
}
```

**Nova Lite + Haiku 4.5 (if you want to switch models via env var):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeBedrockModels",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:eu-west-2::foundation-model/amazon.nova-lite-v1:0",
        "arn:aws:bedrock:eu-west-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"
      ]
    }
  ]
}
```

Name the policy `house-of-knowledge-bedrock-invoke` and save.

To verify the exact model ID available in your account, run:

```bash
aws bedrock list-foundation-models --region eu-west-2 \
  --query "modelSummaries[?contains(modelId, 'nova-lite') || contains(modelId, 'haiku-4-5')].modelId"
```

Use the returned ID as `BEDROCK_MODEL_ID` in `.env`.

### Step 4: Create access key

1. IAM → Users → `house-of-knowledge-bedrock` → **Security credentials**
2. **Create access key** → use case: **Application running outside AWS**
3. Copy the Access key ID and Secret access key (shown once)

### Step 5: Configure `.env` on the NUC

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-2
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

To use Haiku instead:

```bash
BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
```

Never commit `.env` — it is gitignored.

### Step 6: Verify Bedrock access

From a machine with the credentials configured:

```bash
aws bedrock-runtime invoke-model \
  --region eu-west-2 \
  --model-id amazon.nova-lite-v1:0 \
  --content-type application/json \
  --accept application/json \
  --body '{"messages":[{"role":"user","content":[{"text":"Say hello in one word."}]}],"inferenceConfig":{"maxTokens":32}}' \
  /tmp/bedrock-out.json && cat /tmp/bedrock-out.json
```

A JSON response with generated text means the key and policy are correct.

### Step 7: Set a budget alert (recommended)

This uses the AWS console — the app's IAM user does **not** need billing permissions.

1. Open [AWS Budgets](https://console.aws.amazon.com/billing/home#/budgets)
2. **Create budget** → **Cost budget** → **Monthly**
3. Budget amount: **$2**
4. Alert threshold: **80%** and **100%** → email notification
5. Filter (optional): Service = **Amazon Bedrock**

At expected usage (~5 users, weekly sessions), Nova Lite costs well under $1/month.

### Security notes

- Use a dedicated IAM user — not your root account or admin user
- Rotate access keys periodically (IAM → Security credentials → Create new key → disable old)
- If a key is exposed, delete it immediately and create a replacement
- The NUC only needs outbound HTTPS to `bedrock-runtime.eu-west-2.amazonaws.com`

## Quick start (local dev)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in AWS credentials
pytest -v
python app/main.py
```

## Deploy (NUC)

```bash
cp .env.example .env  # fill in AWS credentials
docker compose up -d --build
```

## Index notes

From Mac after a session:

```bash
VAULT_PATH=~/path/to/vault NUC_HOST=nuc.local ./scripts/sync-and-index.sh
```
