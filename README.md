# FingerMusic

A Next.js web app that maps custom body poses detected by MoveNet to solfège notes for playful music training.

## Local development

```bash
npm install
npm run dev
```

Then open <http://localhost:3000> and allow camera access.

### Capturing reliable poses

1. Select a solfège note in **Training mode** and click **Start Timed Capture**.
2. A five-second countdown appears so you can step back and ensure your entire body fits in the frame.
3. After the countdown finishes, the app records five pose samples over a few seconds.
4. Review the thumbnails, pick the snapshot that looks clearest, and save it for the selected note.

You can retake the sequence at any time if the stored pose no longer matches how you perform it.

## Automated Vercel deployments

This repository contains a GitHub Actions workflow that can deploy the site to [Vercel](https://vercel.com/). Follow these steps to connect your Vercel project:

1. **Create a Vercel project**
   - Import your GitHub repository into Vercel, or create a project manually with the Vercel CLI (`vercel link`).
   - Copy the `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, and a [Vercel token](https://vercel.com/docs/concepts/deployments/api#authentication) from the project settings.

2. **Add GitHub secrets**
   - In your GitHub repository, navigate to **Settings → Secrets and variables → Actions**.
   - Create repository secrets named `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, and `VERCEL_TOKEN` with the values from step 1.

3. **Trigger a deployment**
   - Push to the `main` branch or click **Run workflow** on the `Deploy to Vercel` workflow tab. The workflow will:
     1. Install dependencies and build the Next.js app.
     2. Pull the environment configuration from Vercel.
     3. Produce a production build and deploy it using the Vercel CLI.

Once the workflow finishes, Vercel provides a live URL and retains previews for future commits.

### Preview deployments

If you want preview deployments for pull requests, duplicate the workflow and adjust the trigger to `pull_request`, or create a dedicated preview workflow that uses `vercel deploy --prebuilt --token=$VERCEL_TOKEN` without the `--prod` flag.

## Manual deployment

You can also deploy from your local machine:

```bash
npm install --global vercel
vercel login
vercel link
vercel --prod
```

The commands above will build the project, upload it to Vercel, and return the production URL.
