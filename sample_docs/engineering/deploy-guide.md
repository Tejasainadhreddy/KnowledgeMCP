# Deployment Guide

## Release schedule

Production deploys happen Monday to Thursday between 9:00 and 15:00 Pacific time. There are no production deploys on Fridays, weekends or company holidays unless it is a hotfix approved by the engineering manager on duty.

## Pipeline

Every merge to the main branch builds a container image, runs unit and integration tests, and deploys to the staging environment automatically. Promoting to production needs one approval from a second engineer in the deploy dashboard.

## Canary releases

Production deploys start as a canary that receives 5 percent of traffic for 20 minutes. If the error rate of the canary is more than 1 percent higher than the baseline, the deploy rolls back automatically.

## Rollback

To roll back manually, run `deployctl rollback <service> --to <previous-version>`. Rollbacks do not need approval. Announce every rollback in the #deploys channel.

## Feature flags

New user-facing features must ship behind a feature flag. Flags older than 90 days are reviewed and removed by the owning team.
