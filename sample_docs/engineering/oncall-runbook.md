# On-call Runbook

This runbook describes how Northwind Labs engineers handle production incidents.

## Severity levels

SEV1 means a full outage or data loss affecting customers. The on-call engineer must acknowledge a SEV1 page within 5 minutes and post an update in the #incidents channel every 30 minutes until it is resolved.

SEV2 means a major feature is degraded for many customers. Acknowledge within 15 minutes and post updates every hour.

SEV3 means a minor issue with a workaround. Acknowledge within one business day.

## Paging and escalation

Pages are sent through PagerDuty to the primary on-call engineer. If the primary does not acknowledge within 10 minutes, the page escalates automatically to the secondary on-call engineer. If neither responds within 20 minutes, it escalates to the engineering manager on duty.

The on-call rotation is one week long and hands over every Monday at 10:00 Pacific time.

## Database failover

If the primary database is unhealthy for more than 3 minutes, run the failover playbook `db-failover.sh` from the ops bastion host. Only engineers in the `db-admins` group can run it. After failover, confirm replication lag is below 5 seconds before closing the incident.

## After the incident

Every SEV1 and SEV2 incident needs a blameless postmortem. The incident commander writes it within 5 business days using the postmortem template, and it is reviewed in the Thursday reliability meeting.
