"""AWS infrastructure for KnowledgeMCP, designed to stay under ~$10/month at demo traffic.

Cost choices:
- Aurora Serverless v2 with min capacity 0: the database pauses when idle (no compute charge).
- Lambdas reach Aurora through the RDS Data API (HTTPS), so there is no NAT gateway or VPC endpoint.
- One container image, three Lambda handlers; HTTP API (cheaper than REST API) with throttling.
- Optional AWS Budgets email alert at $10/month.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, SecretValue, Stack, Tags, triggers
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "KnowledgeMCP"
SERVICE = "knowledgemcp"


class KnowledgeMcpStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, alert_email: str | None = None,
                 llm_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        Tags.of(self).add("Project", "KnowledgeMCP")

        # --- Database: Aurora PostgreSQL Serverless v2 + pgvector, scales to zero -------------------
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0, subnet_configuration=[
            ec2.SubnetConfiguration(name="db", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24)])
        cluster = rds.DatabaseCluster(
            self, "Db",
            engine=rds.DatabaseClusterEngine.aurora_postgres(version=rds.AuroraPostgresEngineVersion.VER_16_6),
            writer=rds.ClusterInstance.serverless_v2("writer"),
            serverless_v2_min_capacity=0,
            serverless_v2_max_capacity=2,
            serverless_v2_auto_pause_duration=Duration.minutes(10),
            enable_data_api=True,
            default_database_name="kmcp",
            credentials=rds.Credentials.from_generated_secret("kmcp_admin"),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            storage_encrypted=True,
            backup=rds.BackupProps(retention=Duration.days(1)),
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,  # demo project: `cdk destroy` removes everything
        )

        # --- Storage and secrets -----------------------------------------------------------------
        bucket = s3.Bucket(
            self, "Docs",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        # Hashed API keys (added later with scripts/create_api_key.py). CloudFormation only resets this
        # value if the template value changes, so keys you add survive redeploys.
        api_keys = secretsmanager.Secret(
            self, "ApiKeys", description="KnowledgeMCP API keys (SHA-256 hash -> user and teams)",
            secret_string_value=SecretValue.unsafe_plain_text("{}"),
        )

        # --- Compute: one image, three handlers ------------------------------------------------------
        def image(cmd: str) -> lambda_.DockerImageCode:
            return lambda_.DockerImageCode.from_image_asset(
                str(REPO_ROOT), cmd=[cmd], platform=ecr_assets.Platform.LINUX_AMD64,
                exclude=["infra", ".git", "evals/results", "**/__pycache__", ".venv", "docs"])

        env = {
            "DB_MODE": "dataapi",
            "DB_CLUSTER_ARN": cluster.cluster_arn,
            "DB_SECRET_ARN": cluster.secret.secret_arn,
            "DB_NAME": "kmcp",
            "EMBEDDER": "bedrock",
            "EMBED_DIM": "512",
            "LLM": "bedrock",
            "LLM_MODEL_ID": llm_model_id,
            "RERANKER": "bedrock",
            "POWERTOOLS_SERVICE_NAME": SERVICE,
            "POWERTOOLS_METRICS_NAMESPACE": NAMESPACE,
            "POWERTOOLS_LOG_LEVEL": "INFO",
        }

        def function(fid: str, cmd: str, timeout: Duration, extra_env: dict | None = None) -> lambda_.Function:
            log_group = logs.LogGroup(self, f"{fid}Logs", retention=logs.RetentionDays.TWO_WEEKS,
                                      removal_policy=RemovalPolicy.DESTROY)
            fn = lambda_.DockerImageFunction(
                self, fid, code=image(cmd), architecture=lambda_.Architecture.X86_64, memory_size=1024,
                timeout=timeout, environment={**env, **(extra_env or {})}, tracing=lambda_.Tracing.ACTIVE,
                log_group=log_group)
            cluster.grant_data_api_access(fn)
            return fn

        bedrock_invoke = iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=["arn:aws:bedrock:*::foundation-model/*",
                       f"arn:aws:bedrock:*:{self.account}:inference-profile/*"])

        api_fn = function("ApiFunction", "knowledgemcp.api.handler", Duration.seconds(29),
                          {"API_KEYS_SECRET_ARN": api_keys.secret_arn})
        api_fn.add_to_role_policy(bedrock_invoke)
        api_fn.add_to_role_policy(iam.PolicyStatement(actions=["bedrock:Rerank"], resources=["*"]))
        api_keys.grant_read(api_fn)

        ingest_fn = function("IngestFunction", "knowledgemcp.handlers_ingest.ingest_handler", Duration.minutes(5))
        ingest_fn.add_to_role_policy(bedrock_invoke)
        bucket.grant_read(ingest_fn)
        for event in (s3.EventType.OBJECT_CREATED, s3.EventType.OBJECT_REMOVED):
            bucket.add_event_notification(event, s3n.LambdaDestination(ingest_fn))

        # Create tables/indexes after every deploy (idempotent). Waits for the cluster to exist.
        migrate_fn = function("MigrateFunction", "knowledgemcp.handlers_ingest.migrate_handler", Duration.minutes(5))
        triggers.Trigger(self, "MigrateOnDeploy", handler=migrate_fn, execute_after=[cluster],
                         execute_on_handler_change=True, timeout=Duration.minutes(6))

        # --- HTTP API with throttling --------------------------------------------------------------
        http_api = apigw.HttpApi(self, "HttpApi", api_name="knowledgemcp", create_default_stage=False,
                                 default_integration=integrations.HttpLambdaIntegration("ApiIntegration", api_fn))
        stage = apigw.HttpStage(self, "DefaultStage", http_api=http_api, stage_name="$default", auto_deploy=True,
                                throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10))

        # --- Observability: CloudWatch dashboard from the app's EMF metrics -------------------------
        def m(name: str, stat: str = "Sum", label: str | None = None) -> cw.Metric:
            return cw.Metric(namespace=NAMESPACE, metric_name=name, dimensions_map={"service": SERVICE},
                             statistic=stat, period=Duration.minutes(5), label=label or f"{name} ({stat})")

        dashboard = cw.Dashboard(self, "Dashboard", dashboard_name="KnowledgeMCP")
        dashboard.add_widgets(
            cw.GraphWidget(title="Questions and refusals", left=[m("Questions"), m("Refusals"), m("GuardrailBlocks")]),
            cw.GraphWidget(title="Answer latency (ms)", left=[m("QueryLatencyMs", "p50"), m("QueryLatencyMs", "p95")]),
            cw.GraphWidget(title="Tokens", left=[m("InputTokens"), m("OutputTokens")]),
            cw.GraphWidget(title="Model cost (micro-USD)", left=[m("CostMicroUSD")]),
            cw.GraphWidget(title="Ingestion", left=[m("DocumentsIndexed"), m("ChunksIndexed")]),
            cw.GraphWidget(title="Lambda errors", left=[api_fn.metric_errors(), ingest_fn.metric_errors()]),
        )

        # --- Cost guardrail ------------------------------------------------------------------------
        if alert_email:
            budgets.CfnBudget(
                self, "MonthlyBudget",
                budget=budgets.CfnBudget.BudgetDataProperty(
                    budget_name="knowledgemcp-monthly", budget_type="COST", time_unit="MONTHLY",
                    budget_limit=budgets.CfnBudget.SpendProperty(amount=10, unit="USD")),
                notifications_with_subscribers=[
                    budgets.CfnBudget.NotificationWithSubscribersProperty(
                        notification=budgets.CfnBudget.NotificationProperty(
                            notification_type=kind, comparison_operator="GREATER_THAN", threshold=pct,
                            threshold_type="PERCENTAGE"),
                        subscribers=[budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL",
                                                                          address=alert_email)])
                    for kind, pct in (("ACTUAL", 80), ("FORECASTED", 100))
                ],
            )

        CfnOutput(self, "ApiUrl", value=stage.url)
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "ClusterArn", value=cluster.cluster_arn)
        CfnOutput(self, "DbSecretArn", value=cluster.secret.secret_arn)
        CfnOutput(self, "ApiKeysSecretArn", value=api_keys.secret_arn)
        CfnOutput(self, "DashboardUrl", value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                                              f"?region={self.region}#dashboards:name=KnowledgeMCP")
