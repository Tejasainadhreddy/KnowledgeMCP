"""Synthesize the stack (no AWS calls, no Docker build) and check the cost and security choices."""

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from knowledgemcp_stack import KnowledgeMcpStack


def synth(**kw) -> Template:
    app = cdk.App()
    stack = KnowledgeMcpStack(app, "Test", env=cdk.Environment(account="123456789012", region="us-west-2"), **kw)
    return Template.from_stack(stack)


def test_aurora_scales_to_zero_with_data_api():
    t = synth()
    t.has_resource_properties("AWS::RDS::DBCluster", {
        "Engine": "aurora-postgresql",
        "EnableHttpEndpoint": True,
        "StorageEncrypted": True,
        "ServerlessV2ScalingConfiguration": Match.object_like({"MinCapacity": 0, "MaxCapacity": 2}),
    })


def test_no_nat_gateway():
    synth().resource_count_is("AWS::EC2::NatGateway", 0)


def test_three_container_lambdas_with_tracing():
    t = synth()
    fns = t.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})
    assert len(fns) == 3
    assert all(f["Properties"]["TracingConfig"]["Mode"] == "Active" for f in fns.values())


def test_bucket_is_private_and_triggers_ingest():
    t = synth()
    t.has_resource_properties("AWS::S3::Bucket", {"PublicAccessBlockConfiguration": {
        "BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}})
    t.resource_count_is("Custom::S3BucketNotifications", 1)


def test_api_is_throttled():
    synth().has_resource_properties("AWS::ApiGatewayV2::Stage", {
        "StageName": "$default", "DefaultRouteSettings": {"ThrottlingBurstLimit": 10, "ThrottlingRateLimit": 5}})


def test_budget_only_when_email_given():
    synth().resource_count_is("AWS::Budgets::Budget", 0)
    synth(alert_email="me@example.com").has_resource_properties("AWS::Budgets::Budget", {
        "Budget": Match.object_like({"BudgetLimit": {"Amount": 10, "Unit": "USD"}})})


def test_dashboard_exists():
    synth().resource_count_is("AWS::CloudWatch::Dashboard", 1)
