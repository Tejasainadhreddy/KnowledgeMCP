#!/usr/bin/env python3
"""CDK entry point.

    cd infra
    cdk deploy -c alert_email=you@example.com
"""

import os

import aws_cdk as cdk
from knowledgemcp_stack import KnowledgeMcpStack

app = cdk.App()
KnowledgeMcpStack(
    app, "KnowledgeMcpStack",
    alert_email=app.node.try_get_context("alert_email"),
    llm_model_id=app.node.try_get_context("llm_model_id") or "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                        region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2")),
)
app.synth()
