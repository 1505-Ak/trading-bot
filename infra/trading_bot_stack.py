from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    RemovalPolicy
)
from constructs import Construct

class TradingBotStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- ECR Repository for the Trading Bot API ---
        # This repository will store the Docker images built from our Dockerfile.
        self.api_ecr_repository = ecr.Repository(
            self, "TradingBotApiRepository",
            repository_name="trading-bot-api", # Name of the ECR repository
            image_scan_on_push=True,  # Automatically scan images for vulnerabilities
            removal_policy=RemovalPolicy.DESTROY # DESTROY will delete the repo when stack is deleted.
                                                 # RETAIN (default) would keep it. Choose based on your needs.
                                                 # For dev/testing, DESTROY is often fine.
        )

        # Output the ECR repository URI
        # CfnOutput(self, "ApiEcrRepoUri", value=self.api_ecr_repository.repository_uri)
        # Using the below for more direct access to the value if needed by other tools/scripts
        self.ecr_repo_uri = self.api_ecr_repository.repository_uri

        # --- Next steps would be to define ECS/Fargate services, ALB, etc. ---
        # Example: VPC, ECS Cluster, Fargate Service, Load Balancer 