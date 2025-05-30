from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    RemovalPolicy,
    CfnOutput
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
        CfnOutput(self, "ApiEcrRepoUri", value=self.api_ecr_repository.repository_uri)
        self.ecr_repo_uri = self.api_ecr_repository.repository_uri

        # --- VPC for our application ---
        # This will create a new VPC with public and private subnets across multiple AZs by default.
        self.vpc = ec2.Vpc(
            self, "TradingBotVpc",
            max_azs=2,  # Limit to 2 AZs for cost/simplicity, can be adjusted
            nat_gateways=1 # Cost optimization: 1 NAT Gateway instead of 1 per AZ. 
                           # For high availability, you might want more.
        )
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)

        # --- ECS Cluster ---
        # This cluster will host our Fargate services.
        self.ecs_cluster = ecs.Cluster(
            self, "TradingBotCluster",
            vpc=self.vpc,
            cluster_name="trading-bot-cluster"
        )
        CfnOutput(self, "EcsClusterName", value=self.ecs_cluster.cluster_name)

        # --- Next steps: ECS Task Definition, Fargate Service, Application Load Balancer ---
        # Example: VPC, ECS Cluster, Fargate Service, Load Balancer 