from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    RemovalPolicy,
    CfnOutput,
    Duration
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

        # --- ECS Task Role ---
        # Role that the ECS task will assume to interact with other AWS services
        self.task_role = iam.Role(
            self, "TradingBotTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Role for Trading Bot ECS tasks"
        )
        # Add permissions as needed, e.g., to read from S3 if checkpoints are there
        # self.task_role.add_to_policy(iam.PolicyStatement(
        #     actions=["s3:GetObject"],
        #     resources=["arn:aws:s3:::your-checkpoint-bucket/*"]
        # ))

        # --- ECS Task Definition for Fargate ---
        self.fargate_task_definition = ecs.FargateTaskDefinition(
            self, "TradingBotTaskDef",
            memory_limit_mib=512,  # 0.5 GB
            cpu=256,              # 0.25 vCPU
            task_role=self.task_role,
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.X86_64 # Or ARM64 if your image is ARM
            )
        )

        # Define the Log Group for the container
        log_group = logs.LogGroup(
            self, "TradingBotApiLogGroup",
            log_group_name=f"/ecs/{self.ecs_cluster.cluster_name}/trading-bot-api",
            retention=logs.RetentionDays.ONE_MONTH, # Adjust as needed
            removal_policy=RemovalPolicy.DESTROY
        )

        self.api_container = self.fargate_task_definition.add_container(
            "TradingBotApiContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.api_ecr_repository, tag="latest"), # Assumes image tagged 'latest'
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="api",
                log_group=log_group
            ),
            port_mappings=[ecs.PortMapping(container_port=8000)],
            environment={
                "API_CHECKPOINT_TO_LOAD_PATH": "/app/rllib_checkpoints/your_checkpoint_file_path_here", # Placeholder!
                "PYTHONUNBUFFERED": "1", # Ensures logs are sent out immediately
                # Add other necessary environment variables
            }
        )

        # --- Fargate Service ---
        # This service will run our task definition on Fargate.
        self.fargate_service = ecs.FargateService(
            self, "TradingBotFargateService",
            cluster=self.ecs_cluster,
            task_definition=self.fargate_task_definition,
            desired_count=1,  # Start with one instance
            assign_public_ip=True, # Assign public IP for direct access (for testing). For prod, use ALB.
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            # health_check_grace_period=Duration.seconds(60) # If your app takes time to start health checks
        )

        CfnOutput(self, "FargateServiceName", value=self.fargate_service.service_name)
        # Note: To access the service, you'd get the public IP of the running task.
        # An ALB would provide a stable DNS endpoint.

        # --- Next steps: Application Load Balancer ---
        # Example: VPC, ECS Cluster, Fargate Service, Load Balancer 