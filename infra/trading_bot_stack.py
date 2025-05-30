from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_ecs_patterns as ecs_patterns,
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

        # Define the Log Group for the container
        log_group = logs.LogGroup(
            self, "TradingBotApiLogGroup",
            log_group_name=f"/ecs/{self.ecs_cluster.cluster_name}/trading-bot-api",
            retention=logs.RetentionDays.ONE_MONTH, # Adjust as needed
            removal_policy=RemovalPolicy.DESTROY
        )

        # --- Application Load Balanced Fargate Service ---
        # This pattern creates an ALB, Target Group, Fargate Service, and Task Definition.
        self.load_balanced_fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "TradingBotAlbFargateService",
            cluster=self.ecs_cluster,  # Use the existing cluster
            cpu=256,                   # 0.25 vCPU
            memory_limit_mib=512,      # 0.5 GB
            desired_count=1,           # Number of tasks to run
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_ecr_repository(self.api_ecr_repository, tag="latest"),
                container_port=8000, # The port your FastAPI app listens on
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="api-alb",
                    log_group=log_group
                ),
                environment={
                    "API_CHECKPOINT_TO_LOAD_PATH": "/app/rllib_checkpoints/your_checkpoint_file_path_here", # Placeholder!
                    "PYTHONUNBUFFERED": "1",
                    # Add other necessary environment variables for your application
                },
                task_role=self.task_role # Assign the task role here
            ),
            public_load_balancer=True,  # Creates a public ALB
            # listener_port=80, # Default is 80 for HTTP. For HTTPS, you'd configure a certificate.
            # redirect_http=True, # If using HTTPS and want to redirect HTTP to HTTPS
            # domain_name="your.domain.com", # If using a custom domain with Route53
            # domain_zone=route53.HostedZone.from_lookup(self, "MyZone", domain_name="your.domain.com"), # If using Route53
            # certificate=acm.Certificate.from_certificate_arn(self, "Cert", "your_cert_arn"), # For HTTPS
        )

        # Health Check configuration for the Target Group
        self.load_balanced_fargate_service.target_group.configure_health_check(
            path="/",  # Assuming your API root path (health check) returns 200 OK
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=2,
            # port="8000" # Traffic port, usually not needed if containerPort is set for task_image_options
        )
        
        # Optional: Add auto-scaling for the Fargate service
        # scaling = self.load_balanced_fargate_service.service.auto_scale_task_count(
        #     max_capacity=3,
        #     min_capacity=1
        # )
        # scaling.scale_on_cpu_utilization("CpuScaling",
        #     target_utilization_percent=70,
        #     scale_in_cooldown=Duration.seconds(60),
        #     scale_out_cooldown=Duration.seconds(60)
        # )

        CfnOutput(self, "LoadBalancerDns", value=self.load_balanced_fargate_service.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "FargateServiceFullName", value=self.load_balanced_fargate_service.service.service_name)

        # Note: The previous self.fargate_task_definition and self.fargate_service are now managed by ApplicationLoadBalancedFargateService

        # --- Next steps: Application Load Balancer ---
        # Example: VPC, ECS Cluster, Fargate Service, Load Balancer 