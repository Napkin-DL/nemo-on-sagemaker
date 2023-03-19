"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor,
)
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    CacheConfig,
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from pipelines.nemo_asr.config.config import config_handler
from utils.ssm import parameter_store
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables

from time import strftime
from smexperiments.trial import Trial
from smexperiments.experiment import Experiment
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum, SageMakerJobExceptionTypeEnum, SageMakerJobStepRetryPolicy
from sagemaker.workflow.step_collections import RegisterModel


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

class sm_pipeline():
    
    def __init__(self, region, sagemaker_project_arn, role, default_bucket, \
                 model_package_group_name, base_job_prefix, \
                 input_data_path):
        
        self.pipeline_config = config_handler(strConfigPath="config-pipeline.ini")
        
        
        self.region=region
        self.default_bucket = default_bucket
        self.base_job_prefix = base_job_prefix
        self.model_package_group_name = model_package_group_name
        self.pm = parameter_store(self.region)
        if role is None: self.role = sagemaker.session.get_execution_role(sagemaker_session)
        else: self.role=role
        
        self.sagemaker_session = get_session(self.region, default_bucket)
        self.pipeline_session = get_pipeline_session(region, default_bucket)
        self.pipeline_name = self.pipeline_config.get_value("PIPELINE", "name") #pipeline_name        
        self.cache_config = CacheConfig(
            enable_caching=self.pipeline_config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.pipeline_config.get_value("PIPELINE", "expire_after")
        )    
        
        self.pipeline_config.set_value("PREPROCESSING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        self.pipeline_config.set_value("TRAINING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        self.pipeline_config.set_value("EVALUATION", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        
        self.model_approval_status ="PendingManualApproval"
        # self.model_approval_status = ParameterString(
        #     name="ModelApprovalStatus",
        #     default_value="PendingManualApproval"
        # )
        
        self.proc_prefix = "/opt/ml/processing"        
        self.input_data_path = input_data_path
        
        self.experiment_name = ''.join([self.base_job_prefix, "nemo-asr-exp"])
    
    def create_trial(self, experiment_name):
        
        create_date = strftime("%m%d-%H%M%s")
        sm_trial = Trial.create(trial_name=f'{experiment_name}-{create_date}',
                                experiment_name=experiment_name)
        job_name = f'{sm_trial.trial_name}'
        return job_name
    
    def create_experiment(self, experiment_name):
        try:
            sm_experiment = Experiment.load(experiment_name)
        except:
            sm_experiment = Experiment.create(experiment_name=experiment_name)
            
        return sm_experiment
    
    def _step_preprocessing(self, ):
        
        dataset_processor = FrameworkProcessor(
            estimator_cls=PyTorch,
            framework_version=None,
            image_uri=self.pipeline_config.get_value("PREPROCESSING", "image_uri"),
            instance_type=self.pipeline_config.get_value("PREPROCESSING", "instance_type"), #self.processing_instance_type,
            instance_count=self.pipeline_config.get_value("PREPROCESSING", "instance_count", dtype="int"), #self.processing_instance_count,
            role=self.role,
            base_job_name=f"{self.base_job_prefix}/preprocessing", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session
        )
        
        step_args = dataset_processor.run(
            job_name="preprocessing", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./preprocessing.py', #소스 디렉토리 안에서 파일 path
            source_dir="./an4_nemo_sagemaker/code/preprocessing/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            inputs=[
                ProcessingInput(
                    input_name="input-data",
                    source=self.input_data_path,
                    destination=os.path.join(self.proc_prefix, "input")
                ),
            ],
            outputs=[       
                ProcessingOutput(
                    output_name="output-data",
                    source=os.path.join(self.proc_prefix, "output"),
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.default_bucket),
                            self.pipeline_name,
                            #ExecutionVariables.PROCESSING_JOB_NAME,
                            "preprocessing",
                            "output-data"
                        ],
                    )
                ),
            ],
            
            arguments=["--proc_prefix", self.proc_prefix, "--region", self.region , \
                       "--train_mount_dir", "/opt/ml/input/data/training/", \
                       "--test_mount_dir", "/opt/ml/input/data/testing/"],
        )
        
        self.preprocessing_process = ProcessingStep(
            name="PreprocessingProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Preprocessing Step ==")
        print ("   \nArgs: ", self.preprocessing_process.arguments.items())
        
    def _step_preprocessing_2(self, ):
        
        dataset_processor = FrameworkProcessor(
            estimator_cls=PyTorch,
            framework_version=None,
            image_uri=self.pipeline_config.get_value("PREPROCESSING", "image_uri"),
            instance_type=self.pipeline_config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.pipeline_config.get_value("PREPROCESSING", "instance_count", dtype="int"), 
            role=self.role,
            base_job_name=f"{self.base_job_prefix}/preprocessing-2", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=self.pipeline_session
        )
        
        step_args = dataset_processor.run(
            job_name="preprocessing-2", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='./preprocessing.py', #소스 디렉토리 안에서 파일 path
            source_dir="./an4_nemo_sagemaker/code/preprocessing/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            inputs=[
                ProcessingInput(
                    input_name="input-data",
                    source=self.input_data_path,
                    destination=os.path.join(self.proc_prefix, "input")
                ),
            ],
            outputs=[       
                ProcessingOutput(
                    output_name="output-data-2",
                    source=os.path.join(self.proc_prefix, "output"),
                    destination=Join(
                        on="/",
                        values=[
                            "s3://{}".format(self.default_bucket),
                            self.pipeline_name,
                            #ExecutionVariables.PROCESSING_JOB_NAME,
                            "preprocessing",
                            "output-data-2"
                        ],
                    )
                ),
            ],
            
            arguments=["--proc_prefix", self.proc_prefix, "--region", self.region , \
                       "--train_mount_dir", "/opt/ml/input/data/training/", \
                       "--test_mount_dir", "/opt/ml/input/data/testing/"],
        )
        
        self.preprocessing_process_2 = ProcessingStep(
            name="PreprocessingProcess-2", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Preprocessing Step 2 ==")
        print ("   \nArgs: ", self.preprocessing_process_2.arguments.items())
    
    def _step_training(self, ):
        
        self.estimator = PyTorch(
            entry_point="speech_to_text_ctc.py", # the script we want to run
            source_dir="./an4_nemo_sagemaker/code/training/", # where our conf/script is
            role=self.role,
            instance_type=self.pipeline_config.get_value("TRAINING", "instance_type"),
            instance_count=self.pipeline_config.get_value("TRAINING", "instance_count", dtype="int"), 
            image_uri=self.pipeline_config.get_value("TRAINING", "image_uri"),
            volume_size=1024,
            output_path=Join(
                on="/",
                values=[
                    "s3://{}".format(self.default_bucket),
                    self.pipeline_name,
                    #ExecutionVariables.TRAINING_JOB_NAME,
                    "training",
                    "model-output"
                ],
            ),
            disable_profiler=True,
            debugger_hook_config=False,
            hyperparameters={'config-path': 'conf'},
            #distribution={"smdistributed":{"dataparallel":{"enabled":True, "fp16": True}}},
            sagemaker_session=self.pipeline_session,
            checkpoint_s3_uri = Join(
                on="/",
                values=[
                    "s3://{}".format(self.default_bucket),
                    self.pipeline_name,
                    #ExecutionVariables.TRAINING_JOB_NAME,
                    "training",
                    "ckpt"
                ],
            ),
            metric_definitions=[
                {"Name": "train_loss", "Regex": "loss.*=\D*(.*?)$"}
            ],
            max_run=1*60*60,
        )
        
        sm_experiment = self.create_experiment(self.experiment_name)
        job_name = self.create_trial(self.experiment_name)

        step_training_args = self.estimator.fit(
            inputs={
                "training":self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["output-data"].S3Output.S3Uri,
                "testing":self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["output-data"].S3Output.S3Uri,
            }, 
            job_name=job_name,
            experiment_config={
              'TrialName': job_name,
              'TrialComponentDisplayName': job_name,
            },
            logs="All",
        )

        self.training_process = TrainingStep(
            name="TrainingProcess",
            step_args=step_training_args,
            cache_config=self.cache_config,
            depends_on=[self.preprocessing_process, self.preprocessing_process_2],
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Training Step ==")
        print ("   \nArgs: ", self.training_process.arguments.items())
        
    def _step_evaluation(self, ):
        
        eval_processor = FrameworkProcessor(
            estimator_cls=PyTorch,
            framework_version=None,
            role=self.role, 
            image_uri=self.pipeline_config.get_value("EVALUATION", "image_uri"),
            instance_type=self.pipeline_config.get_value("EVALUATION", "instance_type"),
            instance_count=self.pipeline_config.get_value("EVALUATION", "instance_count", dtype="int"),
            env={
                'TEST_MANIFEST_PATH': '/opt/ml/input/data/testing/an4/wav', 
                'WAV_PATH' : '/opt/ml/processing/input/wav'
            },
            sagemaker_session=self.pipeline_session,
            base_job_name=f"{self.base_job_prefix}/evaluation", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
        )
        
        sm_experiment = self.create_experiment(self.experiment_name)
        job_name = self.create_trial(self.experiment_name)
        
        self.evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation-metrics", ## evaluation의 ProcessingOutput의 output_name
            path="evaluation.json", ## evaluate.py 에서 write하는 부분
        )
        
        step_args = eval_processor.run(
            code="evaluate.py",
            source_dir="./an4_nemo_sagemaker/code/evaluation/",
            inputs=[
                ProcessingInput(
                    source=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
                    input_name="model_artifact",
                    destination=os.path.join(self.proc_prefix, "model")#  "/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=Join(
                        on="/",
                        values=[
                            self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["output-data"].S3Output.S3Uri,
                            "an4",
                            "test_manifest.json",
                        ],
                    ),
                    input_name="test_manifest_file",
                    destination=os.path.join(self.proc_prefix, "input", "manifest") #"/opt/ml/processing/input/manifest"
                ),
                ProcessingInput(
                    source=Join(
                        on="/",
                        values=[
                            self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["output-data"].S3Output.S3Uri,
                            "an4",
                            "wav",
                        ],
                    ),
                    input_name="wav_dataset",
                    destination=os.path.join(self.proc_prefix, "input", "wav") #"/opt/ml/processing/input/wav"
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-metrics",
                    source=os.path.join(self.proc_prefix, "evaluation"), #"/opt/ml/processing/evaluation",
                    destination=os.path.join(
                        "s3://",
                        self.pm.get_params(key=self.base_job_prefix + 'BUCKET'),
                        self.pipeline_name,
                        "evaluation",
                        "output",
                        "evaluation-metrics"
                    ),
                ),
            ],
            job_name=job_name,
            experiment_config={
              'TrialName': job_name,
              'TrialComponentDisplayName': job_name,
            },
        )
        
        self.evaluation_process = ProcessingStep(
            name="EvaluationProcess", ## Processing job이름
            step_args=step_args,
            depends_on=[self.training_process],
            property_files=[self.evaluation_report],
            cache_config=self.cache_config,
            retry_policies=[                
                # retry when resource limit quota gets exceeded
                SageMakerJobStepRetryPolicy(
                    exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                    expire_after_mins=180,
                    interval_seconds=600,
                    backoff_rate=1.0
                ),
            ]
        )
        
        print ("  \n== Evaluation Step ==")
        print ("   \nArgs: ", self.evaluation_process.arguments.items())
        
    def _step_model_registration(self, ):
      
        #self.model_package_group_name = ''.join([self.prefix, self.model_name])
        self.pm.put_params(key=self.base_job_prefix + "MODEL-GROUP-NAME", value=self.model_package_group_name, overwrite=True)
                                                                              
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                        #print (self.evaluation_process.arguments.items())로 확인가능
                        "evaluation-metrics.json"
                    ],
                ),
                content_type="application/json")
        )
                                       
        self.register_process = RegisterModel(
            name="ModelRegisterProcess", ## Processing job이름
            estimator=self.estimator,
            image_uri=self.training_process.properties.AlgorithmSpecification.TrainingImage, 
            model_data=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=self.pipeline_config.get_value("MODEL_REGISTER", "inference_instances", dtype="list"),
            transform_instances=self.pipeline_config.get_value("MODEL_REGISTER", "transform_instances", dtype="list"),
            model_package_group_name=self.model_package_group_name,
            approval_status=self.pipeline_config.get_value("MODEL_REGISTER", "model_approval_status_default"),
            ## “Approved”, “Rejected”, or “PendingManualApproval” (default: “PendingManualApproval”).
            model_metrics=model_metrics,
            depends_on=[self.evaluation_process]
        )
        
        print ("  \n== Registration Step ==")

    def _execution_steps(self, ):
        
        self._step_preprocessing()
        self._step_preprocessing_2()
        self._step_training()
        self._step_evaluation()
        self._step_model_registration()
        
    def get_pipeline(self, ):
        
        self._execution_steps()
        
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.preprocessing_process, self.preprocessing_process_2, \
                   self.training_process, \
                   self.evaluation_process, \
                   self.register_process, \
            ],
            sagemaker_session=self.pipeline_session
        )

        return pipeline
        
        
      
        







def get_pipeline(
    region,
    input_data_path,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    base_job_prefix="Abalone",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    #sagemaker_session = get_session(region, default_bucket)
    #if role is None:
    #    role = sagemaker.session.get_execution_role(sagemaker_session)

    #pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    #processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    # model_approval_status = ParameterString(
    #     name="ModelApprovalStatus", default_value="PendingManualApproval"
    # )
    #input_data = ParameterString(
    #    name="InputDataUrl",
    #    default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    #)

    # processing step for feature engineering
    # sklearn_processor = SKLearnProcessor(
    #     framework_version="0.23-1",
    #     instance_type=processing_instance_type,
    #     instance_count=processing_instance_count,
    #     base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
    #     sagemaker_session=pipeline_session,
    #     role=role,
    # )
    # step_args = sklearn_processor.run(
    #     outputs=[
    #         ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
    #         ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
    #         ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    #     ],
    #     code=os.path.join(BASE_DIR, "preprocess.py"),
    #     arguments=["--input-data", input_data],
    # )
    # step_process = ProcessingStep(
    #     name="PreprocessAbaloneData",
    #     step_args=step_args,
    # )

    # training step for generating model artifacts
    
    
    
#     model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
#     image_uri = sagemaker.image_uris.retrieve(
#         framework="xgboost",
#         region=region,
#         version="1.0-1",
#         py_version="py3",
#         instance_type=training_instance_type,
#     )
#     xgb_train = Estimator(
#         image_uri=image_uri,
#         instance_type=training_instance_type,
#         instance_count=1,
#         output_path=model_path,
#         base_job_name=f"{base_job_prefix}/abalone-train",
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     xgb_train.set_hyperparameters(
#         objective="reg:linear",
#         num_round=50,
#         max_depth=5,
#         eta=0.2,
#         gamma=4,
#         min_child_weight=6,
#         subsample=0.7,
#         silent=0,
#     )
#     step_args = xgb_train.fit(
#         inputs={
#             "train": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "train"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#             "validation": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "validation"
#                 ].S3Output.S3Uri,
#                 content_type="text/csv",
#             ),
#         },
#     )
#     step_train = TrainingStep(
#         name="TrainAbaloneModel",
#         step_args=step_args,
#     )

#     # processing step for evaluation
#     script_eval = ScriptProcessor(
#         image_uri=image_uri,
#         command=["python3"],
#         instance_type=processing_instance_type,
#         instance_count=1,
#         base_job_name=f"{base_job_prefix}/script-abalone-eval",
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     step_args = script_eval.run(
#         inputs=[
#             ProcessingInput(
#                 source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#                 destination="/opt/ml/processing/model",
#             ),
#             ProcessingInput(
#                 source=step_process.properties.ProcessingOutputConfig.Outputs[
#                     "test"
#                 ].S3Output.S3Uri,
#                 destination="/opt/ml/processing/test",
#             ),
#         ],
#         outputs=[
#             ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
#         ],
#         code=os.path.join(BASE_DIR, "evaluate.py"),
#     )
#     evaluation_report = PropertyFile(
#         name="AbaloneEvaluationReport",
#         output_name="evaluation",
#         path="evaluation.json",
#     )
#     step_eval = ProcessingStep(
#         name="EvaluateAbaloneModel",
#         step_args=step_args,
#         property_files=[evaluation_report],
#     )

#     # register model step that will be conditionally executed
#     model_metrics = ModelMetrics(
#         model_statistics=MetricsSource(
#             s3_uri="{}/evaluation.json".format(
#                 step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
#             ),
#             content_type="application/json"
#         )
#     )
#     model = Model(
#         image_uri=image_uri,
#         model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     step_args = model.register(
#         content_types=["text/csv"],
#         response_types=["text/csv"],
#         inference_instances=["ml.t2.medium", "ml.m5.large"],
#         transform_instances=["ml.m5.large"],
#         model_package_group_name=model_package_group_name,
#         approval_status=model_approval_status,
#         model_metrics=model_metrics,
#     )
#     step_register = ModelStep(
#         name="RegisterAbaloneModel",
#         step_args=step_args,
#     )

#     # condition step for evaluating model quality and branching execution
#     cond_lte = ConditionLessThanOrEqualTo(
#         left=JsonGet(
#             step_name=step_eval.name,
#             property_file=evaluation_report,
#             json_path="regression_metrics.mse.value"
#         ),
#         right=6.0,
#     )
#     step_cond = ConditionStep(
#         name="CheckMSEAbaloneEvaluation",
#         conditions=[cond_lte],
#         if_steps=[step_register],
#         else_steps=[],
#     )
    

    nemo_asr_pipeline = sm_pipeline(
        region=region,
        sagemaker_project_arn=sagemaker_project_arn,
        role=role,
        default_bucket=default_bucket,
        model_package_group_name=model_package_group_name,
        base_job_prefix=base_job_prefix,
        input_data_path=input_data_path)
    
    return nemo_asr_pipeline.get_pipeline()
    
    
    
    # # pipeline instance
    # pipeline = Pipeline(
    #     name=pipeline_name,
    #     parameters=[
    #         processing_instance_type,
    #         processing_instance_count,
    #         training_instance_type,
    #         model_approval_status,
    #         input_data,
    #     ],
    #     steps=[step_process, step_train, step_eval, step_cond],
    #     sagemaker_session=pipeline_session,
    # )
    # return pipeline
