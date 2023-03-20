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
                 pipeline_name, model_package_group_name, base_job_prefix, \
                ):
        
        self.pipeline_config = config_handler(strConfigPath="config-pipeline.ini")
        
        
        self.region=region
        self.default_bucket = default_bucket
        self.model_package_group_name = model_package_group_name
        self.pm = parameter_store(self.region)
        if role is None: self.role = sagemaker.session.get_execution_role(sagemaker_session)
        else: self.role=role
        if pipeline_name is None: self.pipeline_name = self.pipeline_config.get_value("PIPELINE", "name")
        else: self.pipeline_name = pipeline_name
        if base_job_prefix is None: self.base_job_prefix = self.pipeline_config.get_value("COMMON", "base_job_prefix")
        else: self.base_job_prefix = base_job_prefix
        
        
        self.sagemaker_session = get_session(self.region, default_bucket)
        self.pipeline_session = get_pipeline_session(region, default_bucket)
        
        self.cache_config = CacheConfig(
            enable_caching=self.pipeline_config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.pipeline_config.get_value("PIPELINE", "expire_after")
        )    
        
        self.pipeline_config.set_value("PREPROCESSING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        self.pipeline_config.set_value("TRAINING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        self.pipeline_config.set_value("EVALUATION", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        
        #self.model_approval_status = self.pipeline_config.get_value("MODEL_REGISTER", "model_approval_status_default") #"PendingManualApproval"
        
        
        # self.model_approval_status = ParameterString(
        #     name="ModelApprovalStatus",
        #     default_value="PendingManualApproval"
        # )
        
        self.proc_prefix = "/opt/ml/processing"        
        self.input_data_path = self.pipeline_config.get_value("INPUT", "input_data_s3_uri") 
        
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
        
        num_re = "([0-9\\.]+)(e-?[[01][0-9])?"
        
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
            metric_definitions = [
                {"Name": "train_loss", "Regex": f"loss={num_re}"},
                {"Name": "wer", "Regex": f"wer:{num_re}"}
            ],
            enable_sagemaker_metrics=True,
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
                        "evaluation.json"
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
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    pipeline_name=None,
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

    nemo_asr_pipeline = sm_pipeline(
        region=region,
        sagemaker_project_arn=sagemaker_project_arn,
        role=role,
        default_bucket=default_bucket,
        pipeline_name=pipeline_name,
        model_package_group_name=model_package_group_name,
        base_job_prefix=base_job_prefix
    )
    
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
