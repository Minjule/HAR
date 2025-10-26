import mlflow

mlflow.create_experiment("EXP_1")
mlflow.set_experiment_tag("scikit-learn", "lr")
mlflow.set_experiment("EXP_1")
mlflow.start_run(experiment_id="1", run_name="RUN_1")
mlflow.log_metric("accuracy", 0.95)
mlflow.log_artifact("s.py")
mlflow.end_run()