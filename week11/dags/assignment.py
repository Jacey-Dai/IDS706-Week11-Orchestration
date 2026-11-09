# Register only the assignment DAG in the smaller Week 11 development stack.
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("week11_assignment_dag", "/repo/airflow/dags/week11_fraud_pipeline.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
week11_fraud_pipeline = module.week11_fraud_pipeline
