#!/usr/bin/env python
from dotenv import load_dotenv
from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.utils import get_logger
from module_template.schemas import InputSchema

load_dotenv()

logger = get_logger(__name__)

# You can create your module as a class or function
class BasicModule:
    def __init__(self, module_run):
        self.module_run = module_run

    def func(self, input_data):
        logger.info(f"Running module function")
        return input_data

# Default entrypoint when the module is executed
def run(module_run):
    basic_module = BasicModule(module_run)
    method = getattr(basic_module, module_run.inputs.func_name, None)
    return method(module_run.inputs.func_input_data)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("agent", "module_template/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    input_params = InputSchema(
        func_name="func",
        func_input_data="gm...",
    )

    module_run = AgentRunInput(
        inputs=input_params,
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    response = run(module_run)

    print("Response: ", response)