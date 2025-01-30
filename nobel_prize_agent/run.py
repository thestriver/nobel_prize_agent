import logging
import os
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.inference import InferenceClient
from naptha_sdk.schemas import AgentDeployment, AgentRunInput, KBRunInput
from nobel_prize_agent.schemas import InputSchema, SystemPromptSchema
from naptha_sdk.user import sign_consumer_id

load_dotenv()
logger = logging.getLogger(__name__)

class NobelPrizeAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.nobel_kb = KnowledgeBase(kb_deployment=self.deployment.kb_deployments[0])
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.inference_provider = InferenceClient(self.deployment.node)

    async def run_nobel_agent(self, module_run: AgentRunInput):
        logger.info("Checking if knowledge base exists")

        # First make sure Nobel Prize KB exists
        kb_run_input = KBRunInput(
            consumer_id=module_run.consumer_id,
            inputs={
                "func_name": "init",
                "func_input_data": None
            },
            deployment=self.deployment.kb_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, os.getenv("PRIVATE_KEY"))
        )
        result = await self.nobel_kb.call_kb_func(kb_run_input)
        logger.info(f"KB run result: {result}")

        # Now run the query
        kb_run_input = KBRunInput(
            consumer_id=module_run.consumer_id,
            inputs={
                "func_name": "run_query",
                "func_input_data": {"query": module_run.inputs.query}
            },
            deployment=self.deployment.kb_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, os.getenv("PRIVATE_KEY"))
        )
        
        laureate_info = await self.nobel_kb.call_kb_func(kb_run_input)
        
        if not laureate_info:
            return {"error": "Laureate not found"}
            
        logger.info(f"Laureate information: {laureate_info}")

        # Construct message for LLM
        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {
                "role": "user",
                "content": f"""The user asked: {module_run.inputs.question}
                Nobel laureate information: {laureate_info}
                
                Please provide an answer based on the Nobel laureate information. Include:
                1. When they won the prize
                2. Their contribution (motivation)
                3. Relevant biographical details
                
                Answer the specific question while providing this context."""
            }
        ]
        
        logger.info(f"Messages: {messages}")
        
        # Get response from LLM
        llm_response = await self.inference_provider.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": messages,
            "temperature": self.deployment.config.llm_config.temperature,
            "max_tokens": self.deployment.config.llm_config.max_tokens
        })

        return llm_response.choices[0].message.content

async def run(module_run: Dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    nobel_prize_agent = NobelPrizeAgent(module_run.deployment)
    answer = await nobel_prize_agent.run_nobel_agent(module_run)
    return answer

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()
    
    deployment = asyncio.run(setup_module_deployment(
        "agent",
        "nobel_prize_agent/configs/deployment.json",
        node_url = os.getenv("NODE_URL"),
        user_id=naptha.user.id
    ))

    # Example query
    query = "Hinton"
    question = "What did Geoffrey Hinton win the Nobel Prize for and what is the significance of his work?"

    input_params = {
        "func_name": "run_query",
        "query": query,
        "question": question,
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))
    print(response)