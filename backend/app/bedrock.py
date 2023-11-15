import json
import logging
import os

from app.config import GENERATION_CONFIG
from app.utils import get_bedrock_client

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.vectorstores import SingleStoreDB

logger = logging.getLogger(__name__)


client = get_bedrock_client()

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=client)
os.environ["SINGLESTOREDB_URL"] = "admin:amazonPW!@svc-53a789ca-e9f1-48e8-87b0-5ccf8edabdda-dml.aws-oregon-3.svc.singlestore.com:3306/titan_embedding"
vectorstore_s2 = SingleStoreDB(bedrock_embeddings, table_name='shareholder_letter')
wrapper_store_s2 = VectorStoreIndexWrapper(vectorstore=vectorstore_s2)

llm = Bedrock(model_id="anthropic.claude-v2", 
              client=client, 
              model_kwargs={
                  'max_tokens_to_sample': 200
              })

def _create_body(model: str, prompt: str):
    if model in ("claude-instant-v1", "claude-v2"):
        parameter = GENERATION_CONFIG
        parameter["prompt"] = prompt
        return json.dumps(parameter)
    else:
        raise NotImplementedError()


def _extract_output_text(model: str, response) -> str:
    if model in ("claude-instant-v1", "claude-v2"):
        output = json.loads(response.get("body").read())
        output_txt = output["completion"]
        if output_txt[0] == " ":
            # claude outputs a space at the beginning of the text
            output_txt = output_txt[1:]
        return output_txt
    else:
        raise NotImplementedError()


def get_model_id(model: str) -> str:
    # Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
    if model == "claude-v2":
        return "anthropic.claude-v2"
    elif model == "claude-instant-v1":
        return "anthropic.claude-instant-v1"
    else:
        raise NotImplementedError()


def invoke(prompt: str, model: str) -> str:
    logger.debug(f"Bedrock Invoke prompt: {prompt}")
    output_txt = wrapper_store_s2.query(question=prompt, llm=llm)
    logger.debug(f"Bedrock Invoke response {output_txt}")
    return output_txt
