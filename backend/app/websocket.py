import json
import logging
from datetime import datetime
import os

import boto3
from app.auth import verify_token
from app.repositories.conversation import store_conversation
from app.repositories.model import ContentModel, MessageModel
from app.route_schema import ChatInputWithToken
from app.usecase import get_invoke_payload, prepare_conversation
from app.utils import get_bedrock_client
from ulid import ULID

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.vectorstores import SingleStoreDB

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

client = get_bedrock_client()

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=client)
os.environ["SINGLESTOREDB_URL"] = "admin:amazonPW!@svc-53a789ca-e9f1-48e8-87b0-5ccf8edabdda-dml.aws-oregon-3.svc.singlestore.com:3306/titan_embedding"
vectorstore_s2 = SingleStoreDB(bedrock_embeddings, table_name='shareholder_letter')

llm = Bedrock(model_id="anthropic.claude-v2", 
              client=client, 
              model_kwargs={
                  'max_tokens_to_sample': 200
              })

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_chunk(stream) -> bytes:
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_bytes = chunk.get("bytes")
                yield chunk_bytes


def handler(event, context):
    print(f"Received event: {event}")
    # Extracting the SNS message and its details
    # NOTE: All notification messages will contain a single published message.
    # See `Reliability` section of: https://aws.amazon.com/sns/faqs/
    sns_message = event["Records"][0]["Sns"]["Message"]
    message_content = json.loads(sns_message)

    route_key = message_content["requestContext"]["routeKey"]

    connection_id = message_content["requestContext"]["connectionId"]
    domain_name = message_content["requestContext"]["domainName"]
    stage = message_content["requestContext"]["stage"]
    message = message_content["body"]
    endpoint_url = f"https://{domain_name}/{stage}"
    gatewayapi = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint_url)

    chat_input = ChatInputWithToken(**json.loads(message))
    logger.debug(f"Received chat input: {chat_input}")

    try:
        # Verify JWT token
        decoded = verify_token(chat_input.token)
    except Exception as e:
        print(f"Invalid token: {e}")
        return {"statusCode": 403, "body": "Invalid token."}

    user_id = decoded["sub"]
    user_msg_id, conversation = prepare_conversation(user_id, chat_input)
    payload = get_invoke_payload(conversation, chat_input)

    logger.debug(f"invoke bedrock payload: {payload}")

    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Assistant:"""
  
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_s2.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    query = message.message.content.body
    logger.debug(f"invoke bedrock query: {query}")
    
    concatenated = qa({"query": query})
    

    # Append entire completion as the last message
    assistant_msg_id = str(ULID())
    message = MessageModel(
        role="assistant",
        content=ContentModel(content_type="text", body=concatenated),
        model=chat_input.message.model,
        children=[],
        parent=user_msg_id,
        create_time=datetime.now().timestamp(),
    )
    conversation.message_map[assistant_msg_id] = message
    # Append children to parent
    conversation.message_map[user_msg_id].children.append(assistant_msg_id)
    conversation.last_message_id = assistant_msg_id

    # Persist conversation
    store_conversation(user_id, conversation)

    return {"statusCode": 200, "body": json.dumps({"conversationId": conversation.id})}
