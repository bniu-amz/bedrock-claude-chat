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
from langchain.vectorstores import SingleStoreDB

client = get_bedrock_client()

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=client)
os.environ["SINGLESTOREDB_URL"] = "admin:amazonPW!@svc-53a789ca-e9f1-48e8-87b0-5ccf8edabdda-dml.aws-oregon-3.svc.singlestore.com:3306/titan_embedding"
vectorstore_s2 = SingleStoreDB(bedrock_embeddings, table_name='shareholder_letter')

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
    
  
    query = chat_input.message.content.body

    docs = vectorstore_s2.similarity_search_with_relevance_scores(query)
    contexts ='';
    prompt = '';
  
    for doc, score in docs:
      if score < -100:
        contexts += " "+doc.page_content

    if(contexts != ''):
      prompt = """
      Use the following pieces of context to provide a concise answer to the question at the end. ignore the context if it's not applicable.
      """ + contexts + """
      Question: """ + query
    else:
      prompt = query
      
    logger.debug("invoke bedrock prompt: " + prompt)    
  
    chat_input.message.content.body = prompt
  
    payload = get_invoke_payload(conversation, chat_input)  
    logger.debug(f"invoke bedrock payload: {payload}")

    try:
        # Invoke bedrock streaming api
        response = client.invoke_model_with_response_stream(
            body=payload["body"],
            modelId=payload["model_id"],
            accept=payload["accept"],
            contentType=payload["content_type"],
        )
    except Exception as e:
        print(f"Failed to invoke bedrock: {e}")
        return {"statusCode": 500, "body": "Failed to invoke bedrock."}

    stream = response.get("body")
    completions = []
    for chunk in generate_chunk(stream):
        try:
            # Send completion
            gatewayapi.post_to_connection(ConnectionId=connection_id, Data=chunk)
            chunk_data = json.loads(chunk.decode("utf-8"))
            completions.append(chunk_data["completion"])
        except Exception as e:
            print(f"Failed to post message: {str(e)}")
            return {"statusCode": 500, "body": "Failed to send message to connection."}

    concatenated = "".join(completions)
  
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
