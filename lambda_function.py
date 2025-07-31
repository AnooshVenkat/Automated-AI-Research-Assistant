import os
import json
import boto3
import uuid
import datetime
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.utilities import SerpAPIWrapper

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY') 
def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    if not GOOGLE_API_KEY:
        print("ERROR: Environment variable GOOGLE_API_KEY is not set.")
        return {'statusCode': 500, 'body': json.dumps({'error': 'Server configuration error: Missing Google API Key.'})}
    if not SERPAPI_API_KEY:
        print("ERROR: Environment variable SERPAPI_API_KEY is not set.")
        return {'statusCode': 500, 'body': json.dumps({'error': 'Server configuration error: Missing SerpApi API Key.'})}
    if not S3_BUCKET_NAME or not DYNAMODB_TABLE_NAME:
        print("ERROR: S3_BUCKET_NAME or DYNAMODB_TABLE_NAME is not set.")
        return {'statusCode': 500, 'body': json.dumps({'error': 'Server configuration error: Missing AWS resource names.'})}


    try:
        body = json.loads(event.get('body', '{}'))
        research_topic = body.get('topic')
        if not research_topic:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing "topic" in request body.'})
            }
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body.'})
        }

    task_id = str(uuid.uuid4())
    print(f"Starting task {task_id} for topic: '{research_topic}'")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="WebSearch",
                func=search.run,
                description="Useful for when you need to answer questions about current events or find information on the internet."
            )
        ]

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        prompt = f"""
        You are a diligent AI Research Assistant. Your task is to investigate the following topic: "{research_topic}".
        
        Follow these steps:
        1. Use the WebSearch tool to find relevant information.
        2. Synthesize the information you find into a concise, well-structured report.
        3. The final report should be your final answer. Do not include your thought process, just the report itself.
        
        Begin your work now.
        """
        final_report = agent.run(prompt)
        print(f"Agent finished. Final report generated.")
        s3_file_key = f"reports/{task_id}.txt"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_file_key,
            Body=final_report.encode('utf-8'),
            ContentType='text/plain'
        )
        print(f"Report saved to S3: s3://{S3_BUCKET_NAME}/{s3_file_key}")

        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        table.put_item(
            Item={
                'task_id': task_id,
                'research_topic': research_topic,
                's3_report_key': s3_file_key,
                'status': 'COMPLETED',
                'completed_at': datetime.datetime.utcnow().isoformat()
            }
        )
        print(f"Task metadata saved to DynamoDB table: {DYNAMODB_TABLE_NAME}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Research task completed successfully!',
                'task_id': task_id,
                's3_bucket': S3_BUCKET_NAME,
                's3_key': s3_file_key
            })
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"An error occurred: {e}")
        print("Full traceback:")
        print(error_trace)
        
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        table.put_item(
            Item={
                'task_id': task_id,
                'research_topic': research_topic,
                'status': 'FAILED',
                'error_message': error_trace, 
                'completed_at': datetime.datetime.utcnow().isoformat()
            }
        )
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'An internal server error occurred. Check CloudWatch logs for details.'})
        }

